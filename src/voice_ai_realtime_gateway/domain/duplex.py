"""Full-duplex session controller with low-latency barge-in.

The controller binds the :class:`~.machine.SessionStateMachine` to the egress playback
path (``ring -> pump task -> transport``) and to the turn-scoped generation tasks.

Barge-in critical path (:meth:`DuplexSession.on_user_speech_started` while
``PROCESSING_THOUGHT`` / ``SPEAKING``) is **synchronous** up to ``DRAINING``::

    fire(BARGE_IN) -> cancel turn tasks -> abort egress ring -> fire(DRAIN)

No ``await`` occurs between speech detection and ``DRAINING``, so no other coroutine can
interleave and the latency is bounded by O(tasks + queued frames) pure-Python work --
microseconds, far inside :data:`BARGE_IN_BUDGET_NS`. Only *after* ``DRAINING`` does the
controller await task reaping (bounded by ``drain_timeout_s``); tasks that ignore
cancellation are reported as stragglers and fault the session (fail closed).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, final, override

from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy
from voice_ai_realtime_gateway.core.supervisor import JoinReport, TaskSupervisor
from voice_ai_realtime_gateway.domain.errors import SessionStateError, StaleSpanError
from voice_ai_realtime_gateway.domain.machine import SessionStateMachine
from voice_ai_realtime_gateway.domain.states import (
    ACTIVE_OUTPUT_STATES,
    SessionState,
    SessionTrigger,
)
from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    FirstTokenLatency,
    TaskFailed,
    VadSpeechStarted,
    VadSpeechStopped,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.latency import FirstTokenTracker, GlassToGlassTracker

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.domain.machine import Transition
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink

__all__ = [
    "BARGE_IN_BUDGET_NS",
    "DEFAULT_EGRESS_RING_BYTES",
    "DEFAULT_EGRESS_RING_FRAMES",
    "AudioEgress",
    "BargeInReport",
    "DuplexSession",
]

BARGE_IN_BUDGET_NS: Final = 15_000_000
"""Hard budget from detected user speech to ``DRAINING`` (15 ms)."""

DEFAULT_EGRESS_RING_FRAMES: Final = 256
"""5.12 s of 20 ms frames."""

DEFAULT_EGRESS_RING_BYTES: Final = 512 * 1024
"""~2.7 s of 48 kHz mono PCM16, ~65 s of 64 kbit/s Opus."""

_PUMP_TASK_PREFIX: Final = "egress-pump"


class AudioEgress(Protocol):
    """Transport port that delivers agent audio to the remote peer."""

    async def send(self, frame: AudioFrame, /) -> None:
        """Deliver one frame; may suspend for pacing / transport backpressure."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeInReport:
    """Outcome of one barge-in."""

    detect_to_draining_ns: int
    within_budget: bool
    cancelled_tasks: int
    dropped_frames: int
    stragglers: int
    resumed_listening: bool


@final
class DuplexSession:
    """One full-duplex voice session: state machine + egress playback + turn tasks."""

    __slots__ = (
        "_barge_in_budget_ns",
        "_clock",
        "_drain_timeout_s",
        "_egress",
        "_faulted",
        "_first_token",
        "_g2g",
        "_machine",
        "_pending_turn_end",
        "_playback_done",
        "_ring",
        "_sink",
        "_speech_started_at",
        "_tasks",
        "_terminated",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        *,
        egress: AudioEgress,
        sink: TelemetrySink,
        trace: TraceContext | None = None,
        clock: MonotonicClock = time.monotonic_ns,
        ring: AudioRingBuffer | None = None,
        drain_timeout_s: float = 0.1,
        barge_in_budget_ns: int = BARGE_IN_BUDGET_NS,
    ) -> None:
        """Create an ``IDLE`` session.

        Raises:
            ValueError: on a non-positive ``drain_timeout_s`` or ``barge_in_budget_ns``.
        """
        if drain_timeout_s <= 0:
            msg = f"drain_timeout_s must be positive, got {drain_timeout_s}"
            raise ValueError(msg)
        if barge_in_budget_ns <= 0:
            msg = f"barge_in_budget_ns must be positive, got {barge_in_budget_ns}"
            raise ValueError(msg)
        self._egress = egress
        self._sink = sink
        self._clock = clock
        self._drain_timeout_s = drain_timeout_s
        self._barge_in_budget_ns = barge_in_budget_ns
        self._terminated = asyncio.Event()
        self._playback_done = asyncio.Event()
        self._playback_done.set()
        self._faulted = False
        self._machine = SessionStateMachine(
            trace=TraceContext.new() if trace is None else trace,
            sink=sink,
            clock=clock,
            on_transition=self._observe,
        )
        self._ring = (
            AudioRingBuffer(
                capacity_frames=DEFAULT_EGRESS_RING_FRAMES,
                capacity_bytes=DEFAULT_EGRESS_RING_BYTES,
                policy=OverflowPolicy.BACKPRESSURE,
                sink=sink,
            )
            if ring is None
            else ring
        )
        self._tasks = TaskSupervisor(on_failure=self._on_task_failure)
        self._g2g = GlassToGlassTracker(sink)
        self._first_token = FirstTokenTracker(sink)
        self._speech_started_at: int | None = None
        self._pending_turn_end = False

    # ------------------------------------------------------------------ introspection

    @property
    def machine(self) -> SessionStateMachine:
        """The underlying state machine."""
        return self._machine

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._machine.state

    @property
    def trace(self) -> TraceContext:
        """Current correlation context (the current response span)."""
        return self._machine.trace

    @property
    def ring(self) -> AudioRingBuffer:
        """Egress playback ring."""
        return self._ring

    @property
    def glass_to_glass(self) -> GlassToGlassTracker:
        """Per-packet egress latency tracker."""
        return self._g2g

    @property
    def first_token(self) -> FirstTokenTracker:
        """End-of-turn -> first-token latency tracker."""
        return self._first_token

    @property
    def terminated(self) -> bool:
        """Whether the session reached ``TERMINATED``."""
        return self._terminated.is_set()

    @property
    def faulted(self) -> bool:
        """Whether the session terminated through ``FAULT`` rather than ``CLOSE``."""
        return self._faulted

    async def wait_terminated(self) -> None:
        """Suspend until the session reaches ``TERMINATED`` (by any path)."""
        await self._terminated.wait()

    async def wait_playback_complete(self) -> None:
        """Suspend until the current response has left ``SPEAKING``.

        Resolves on natural completion (every queued frame was sent), barge-in, close
        or fault. Returns immediately when no response is playing. Callers that must
        know what the user actually heard wait here rather than trusting that TTS
        finished *generating*.
        """
        await self._playback_done.wait()

    @property
    def active_tasks(self) -> int:
        """Supervised tasks that have not finished."""
        return len(self._tasks)

    @override
    def __repr__(self) -> str:
        return f"DuplexSession(state={self.state.value}, tasks={len(self._tasks)})"

    # ------------------------------------------------------------------ signalling

    def connect(self, *, observed: int | None = None) -> Transition:
        """Transport negotiation started (``IDLE -> CONNECTING``)."""
        return self._machine.fire(SessionTrigger.CONNECT, observed=observed)

    def connected(self, *, observed: int | None = None) -> Transition:
        """Media path is up (``CONNECTING -> LISTENING``)."""
        return self._machine.fire(SessionTrigger.CONNECTED, observed=observed)

    # ------------------------------------------------------------------ VAD input

    async def on_user_speech_started(self, *, at_ns: int | None = None) -> BargeInReport | None:
        """VAD onset. Triggers a barge-in while the agent is thinking or speaking.

        Args:
            at_ns: monotonic detection timestamp from the VAD (defaults to now); the
                barge-in latency budget is measured from it.

        Returns:
            The barge-in report, or ``None`` if no barge-in was needed.

        If the caller is cancelled while the drain is being reaped, the cancellation
        propagates; the session stays ``DRAINING`` and :meth:`aclose` completes it.
        """
        now = self._clock() if at_ns is None else at_ns
        self._speech_started_at = now
        self._pending_turn_end = False
        self._sink.emit(VadSpeechStarted(trace=self._machine.trace, at_ns=now))
        if self._machine.state in ACTIVE_OUTPUT_STATES:
            return await self._barge_in(now)
        return None

    async def interrupt(self, *, at_ns: int | None = None) -> BargeInReport | None:
        """Cancel agent output on an explicit request (e.g. a client cancel signal).

        Takes exactly the barge-in path, without touching VAD turn bookkeeping.

        Returns:
            The barge-in report, or ``None`` if the agent was neither thinking nor
            speaking.
        """
        if self._machine.state in ACTIVE_OUTPUT_STATES:
            return await self._barge_in(self._clock() if at_ns is None else at_ns)
        return None

    def on_user_speech_stopped(self, *, at_ns: int | None = None) -> Transition | None:
        """VAD offset. Ends the user turn when ``LISTENING``.

        While a barge-in drain is in flight the turn end is deferred and replayed as soon
        as the session is back in ``LISTENING`` -- the user's utterance is never lost.

        Returns:
            The ``USER_TURN_ENDED`` transition if it was committed now, else ``None``.
        """
        now = self._clock() if at_ns is None else at_ns
        started, self._speech_started_at = self._speech_started_at, None
        self._sink.emit(
            VadSpeechStopped(
                trace=self._machine.trace,
                at_ns=now,
                speech_duration_ns=None if started is None else now - started,
            )
        )
        state = self._machine.state
        if state is SessionState.LISTENING:
            return self._end_user_turn(now)
        if state in {SessionState.BARGED_IN, SessionState.DRAINING}:
            self._pending_turn_end = True
        return None

    # ------------------------------------------------------------------ agent output

    def spawn(self, coro: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
        """Run a turn-scoped task (e.g. LLM generation); cancelled on barge-in.

        Raises:
            SessionStateError: unless ``PROCESSING_THOUGHT`` or ``SPEAKING``; ``coro`` is
                closed.
        """
        if self._machine.state not in ACTIVE_OUTPUT_STATES:
            coro.close()
            msg = f"cannot spawn {name!r} in state {self._machine.state.value!r}"
            raise SessionStateError(msg)
        return self._tasks.spawn(coro, name=name)

    def mark_first_token(self, *, at_ns: int | None = None) -> FirstTokenLatency | None:
        """Report the first generated token of the current response (once per turn)."""
        now = self._clock() if at_ns is None else at_ns
        return self._first_token.first_token(trace=self._machine.trace, at_ns=now)

    def begin_response(self, *, observed: int | None = None) -> TraceContext:
        """First response audio is ready (``PROCESSING_THOUGHT -> SPEAKING``).

        Re-arms the egress ring (fencing any stale producer) and starts the pump.

        Returns:
            The trace context every frame of this response must carry.
        """
        transition = self._machine.fire(SessionTrigger.RESPONSE_AUDIO_STARTED, observed=observed)
        self._playback_done.clear()
        self._ring.reset()
        _ = self._tasks.spawn(self._pump(), name=f"{_PUMP_TASK_PREFIX}:{transition.trace.span_id}")
        return transition.trace

    def skip_response(self, *, observed: int | None = None) -> Transition:
        """The agent decided not to speak (``PROCESSING_THOUGHT -> LISTENING``)."""
        transition = self._machine.fire(SessionTrigger.RESPONSE_SKIPPED, observed=observed)
        self._first_token.disarm()
        return transition

    async def push_audio(self, frame: AudioFrame) -> int:
        """Queue one response frame for playback, honouring ring backpressure.

        Returns:
            Frames evicted to admit it (non-zero only for a DROP_OLDEST ring).

        Raises:
            SessionStateError: unless ``SPEAKING``.
            StaleSpanError: if ``frame`` belongs to another span.
            RingClosedError: if a barge-in aborted the response while waiting.
        """
        if self._machine.state is not SessionState.SPEAKING:
            msg = f"cannot push audio in state {self._machine.state.value!r}"
            raise SessionStateError(msg)
        if frame.trace != self._machine.trace:
            msg = f"frame span {frame.trace.span_id} is not the current response span"
            raise StaleSpanError(msg)
        return await self._ring.put(frame)

    def end_response(self) -> None:
        """No more frames for this response; playback drains, then ``-> LISTENING``.

        Raises:
            SessionStateError: unless ``SPEAKING``.
        """
        if self._machine.state is not SessionState.SPEAKING:
            msg = f"cannot end a response in state {self._machine.state.value!r}"
            raise SessionStateError(msg)
        self._ring.close()

    # ------------------------------------------------------------------ shutdown

    async def aclose(self) -> JoinReport:
        """Gracefully terminate: ``CLOSE -> DRAINING -> TERMINATED``. Idempotent.

        Cancels every turn task, drops queued audio and reaps tasks within
        ``drain_timeout_s``; stragglers fault the session.
        """
        machine = self._machine
        self._tasks.close()
        if not machine.terminated:
            machine.fire(SessionTrigger.CLOSE)
        self._tasks.cancel()
        self._ring.abort()
        self._first_token.disarm()
        report = await self._tasks.join(self._drain_timeout_s)
        if report.stragglers:
            self._fault()
        if machine.state is SessionState.DRAINING:
            machine.fire(SessionTrigger.DRAIN_COMPLETE)
        return report

    async def abort(self) -> JoinReport:
        """Fail the session immediately (``FAULT -> TERMINATED``). Idempotent.

        Used by supervisors when something outside the session failed. Turn tasks are
        cancelled, queued audio is dropped and tasks are reaped within
        ``drain_timeout_s``.
        """
        self._tasks.close()
        self._fault()
        return await self._tasks.join(self._drain_timeout_s)

    # ------------------------------------------------------------------ internals

    def _observe(self, transition: Transition) -> None:
        if transition.source is SessionState.SPEAKING:
            self._playback_done.set()
        if transition.target is SessionState.TERMINATED:
            self._faulted = transition.trigger is SessionTrigger.FAULT
            self._terminated.set()

    def _end_user_turn(self, now: int) -> Transition:
        self._machine.open_span()
        transition = self._machine.fire(SessionTrigger.USER_TURN_ENDED)
        self._first_token.user_turn_ended(now)
        return transition

    async def _barge_in(self, detected_at: int) -> BargeInReport:
        machine = self._machine
        # ---- synchronous critical section: no await until DRAINING ----
        machine.fire(SessionTrigger.BARGE_IN)
        cancelled = self._tasks.cancel()
        dropped = self._ring.abort()
        self._first_token.disarm()
        machine.fire(SessionTrigger.DRAIN)
        latency = self._clock() - detected_at
        # ---- reap (bounded) ----
        joined = await self._tasks.join(self._drain_timeout_s)
        if joined.stragglers:
            self._fault()
        resumed = False
        if machine.state is SessionState.DRAINING:
            resumed = machine.fire(SessionTrigger.DRAIN_COMPLETE).target is SessionState.LISTENING
        if resumed and self._pending_turn_end:
            self._pending_turn_end = False
            self._end_user_turn(self._clock())
        self._sink.emit(
            BargeInHandled(
                trace=machine.trace,
                at_ns=self._clock(),
                detect_to_draining_ns=latency,
                budget_ns=self._barge_in_budget_ns,
                within_budget=latency < self._barge_in_budget_ns,
                cancelled_tasks=cancelled,
                dropped_frames=dropped,
                stragglers=joined.stragglers,
            )
        )
        return BargeInReport(
            detect_to_draining_ns=latency,
            within_budget=latency < self._barge_in_budget_ns,
            cancelled_tasks=cancelled,
            dropped_frames=dropped,
            stragglers=joined.stragglers,
            resumed_listening=resumed,
        )

    async def _pump(self) -> None:
        ring, egress, g2g, clock = self._ring, self._egress, self._g2g, self._clock
        while (frame := await ring.get()) is not None:
            await egress.send(frame)
            g2g.record(
                trace=frame.trace,
                sequence=frame.sequence,
                origin_ns=frame.origin_ns,
                egress_ns=clock(),
            )
        self._machine.fire(SessionTrigger.RESPONSE_AUDIO_FINISHED)

    def _on_task_failure(self, task: asyncio.Task[None], error: BaseException) -> None:
        self._sink.emit(
            TaskFailed(
                trace=self._machine.trace,
                at_ns=self._clock(),
                task_name=task.get_name(),
                error_type=type(error).__qualname__,
            )
        )
        self._fault()

    def _fault(self) -> None:
        if not self._machine.terminated:
            self._machine.fire(SessionTrigger.FAULT)
        self._tasks.cancel()
        self._ring.abort()
        self._first_token.disarm()

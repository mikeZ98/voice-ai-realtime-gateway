"""End-to-end ingress-to-egress session orchestration.

``GatewaySessionRunner`` binds one :class:`~..transport.base.VoiceTransport`, one
:class:`~..vad.base.StreamingVAD` and one :class:`~..domain.duplex.DuplexSession`. The
session uses the transport as its audio egress. Four tasks run in one
:class:`asyncio.TaskGroup`::

    transport  -- reads the socket, demultiplexes audio / control, heartbeat
    ingress    -- ingress frames -> VAD bridge -> session (barge-in, turn end)
    control    -- vad.config / response.cancel / session.close -> actions
    watcher    -- a session that terminates on its own closes the socket

Lifecycle guarantees:

* Transport closed (by the peer, network loss, heartbeat or a protocol violation):
  ingress ends, the session closes gracefully (``CLOSE -> DRAINING -> TERMINATED``),
  and :meth:`run` returns a :class:`RunOutcome`.
* Session faulted (a turn task or the egress pump failed): the socket closes with
  ``1011``.
* A runner task failed: the failure is reported (``TaskFailed``), the session is
  aborted (``FAULT``), the socket closes with ``1011``, and the exception group
  propagates. Bugs are never swallowed.
* The runner is cancelled (server shutdown): the socket closes with ``1001``, the session
  closes, and the cancellation propagates.
* After every barge-in (VAD onset or ``response.cancel``) the client receives
  ``playback.flush`` so it drops audio already in its jitter buffer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, final, override

from voice_ai_realtime_gateway.domain.duplex import DuplexSession
from voice_ai_realtime_gateway.domain.states import SessionState
from voice_ai_realtime_gateway.telemetry.events import TaskFailed
from voice_ai_realtime_gateway.transport.control import (
    CancelResponse,
    PlaybackFlush,
    VadConfigUpdate,
)
from voice_ai_realtime_gateway.transport.errors import CloseCode
from voice_ai_realtime_gateway.vad.bridge import VadBridge

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.telemetry.ids import AudioSpanId
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import CloseInitiator, VoiceTransport
    from voice_ai_realtime_gateway.vad.base import StreamingVAD

__all__ = ["GatewaySessionRunner", "RunOutcome", "TurnHandler"]

type TurnHandler = Callable[[DuplexSession], None]
"""Called once per user turn, in ``PROCESSING_THOUGHT``; typically spawns generation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOutcome:
    """How a session run ended."""

    close_code: int | None
    close_reason: str
    closed_by: CloseInitiator | None
    final_state: SessionState
    faulted: bool
    turns: int
    barge_ins: int


@final
class GatewaySessionRunner:
    """Supervises one client session from first packet to socket close."""

    __slots__ = (
        "_barge_ins",
        "_bridge",
        "_clock",
        "_dispatched_span",
        "_on_user_turn",
        "_session",
        "_sink",
        "_started",
        "_transport",
        "_turns",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        *,
        transport: VoiceTransport,
        vad: StreamingVAD,
        sink: TelemetrySink,
        clock: MonotonicClock = time.monotonic_ns,
        on_user_turn: TurnHandler | None = None,
        drain_timeout_s: float = 0.1,
    ) -> None:
        """Create the session for ``transport``'s correlation context."""
        self._transport = transport
        self._sink = sink
        self._clock = clock
        self._on_user_turn = on_user_turn
        self._session = DuplexSession(
            egress=transport,
            sink=sink,
            trace=transport.trace,
            clock=clock,
            drain_timeout_s=drain_timeout_s,
        )
        self._bridge = VadBridge(vad, self._session)
        self._dispatched_span: AudioSpanId | None = None
        self._turns = 0
        self._barge_ins = 0
        self._started = False

    @property
    def session(self) -> DuplexSession:
        """The supervised session (for generation and TTS code)."""
        return self._session

    @property
    def transport(self) -> VoiceTransport:
        """The client transport."""
        return self._transport

    @property
    def vad(self) -> StreamingVAD:
        """The voice-activity detector."""
        return self._bridge.vad

    @override
    def __repr__(self) -> str:
        return (
            f"GatewaySessionRunner(state={self._session.state.value}, turns={self._turns}, "
            f"barge_ins={self._barge_ins})"
        )

    async def run(self) -> RunOutcome:
        """Run the session until the transport closes.

        Raises:
            RuntimeError: if called more than once.
            ExceptionGroup: if a runner task failed (after cleanup).
            asyncio.CancelledError: if the runner was cancelled (after cleanup).
        """
        if self._started:
            msg = "GatewaySessionRunner.run() may only be called once"
            raise RuntimeError(msg)
        self._started = True
        session, transport = self._session, self._transport
        session.connect()
        session.connected()
        try:
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(transport.run(), name="transport")
                    group.create_task(self._ingress_loop(), name="ingress")
                    group.create_task(self._control_loop(), name="control")
                    group.create_task(self._watch_session(), name="session-watch")
            except* Exception as failures:
                now = self._clock()
                for error in failures.exceptions:
                    self._sink.emit(
                        TaskFailed(
                            trace=session.trace,
                            at_ns=now,
                            task_name="gateway-runner",
                            error_type=type(error).__qualname__,
                        )
                    )
                await session.abort()
                await transport.close(CloseCode.INTERNAL_ERROR, "internal error")
                raise
        except asyncio.CancelledError:
            await transport.close(CloseCode.GOING_AWAY, "server shutting down")
            await session.aclose()
            raise
        return RunOutcome(
            close_code=transport.close_code,
            close_reason=transport.close_reason,
            closed_by=transport.closed_by,
            final_state=session.state,
            faulted=session.faulted,
            turns=self._turns,
            barge_ins=self._barge_ins,
        )

    async def _ingress_loop(self) -> None:
        async for frame in self._transport.ingress:
            outcome = await self._bridge.feed(frame)
            if outcome.barge_in is not None:
                await self._flush_client()
            self._dispatch_turn()
        await self._session.aclose()

    async def _control_loop(self) -> None:
        async for message in self._transport.control:
            if isinstance(message, VadConfigUpdate):
                self._bridge.vad.reconfigure(message.config)
            elif isinstance(message, CancelResponse):
                if await self._session.interrupt() is not None:
                    await self._flush_client()
            else:
                await self._transport.close(CloseCode.NORMAL, message.reason or "client hangup")

    async def _watch_session(self) -> None:
        session = self._session
        await session.wait_terminated()
        if session.faulted:
            await self._transport.close(CloseCode.INTERNAL_ERROR, "session fault")
        else:
            await self._transport.close(CloseCode.NORMAL, "session ended")

    async def _flush_client(self) -> None:
        self._barge_ins += 1
        await self._transport.control.send(PlaybackFlush())

    def _dispatch_turn(self) -> None:
        session = self._session
        if session.state is not SessionState.PROCESSING_THOUGHT:
            return
        span = session.trace.span_id
        if span == self._dispatched_span:
            return
        self._dispatched_span = span
        self._turns += 1
        if self._on_user_turn is not None:
            self._on_user_turn(session)

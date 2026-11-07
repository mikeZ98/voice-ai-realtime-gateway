"""Bridge from streaming VAD decisions to the duplex session.

``ONSET`` maps to :meth:`DuplexSession.on_user_speech_started`, which performs a
barge-in when the agent is thinking or speaking. ``OFFSET`` maps to
:meth:`DuplexSession.on_user_speech_stopped`, which ends the user turn. Both receive the
decision's ``at_ns``, so barge-in latency is measured from the moment the gateway
received the triggering audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from voice_ai_realtime_gateway.vad.base import VadEvent

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.duplex import BargeInReport, DuplexSession
    from voice_ai_realtime_gateway.domain.machine import Transition
    from voice_ai_realtime_gateway.vad.base import StreamingVAD, VadDecision

__all__ = ["BridgeOutcome", "VadBridge"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeOutcome:
    """What one ingress frame caused.

    Attributes:
        decision: the VAD's classification of the frame.
        barge_in: the barge-in report if the onset interrupted agent output.
        turn: the ``USER_TURN_ENDED`` transition if the offset ended the user turn.
    """

    decision: VadDecision
    barge_in: BargeInReport | None
    turn: Transition | None


@final
class VadBridge:
    """Feeds ingress frames through a VAD and applies edges to a session."""

    __slots__ = ("_session", "_vad")

    def __init__(self, vad: StreamingVAD, session: DuplexSession) -> None:
        """Bind ``vad`` decisions to ``session``."""
        self._vad = vad
        self._session = session

    @property
    def vad(self) -> StreamingVAD:
        """The detector."""
        return self._vad

    async def feed(self, frame: AudioFrame) -> BridgeOutcome:
        """Classify ``frame`` and apply any speech edge to the session."""
        decision = self._vad.process_chunk(frame)
        barge_in: BargeInReport | None = None
        turn: Transition | None = None
        if decision.event is VadEvent.ONSET:
            barge_in = await self._session.on_user_speech_started(at_ns=decision.at_ns)
        elif decision.event is VadEvent.OFFSET:
            turn = self._session.on_user_speech_stopped(at_ns=decision.at_ns)
        return BridgeOutcome(decision=decision, barge_in=barge_in, turn=turn)

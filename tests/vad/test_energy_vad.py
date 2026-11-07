"""Energy VAD: level estimation, hysteresis contracts, determinism and per-frame cost."""

from __future__ import annotations

import math
import statistics
import struct
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from audio_support import LOUD, pcm_frame, square_wave
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.vad.base import SILENCE_FLOOR_DBFS, VadConfig, VadEvent
from voice_ai_realtime_gateway.vad.energy import EnergyVAD, pcm16_level_dbfs

TRACE = TraceContext.new()


def level_of(amplitude: int) -> float:
    return 20.0 * math.log10(amplitude / 32768.0)


# ------------------------------------------------------------------------ level estimation


def test_digital_silence_reads_the_floor() -> None:
    assert pcm16_level_dbfs(memoryview(bytes(640))) == SILENCE_FLOOR_DBFS


@pytest.mark.parametrize("amplitude", [1, 100, 3_277, 16_384, 32_767])
def test_square_wave_level_is_exact(amplitude: int) -> None:
    level = pcm16_level_dbfs(memoryview(square_wave(amplitude, 320)))
    assert level == pytest.approx(level_of(amplitude), abs=1e-9)


def test_samples_are_decoded_little_endian_on_every_host() -> None:
    little_endian_256 = memoryview(b"\x00\x01" * 4)
    assert pcm16_level_dbfs(little_endian_256) == pytest.approx(level_of(256), abs=1e-9)


def test_sub_floor_energy_is_clamped() -> None:
    samples = [0] * 1919 + [1]
    payload = memoryview(struct.pack("<1920h", *samples))
    assert pcm16_level_dbfs(payload) == SILENCE_FLOOR_DBFS


@pytest.mark.parametrize("nbytes", [0, 3])
def test_partial_samples_are_rejected(nbytes: int) -> None:
    with pytest.raises(ValueError, match="whole samples"):
        pcm16_level_dbfs(memoryview(bytes(nbytes)))


# ------------------------------------------------------------------------ configuration


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start_threshold_dbfs": math.nan}, "finite"),
        ({"stop_threshold_dbfs": -math.inf}, "finite"),
        ({"start_threshold_dbfs": 1.0}, "stop <= start <= 0"),
        ({"start_threshold_dbfs": -50.0, "stop_threshold_dbfs": -40.0}, "stop <= start"),
        ({"stop_threshold_dbfs": -121.0}, "stop <= start"),
        ({"min_speech_ms": 0}, "min_speech_ms"),
        ({"min_speech_ms": 2_001}, "min_speech_ms"),
        ({"hangover_ms": 0}, "hangover_ms"),
        ({"hangover_ms": 10_001}, "hangover_ms"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        VadConfig(**kwargs)  # type: ignore[arg-type]


def test_equal_thresholds_are_allowed() -> None:
    assert VadConfig(start_threshold_dbfs=-40.0, stop_threshold_dbfs=-40.0).hangover_ms == 300


# ------------------------------------------------------------------------ hysteresis


def run(vad: EnergyVAD, amplitudes: list[int], ms: int = 20) -> list[VadEvent | None]:
    return [vad.process_chunk(pcm_frame(TRACE, amplitude=a, ms=ms)).event for a in amplitudes]


def test_onset_requires_min_speech_duration() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=60))
    assert run(vad, [LOUD, LOUD, LOUD]) == [None, None, VadEvent.ONSET]
    assert vad.speaking


def test_a_transient_shorter_than_min_speech_is_rejected() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=60))
    assert run(vad, [LOUD, LOUD, 0, LOUD, LOUD, 0]) == [None] * 6
    assert not vad.speaking


def test_offset_requires_trailing_silence_hangover() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=20, hangover_ms=100))
    events = run(vad, [LOUD] + [0] * 5)
    assert events == [VadEvent.ONSET, None, None, None, None, VadEvent.OFFSET]
    assert not vad.speaking


def test_a_pause_shorter_than_hangover_keeps_the_segment_open() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=20, hangover_ms=100))
    events = run(vad, [LOUD, 0, 0, 0, LOUD, 0, 0, 0, 0])
    assert events.count(VadEvent.ONSET) == 1
    assert VadEvent.OFFSET not in events


def test_hysteresis_band_holds_the_current_state() -> None:
    config = VadConfig(
        start_threshold_dbfs=-30.0, stop_threshold_dbfs=-50.0, min_speech_ms=20, hangover_ms=40
    )
    in_band = 328  # ~ -40 dBFS: below start, above stop
    assert -50.0 < level_of(in_band) < -30.0
    silent = EnergyVAD(config)
    assert run(silent, [in_band] * 10) == [None] * 10, "band audio cannot start speech"
    speaking = EnergyVAD(config)
    events = run(speaking, [LOUD] + [in_band] * 10)
    assert events == [VadEvent.ONSET] + [None] * 10, "band audio cannot end speech"


def test_decisions_are_duration_based_not_frame_count_based() -> None:
    config = VadConfig(min_speech_ms=60, hangover_ms=60)
    ten_ms = run(EnergyVAD(config), [LOUD] * 6, ms=10)
    assert ten_ms.index(VadEvent.ONSET) == 5
    twenty_ms = run(EnergyVAD(config), [LOUD] * 3, ms=20)
    assert twenty_ms.index(VadEvent.ONSET) == 2


def test_decision_carries_level_state_and_ingress_timestamp() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=20))
    decision = vad.process_chunk(pcm_frame(TRACE, amplitude=LOUD, origin_ns=123))
    assert decision.event is VadEvent.ONSET
    assert decision.is_speech
    assert decision.at_ns == 123
    assert decision.level_dbfs == pytest.approx(level_of(LOUD))


def test_reconfigure_preserves_state_and_reset_clears_it() -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=20, hangover_ms=1_000))
    run(vad, [LOUD])
    vad.reconfigure(VadConfig(min_speech_ms=20, hangover_ms=40))
    states = [vad.speaking]  # reconfiguring must not drop an active segment
    assert vad.config.hangover_ms == 40
    assert run(vad, [0, 0]) == [None, VadEvent.OFFSET], "new hangover applies immediately"
    run(vad, [LOUD])
    states.append(vad.speaking)
    vad.reset()
    states.append(vad.speaking)
    assert states == [True, True, False]
    assert repr(vad).startswith("EnergyVAD(speaking=False")


def test_encoded_audio_is_rejected() -> None:
    opus = AudioFrame.wrap(
        bytes(40),
        codec=AudioCodec.OPUS,
        sample_rate_hz=48_000,
        channels=1,
        duration_ns=20_000_000,
        sequence=0,
        origin_ns=0,
        trace=TRACE,
    )
    with pytest.raises(ValueError, match="PCM16"):
        EnergyVAD().process_chunk(opus)


@settings(max_examples=200)
@given(
    amplitudes=st.lists(st.sampled_from([0, 50, 328, 1_000, LOUD, 32_767]), max_size=120),
    min_speech_ms=st.integers(min_value=1, max_value=200),
    hangover_ms=st.integers(min_value=1, max_value=400),
)
def test_edges_alternate_and_agree_with_state(
    amplitudes: list[int], min_speech_ms: int, hangover_ms: int
) -> None:
    vad = EnergyVAD(VadConfig(min_speech_ms=min_speech_ms, hangover_ms=hangover_ms))
    replay = EnergyVAD(VadConfig(min_speech_ms=min_speech_ms, hangover_ms=hangover_ms))
    previous_speech = False
    edges: list[VadEvent] = []
    for amplitude in amplitudes:
        frame = pcm_frame(TRACE, amplitude=amplitude)
        decision = vad.process_chunk(frame)
        assert replay.process_chunk(frame) == decision, "deterministic"
        if decision.event is VadEvent.ONSET:
            assert not previous_speech
            assert decision.is_speech
        elif decision.event is VadEvent.OFFSET:
            assert previous_speech
            assert not decision.is_speech
        else:
            assert decision.is_speech is previous_speech
        if decision.event is not None:
            edges.append(decision.event)
        previous_speech = decision.is_speech
    assert edges == [VadEvent.ONSET, VadEvent.OFFSET] * (len(edges) // 2) + (
        [VadEvent.ONSET] if len(edges) % 2 else []
    )


# ------------------------------------------------------------------------ cost


def test_per_frame_cost_is_far_below_the_frame_period() -> None:
    vad = EnergyVAD()
    frames = [
        pcm_frame(TRACE, amplitude=LOUD if i % 7 else 0, rate=48_000, sequence=i)
        for i in range(400)
    ]
    costs = []
    for frame in frames:
        start = time.perf_counter_ns()
        vad.process_chunk(frame)
        costs.append(time.perf_counter_ns() - start)
    assert statistics.median(costs) < 200_000, "median VAD cost must stay under 0.2 ms"
    assert max(costs) < 20_000_000, "no frame may cost a full 20 ms frame period"

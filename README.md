# voice-ai-realtime-gateway

Ultra-low-latency, full-duplex voice agent gateway core for Python 3.12+ and asyncio,
with zero runtime dependencies.

Phase 1 provides:

- **Session state machine.** Formal 8-state lifecycle (`IDLE → CONNECTING → LISTENING →
  PROCESSING_THOUGHT → SPEAKING → BARGED_IN → DRAINING → TERMINATED`) with strict
  rejection of invalid transitions and strictly increasing Lamport timestamps.
- **Barge-in.** User speech cancels agent output and reaches `DRAINING` with no `await`
  on the critical path. Measured at 22 µs p50, against a 15 ms budget. No task leaks,
  no lost cancellations, no stale audio.
- **Bounded zero-copy audio ring.** Frame- and byte-bounded circular buffer of
  read-only `memoryview` frames (PCM16 / Opus). It offers `BACKPRESSURE` or
  `DROP_OLDEST` overflow policies plus epoch fencing.
- **Typed telemetry.** VAD, first-token, glass-to-glass packet latency, transitions and
  barge-in events, all correlated by `X-Session-ID` / `X-Audio-Span-ID`.

Design, the full transition matrix and the memory budget model are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Usage sketch

```python
from voice_ai_realtime_gateway.domain import DuplexSession
from voice_ai_realtime_gateway.telemetry import FlightRecorder

session = DuplexSession(egress=webrtc_track, sink=FlightRecorder())
session.connect()
session.connected()  # LISTENING

await session.on_user_speech_started()  # VAD onset
session.on_user_speech_stopped()  # VAD offset → PROCESSING_THOUGHT
session.spawn(generate_reply(session), name="llm")

# inside generate_reply:
session.mark_first_token()
span = session.begin_response()  # SPEAKING; frames must carry `span`
await session.push_audio(frame)  # backpressured
session.end_response()  # drains → LISTENING

# user talks over the agent:
report = await session.on_user_speech_started()  # BargeInReport, < 15 ms to DRAINING
await session.aclose()  # DRAINING → TERMINATED
```

## Development

```bash
uv sync                                    # Python 3.12, dev toolchain
uv run ruff check . && uv run ruff format --check .
uv run mypy                                # --strict over src/ and tests/
uv run bandit -c pyproject.toml -r src
uv run pytest                              # network-free, warnings are errors
./scripts/coverage-gate.sh                 # 100.00 % line + branch, per file
./scripts/audit-attribution.sh             # sole-committer invariant
./scripts/audit-hygiene.sh                 # no absolute paths, artifacts or cache leaks
```

Every tool cache (pytest, ruff, mypy, coverage, Hypothesis, uv) is redirected under
`./.cache/`. CPython bytecode can only be redirected through the environment, so export
`PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"` in your shell. The gate scripts default to
that location themselves.

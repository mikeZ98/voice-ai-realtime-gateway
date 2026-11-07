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

Phase 2 adds:

- **WebSocket voice transport.** Binary audio packets (54-byte header carrying
  `X-Session-ID` / `X-Audio-Span-ID`) and typed JSON control share one connection.
  Ingress is bounded (`DROP_OLDEST` or real TCP backpressure), protocol violations close
  with their RFC 6455 code, and a heartbeat detects dead peers.
- **Streaming VAD.** A deterministic energy VAD with two-threshold hysteresis
  (speech-start confidence, trailing-silence hangover) drives barge-in and turn-taking.
- **`GatewaySessionRunner`.** Transport, VAD and session run under one supervisor:
  disconnects end calls gracefully, faults close with `1011`, and every barge-in tells
  the client to flush its buffered audio.

Design, the full transition matrix and the memory budget model are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The transport, data flow, wire protocol
and overhead model are in [`docs/TRANSPORT.md`](docs/TRANSPORT.md).

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

## Serving a call

```python
from voice_ai_realtime_gateway.pipeline import GatewaySessionRunner
from voice_ai_realtime_gateway.telemetry import FlightRecorder, TraceContext
from voice_ai_realtime_gateway.transport import WebSocketVoiceTransport
from voice_ai_realtime_gateway.vad import EnergyVAD


async def handle(connection, headers):  # connection: any WebSocketConnection
    sink = FlightRecorder()
    transport = WebSocketVoiceTransport(
        connection, trace=TraceContext.from_headers(headers), sink=sink
    )
    runner = GatewaySessionRunner(
        transport=transport,
        vad=EnergyVAD(),
        sink=sink,
        on_user_turn=lambda session: session.spawn(reply(session), name="llm"),
    )
    outcome = await runner.run()  # returns when the socket closes
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

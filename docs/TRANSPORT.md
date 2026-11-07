# Phase 2: real-time transport and ingress pipeline

Phase 2 connects the Phase 1 core to clients. It adds a WebSocket voice transport, a
binary audio framing codec, a typed control protocol, streaming VAD, and the
orchestrator that supervises a whole call.

## 1. Layering

```
pipeline/   GatewaySessionRunner                  → domain, vad, transport
transport/  WebSocketVoiceTransport, framing,     → core, telemetry, vad.base
            control, MemoryWebSocket, ports
vad/        StreamingVAD, EnergyVAD, VadBridge    → core, domain (bridge only)
domain/ · core/ · telemetry/                      (Phase 1)
```

The gateway still has **zero runtime dependencies**. The transport speaks to a
`WebSocketConnection` port (`recv` / `send` / `close`), which has the shape of the
`websockets` asyncio API. Adapting a real server takes a few lines: translate the
library's `ConnectionClosed` into `TransportClosedError`. Configure the library's own
limits to match (`max_size = HEADER_SIZE + max_payload_bytes`, a small `max_queue`).
Everything above the port is tested over `MemoryWebSocket`, an in-memory pair with
bounded buffers, WebSocket close semantics and simulated network loss. The suite stays
100 % network-free under `pytest-socket`.

## 2. Data flow

```
 client                                gateway (one task group per call)
 ──────                                ─────────────────────────────────────────────────
  mic ─► [packet] ─binary─►  transport reader ─► decode (zero-copy) ─► checks:
                                                   size · session id · format · sequence
                                                        │
                                                        ▼
                                          ingress AudioRingBuffer (bounded, DROP_OLDEST)
                                                        │
                                       ingress loop ◄───┘
                                         EnergyVAD.process_chunk ─► VadBridge
                                           ONSET  ─► session.on_user_speech_started
                                                      (barge-in if thinking/speaking)
                                           OFFSET ─► session.on_user_speech_stopped
                                                      ─► on_user_turn(session) ─► LLM/TTS
  ctrl ─► {json} ─text──►  transport reader ─► ping→pong · session.config (format)
                                             └► control queue (bounded) ─► control loop
                                                  vad.config      ─► vad.reconfigure
                                                  response.cancel ─► session.interrupt
                                                  session.close   ─► transport.close(1000)
 speaker ◄─ [packet] ◄─binary─ transport.send ◄─ egress pump ◄─ egress ring ◄─ TTS
 jitter  ◄─ playback.flush ◄─text─ after every barge-in
 buffer ◄─ ping ◄──────────────── heartbeat (any inbound message = liveness)
                                    watcher: session TERMINATED ─► transport.close
```

## 3. Wire protocol

**Binary (audio).** A fixed 54-byte big-endian header, then the payload. See
`transport/framing.py` for the byte-level table:

```
magic "VG" · version 1 · codec · channels · flags=0 · sample_rate u32 · duration_ns u32
· sequence u32 · sender timestamp_ns u64 · X-Session-ID (16 B) · X-Audio-Span-ID (8 B)
· payload_length u32 · payload
```

The receiver rebases `origin_ns` onto **its own** monotonic clock and returns the sender
timestamp separately, because clocks of different hosts are not comparable. Decoding is
zero-copy: the frame's payload is a read-only view into the received message.
`PacketStreamDecoder` provides the same framing for byte streams, validating the length
before buffering any body.

**Text (control).** Compact JSON with exact schemas: `ping`, `pong`, `session.config`,
`vad.config`, `response.cancel`, `session.close` from the client, and `ping`, `pong`,
`playback.flush` from the server. Strict integers (`true` is not an int), finite numbers
only (`NaN`, `Infinity` and `1e999` are rejected), no unknown or extra fields, and
direction-checked types.

**Close codes (RFC 6455).**

| Code | Cause |
|---|---|
| 1000 | client `session.close`, or the session ended normally |
| 1001 | heartbeat timeout, or server shutdown |
| 1003 | unsupported packet version, codec or format; audio that differs from `session.config` |
| 1006 | network loss (observed, never sent) |
| 1007 | malformed packet or control message, invalid values |
| 1008 | foreign `X-Session-ID` (audio injection), control backlog flood |
| 1009 | audio or control message above the configured limit |
| 1011 | session fault, or a runner task failure |

## 4. Backpressure and bounded ingress

* **`DROP_OLDEST`** (default): the reader never stalls. When downstream (VAD, ASR) is
  slow, the oldest queued audio is evicted (`BufferOverflow` telemetry), so queueing
  delay stays at or below 64 frames (1.28 s at 20 ms).
* **`BACKPRESSURE`**: the reader stops reading the socket when the ring is full. The
  library's receive buffer fills, the TCP window closes and the *client* is throttled.
  No audio is lost. While stalled this way, heartbeat dead-peer detection is suspended:
  the stall is ours, not the peer's.
* **Sequence hygiene** per span: gaps count as loss (`IngressGap`, `lost_packets`); late
  or duplicate packets are dropped (`late_dropped`). Realtime audio is never replayed
  out of order.
* A client cannot drive gateway memory past the ring budget, one in-flight message and
  `max_pending_control` parsed control messages.

## 5. Streaming VAD

`EnergyVAD` measures RMS level in dBFS, decoding PCM16 explicitly as little-endian and
summing squares with `math.sumprod`. Hysteresis has two thresholds and two durations:

* onset after `min_speech_ms` of audio at or above `start_threshold_dbfs` (rejects clicks);
* offset after `hangover_ms` of audio below `stop_threshold_dbfs` (bridges pauses);
* levels between the thresholds hold the current state.

Runs accumulate media *duration*, not frame counts, so 10 ms and 20 ms packetisation
behave identically. Decisions are timestamped with the frame's ingress `origin_ns`, so
barge-in latency includes any ingress queueing.

## 6. Measured costs

CPython 3.12, Apple Silicon, `perf_counter_ns`, 20,000 iterations:

| Operation (20 ms frame) | p50 | p99 |
|---|---|---|
| `encode_packet` 16 kHz PCM16 | 0.46 µs | 0.50 µs |
| `decode_packet` 16 kHz PCM16 | 2.0 µs | 2.6 µs |
| `EnergyVAD.process_chunk` 16 kHz | 4.0 µs | 5.1 µs |
| `EnergyVAD.process_chunk` 48 kHz | 10.6 µs | 12.2 µs |
| `decode_client_message` (`vad.config`) | 2.3 µs | 2.9 µs |
| VAD-triggered barge-in to `DRAINING` | asserted median < 100 µs in `test_vad_bridge.py` |

The whole ingress path (decode, checks, ring, VAD) costs about 10 µs per 20 ms frame,
or 0.05 % of one core per call at 16 kHz.

## 7. Framing and memory overhead model

**Wire overhead per 20 ms packet.** The packet header is 54 B. WebSocket framing adds
2 B (payload ≤ 125 B) or 4 B (≤ 64 KiB) to the base header, plus 4 B of masking on
client-to-server frames.

| Stream | Payload | Wire size | Overhead |
|---|---|---|---|
| PCM16 16 kHz mono (256 kbit/s) | 640 B | 702 B (ingress) | 9.7 % |
| PCM16 48 kHz mono (768 kbit/s) | 1920 B | 1982 B | 3.2 % |
| Opus 64 kbit/s | 160 B | 222 B | 38.8 % |
| Opus 32 kbit/s | 80 B | 142 B | 77.5 % |

For Opus the fixed 62 B per packet (24.8 kbit/s at 50 packets per second) dominates. Most
of it is the per-packet correlation IDs the protocol requires (24 B). Use 40 or 60 ms
Opus packets to amortise it.

**Memory per connection** (defaults, measured with `tracemalloc`):

| Component | Bound |
|---|---|
| Idle transport + session + VAD + runner | ≈ 35 KiB (measured) |
| Ingress frame metadata (frame, view, decoded trace IDs) | ≈ 740 B per queued frame (measured) |
| Ingress payload retained (zero-copy keeps the whole message alive) | payload + 54 B header + 33 B object per frame |
| Ingress ring | 64 frames, ≤ 256 KiB payload |
| Control queue | ≤ 32 parsed messages (< 100 B each) |
| In-flight message | ≤ 54 B + `max_payload_bytes` (32 KiB) |
| Egress (Phase 1 budget) | ≤ 611 KiB worst case |

Full 16 kHz ingress ring: 64 × (727 + 740) B ≈ **92 KiB**. Absolute worst case (max-size
packets filling the byte budget) ≈ 256 KiB + 64 × 830 B ≈ **310 KiB**. With a full egress
ring, a connection is bounded by ≈ **1 MiB**, so 1,000 concurrent calls fit in about
1 GiB with every buffer full. Typical calls use a small fraction of that.

## 8. Verification

| Property | Evidence |
|---|---|
| Lossless framing for every valid frame | Hypothesis round trips over all header fields (`test_framing.py`) |
| Garbage never escapes as anything but `ProtocolError` | arbitrary-byte, single-byte-corruption and arbitrary-JSON fuzzing |
| Stream decoding is chunking-invariant | Hypothesis random chunk boundaries |
| Every violation maps to its close code | parametrised violation tables (`test_control.py`, `test_websocket_transport.py`) |
| Slow consumer: bounded memory, newest audio kept | 200 frames into an 8-frame `DROP_OLDEST` ring |
| Backpressure throttles the client and loses nothing | `BACKPRESSURE` ring with a 2-message socket buffer |
| Heartbeat ping, dead-peer close, liveness, stall suspension | deterministic `ManualSleeper` + fake clock |
| Disconnect, network drop, protocol violation end calls gracefully | end-to-end runner tests (`test_runner.py`) |
| No old-span audio after `playback.flush` | end-to-end barge-in test |
| Session fault → 1011; runner failure → 1011 + propagate; shutdown → 1001 | end-to-end runner tests |
| VAD hysteresis edges alternate and stay deterministic | Hypothesis property over random level sequences |

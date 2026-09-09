# giga-stt

HTTP speech-to-text service based on
[GigaAM-v3-e2e-rnnt](https://huggingface.co/ai-sage/GigaAM-v3) — SOTA ASR models
for Russian with punctuation and text normalization
(upstream: [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM), MIT license).

* Request format: **multipart/form-data** (`POST /v1/transcribe`) or a raw byte
  stream (`POST /v1/transcribe-raw`).
* Audio: any format and sample rate that ffmpeg can decode
  (wav, mp3, ogg/opus, m4a, flac, webm, amr…), mono/stereo.
* Up to 25 seconds — a single model pass; longer audio — automatic segmentation
  into chunks by voice activity and batched inference.
* Optional word timestamps (`word_timestamps=true`).
* GPU acceleration (CUDA) or CPU; interactive Swagger UI docs at `/docs`.

---

## API

### `GET /health`

```json
{
  "status": "ready",
  "version": "1.0.0",
  "model": "v3_e2e_rnnt",
  "device": "cuda",
  "torch": "2.8.0+cu128",
  "vad_backend": "silero",
  "ffmpeg": "/usr/bin/ffmpeg",
  "detail": null
}
```

Statuses: `loading` — the model is being downloaded/loaded; `ready` — you can send audio;
`error` — the model failed to load (the reason is in `detail`).

### `POST /v1/transcribe` (multipart/form-data)

| Field             | Type | Default    | Description                       |
|-------------------|------|------------|-----------------------------------|
| `file`            | file | —          | Audio                             |
| `word_timestamps` | bool | `false`    | Word timestamps                   |
| `with_segments`   | bool | `true`     | Include the segment list          |

### `POST /v1/transcribe-raw` (raw body)

The request body is the raw audio bytes, `Content-Type: audio/wav|audio/mpeg|audio/ogg|…`,
same parameters but in the query string: `?word_timestamps=true&with_segments=true`.

### Response

```json
{
  "text": "Hello! How are you?",
  "duration": 2.73,
  "language": "ru",
  "model": "v3_e2e_rnnt",
  "vad_backend": null,
  "processing_time": 0.61,
  "segments": [
    {"text": "Hello! How are you?", "start": 0.0, "end": 2.73, "words": null}
  ],
  "words": null
}
```

* `segments` — for long audio: segments `[start, end)` with text
  (and words, if timestamps were requested).
* `words` — flat list of words with timestamps (only when `word_timestamps=true`).
* `vad_backend` — which segmentation backend was used
  (set only for audio longer than 25 s).

### Error codes

| Code | Reason                                                       |
|------|--------------------------------------------------------------|
| 400  | Audio could not be decoded                                   |
| 401  | Invalid/missing API key (when `GIGAAM_API_KEY` is set)       |
| 413  | File or duration exceeds the limits                          |
| 422  | No `file` field                                              |
| 500  | Internal error / model failed to load                        |
| 503  | Model is still loading or all workers are busy (see `Retry-After`) |

---

## Examples

### curl

```bash
# short recording (multipart)
curl -s -X POST http://localhost:8000/v1/transcribe \
  -F "file=@voice.wav" | jq -r .text

# with word timestamps
curl -s -X POST http://localhost:8000/v1/transcribe \
  -F "file=@voice.mp3" -F "word_timestamps=true" | jq

# raw stream
curl -s -X POST http://localhost:8000/v1/transcribe-raw \
  -H "Content-Type: audio/ogg" --data-binary @recording.ogg | jq -r .text
```

### Python (requests)

```python
import requests

r = requests.post(
    "http://localhost:8000/v1/transcribe",
    files={"file": open("voice.wav", "rb")},
    data={"word_timestamps": "true"},
    timeout=600,
)
r.raise_for_status()
data = r.json()
print(data["text"])
for seg in data["segments"]:
    print(f"[{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['text']}")
```

### Python (httpx, streaming a long file)

```python
import httpx

with open("call_center.ogg", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/v1/transcribe-raw",
        content=f,
        headers={"Content-Type": "audio/ogg"},
        params={"word_timestamps": "true"},
        timeout=1800,
    )
print(r.json()["text"])
```

---

## Running locally

Requirements: Python ≥ 3.10, NVIDIA GPU with the driver installed (or CPU),
ffmpeg in PATH (on NixOS the service finds ffmpeg in the nix-store on its own;
otherwise set `GIGAAM_FFMPEG_PATH=/path/to/ffmpeg`).

```bash
# virtual environment (torch 2.8.0 wheels come from the pip cache, if present)
python3 -m venv .venv
.venv/bin/pip install -e .

# service (the ~450 MB checkpoint is downloaded to ~/.cache/gigaam on first start)
.venv/bin/giga-stt

# or directly:
#   .venv/bin/uvicorn giga_stt.app:app --host 0.0.0.0 --port 8000

# check it works
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/transcribe -F "file=@voice.wav"

# official GigaAM test sample:
wget -O example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"
```

## Running in Docker (GPU)

Requires the NVIDIA driver and nvidia-container-toolkit to be installed.

```bash
docker compose -f docker/docker-compose.yml up -d --build
# model loading logs
docker compose -f docker/docker-compose.yml logs -f giga-stt
```

The weights are cached in the docker volume `giga-stt-models` (`/models` inside
the container); after the first start the container comes up without downloading.

CPU-only: comment out the `deploy` block in `docker/docker-compose.yml` and set
`GIGAAM_DEVICE=cpu` (inference is noticeably slower).

---

## Configuration (environment variables)

| Variable                | Default           | Description |
|-------------------------|-------------------|-------------|
| `GIGAAM_MODEL`          | `v3_e2e_rnnt`     | Any GigaAM model: `v3_rnnt`, `v3_ctc`, `v3_e2e_ctc`, … |
| `GIGAAM_DEVICE`         | `auto`            | `auto` / `cuda` / `cpu` |
| `GIGAAM_FP16_ENCODER`   | `false`           | fp16 encoder + autocast. ⚠ On GTX 16xx GPUs (TU116/TU117, no tensor cores) cuDNN fp16 convolutions on long inputs (T ≳ 550 frames) produce NaN — do not enable there; on RTX/A100/H100 enabling it speeds up inference |
| `GIGAAM_CACHE_DIR`      | `~/.cache/gigaam` | Weights directory |
| `GIGAAM_MAX_UPLOAD_MB`  | `100`             | Maximum upload size |
| `GIGAAM_MAX_AUDIO_SEC`  | `7200`            | Safety cap on duration |
| `GIGAAM_MAX_CONCURRENCY`| `1`               | Concurrent inferences (the rest wait in the queue) |
| `GIGAAM_QUEUE_TIMEOUT_SEC`| `300`           | Queue wait time before a 503 |
| `GIGAAM_CHUNK_MAX_SEC`  | `22`              | Chunk length for long audio (model limit is 25 s) |
| `GIGAAM_FR_BATCH_SIZE`  | `4`               | Chunk batch size (increase on GPUs with more VRAM) |
| `GIGAAM_VAD_BACKEND`    | `auto`            | `auto` / `pyannote` / `silero` / `energy` |
| `GIGAAM_API_KEY`        | —                 | If set — requests must send `X-API-Key: <key>` or `Bearer <key>` |
| `GIGAAM_FFMPEG_PATH`    | —                 | Path to the ffmpeg binary (if not in PATH) |
| `HF_TOKEN`              | —                 | HF token for the pyannote longform (gated model) |

---

## Long audio (> 25 s): segmentation backends

The GigaAM model works with fragments up to 25 s, so the service cuts long
recordings into chunks and stitches the results together:

1. **pyannote** (the official repository path) — VAD `pyannote/segmentation-3.0`.
   Requires the `pip install -e ".[longform]"` dependencies, an HF token, and
   accepted terms for the gated model. ⚠ The extra upgrades torch to 2.10.x —
   after installing, verify that CUDA is still available
   (`python -c "import torch; print(torch.cuda.is_available())"`).
2. **silero** — lightweight VAD [silero-vad](https://github.com/snakers4/silero-vad)
   (part of the base dependencies); the recommended option without gated access.
3. **energy** — built-in fallback without external models: chunk boundaries are
   placed at local minima of the signal energy.

`auto` picks the first available backend. Cutting quality only affects phrase
boundaries — pyannote/silero keep words intact by cutting at pauses.

## Performance

Measured on a GTX 1660 SUPER 6 GB, `v3_e2e_rnnt`, fp32 encoder without autocast:

| Audio             | Inference time              |
|-------------------|-----------------------------|
| 11.3 s (short)    | ~0.6 s                      |
| 23 s (short)      | ~1.2 s                      |
| 51.7 s (4 chunks) | ~2.2 s                      |
| 1 hour            | ~2.5 min (silero, batch 4)  |

Recommendations: on GPUs with tensor cores (RTX 20xx+, A100) enable
`GIGAAM_FP16_ENCODER=true` and raise `GIGAAM_FR_BATCH_SIZE` to 8–16;
for multiple GPUs run multiple containers (`GIGAAM_DEVICE=cuda:0/1`) behind a load balancer.

## Notes

* The model is tuned for **Russian** speech (the `e2e` variant adds punctuation
  and normalization: numbers, abbreviations, etc.).
* The service does not store audio: the uploaded file lives in a temp file and is
  deleted right after the response.
* The `gigaam` package is pinned to a specific commit of the upstream repository —
  to update it, change the hash in `pyproject.toml` (the service uses the package's
  internal modules: `utils.AudioDataset`, `types`, `vad_utils`).
* For production, put uvicorn behind nginx/traefik (TLS, request body limits)
  and enable `GIGAAM_API_KEY`.

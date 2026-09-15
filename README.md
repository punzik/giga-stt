# giga-stt

HTTP speech-to-text service based on the Russian
[GigaAM-v3-e2e-rnnt](https://huggingface.co/ai-sage/GigaAM-v3) model.

* Request format: **multipart/form-data** (`POST /v1/transcribe`) or a raw byte
  stream (`POST /v1/transcribe-raw`).
* Audio: any format and sample rate that ffmpeg can decode
  (wav, mp3, ogg/opus, m4a, flac, webm, amr…), mono/stereo.
* Up to 25 seconds — a single model pass; longer audio — automatic segmentation
  into chunks by voice activity and batched inference.
* Optional speaker diarization ([pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)):
  `with_speakers=true` adds the dominant speaker to each segment.
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
  "diarization": "ready",
  "ffmpeg": "/usr/bin/ffmpeg",
  "detail": null
}
```

The model loads in the background. Statuses: `loading` — the model is being
downloaded/loaded; `ready` — you can send requests; `error` — the model failed to load
(the reason is in `detail`).

The service recognizes Russian speech.

### `POST /v1/transcribe` (multipart/form-data)

| Field             | Type | Default    | Description                       |
|-------------------|------|------------|-----------------------------------|
| `file`            | file | —          | Audio                             |
| `with_segments`   | bool | `true`     | Include the segment list          |
| `with_speakers`   | bool | `false`    | Add the dominant speaker to segments (requires the diarization backend) |

### `POST /v1/transcribe-raw` (raw body)

The request body is the raw audio bytes, `Content-Type: audio/wav|audio/mpeg|audio/ogg|…`,
same parameters but in the query string: `?with_segments=true`.

### Response

```json
{
  "text": "Привет! Как дела?",
  "duration": 2.73,
  "language": "ru",
  "model": "v3_e2e_rnnt",
  "vad_backend": null,
  "processing_time": 0.61,
  "segments": [
    {"text": "Привет! Как дела?", "start": 0.0, "end": 2.73, "speaker": "SPEAKER_00"}
  ]
}
```

* `segments` — for long audio: segments `[start, end)` with text.
* `vad_backend` — which segmentation backend was used
  (set only for audio longer than 25 s).
* `segments[].speaker` — set when `with_speakers=true` (see below).

### Error codes

| Code | Reason                                                       |
|------|--------------------------------------------------------------|
| 400  | Audio could not be decoded / `with_speakers` requested but diarization is disabled |
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
        timeout=1800,
    )
print(r.json()["text"])
```

---

## Running locally

Requirements: Python ≥ 3.10, NVIDIA GPU with the driver installed (or CPU),
ffmpeg in PATH (or set `GIGAAM_FFMPEG_PATH=/path/to/ffmpeg`).

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

## Running with Docker

Docker is the recommended way to run giga-stt. All examples use
`docker compose -f docker/docker-compose.yml`, abbreviated as `dc` in this doc
(the underlying `docker` commands are shown where they differ). The compose file
drives the GPU via CDI but can run on CPU (see [CPU-only](#cpu-only)).

The image is built **from the remote git repository**
`https://git.embddr.xyz/np/giga-stt.git` (git build context), not from the local
working copy — so the sources must be pushed to that repo before an update is
picked up. The build is pinned to a ref with the `#ref` suffix in the compose file
(currently `#main`).

### Prerequisites

- Docker (≥ 24 with CDI support) and the `docker compose` plugin.
- NVIDIA driver + **nvidia-container-toolkit** on the host, with Docker CDI enabled
  (`features.cdi=true`). On NixOS that is `hardware.nvidia-container-toolkit.enable`;
  the compose file passes the GPU to the container via `--device nvidia.com/gpu=all`.
- Internet on the very first start (the ~450 MB GigaAM checkpoint is downloaded once).
- Optional: an `HF_TOKEN` for speaker diarization (see [Speaker diarization](#speaker-diarization)).

For the commands below, set:

```bash
alias dc='docker compose -f docker/docker-compose.yml'
```

### First run (from scratch)

```bash
# 1. build the image — pulls the sources from the remote git repo, installs the
#    torch 2.8 CUDA wheels, giga-stt and the pyannote diarization dependencies
cd giga-stt && dc build

# 2. start the service in the background
cd giga-stt && dc up -d
```

> The remote repo must be pushed up to date — the build reads the code from
git, not from the local checkout.

First start downloads the model and loads it, which takes a minute or two. Watch it
turn ready:

```bash
dc logs -f giga-stt          # follow the logs; exits with Ctrl-C
# you are ready when you see:
#   INFO giga_stt.engine: Model v3_e2e_rnnt ready (cuda)
```

Or poll the health endpoint until the GPU device appears:

```bash
until curl -fsS http://localhost:8000/health | grep -q '"device":"cuda"'; do sleep 3; done
curl -s http://localhost:8000/health
```

The model weights are cached in the docker volume `giga-stt-models` (`/models` inside
the container), so on subsequent starts the container comes up without re-downloading.
Verify transcription with the official sample:

```bash
wget -O example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"
curl -s -X POST http://localhost:8000/v1/transcribe -F file=@example.wav
```

### Check status and logs

```bash
dc ps                          # container state
curl -s http://localhost:8000/health

dc logs --tail 50 giga-stt     # last 50 lines
# interactive API docs: http://localhost:8000/docs
```

### Change the published port

The service is published on host port **8000** via the `ports:` mapping in
`docker/docker-compose.yml`. The mapping is `host:container` — the left part is
the host port, the right part is the port uvicorn listens on inside the container
(always 8000, no need to touch it).

```yaml
    ports:
      - "8000:8000"   # host(8000) : container(8000)
      # host(9000) : container(8000):
      - "9000:8000"
```

To publish on a different host port, edit the left number and recreate the
container (no image rebuild needed):

```bash
cd giga-stt && dc up -d
```

Now the API is at `http://localhost:9000` (and `/docs`, `/health` follow). The
internal port and the `HEALTHCHECK` are unchanged, so this is purely a host-side
change.

### Update to a new version

Push the new code to the remote repo, then rebuild and recreate the container:

```bash
cd giga-stt && dc build        # fetches the latest source from the remote git repo
cd giga-stt && dc up -d        # recreate the container with the new image
```

No local `git pull` is needed for the build — the source comes from the remote git
targeted by the compose `context:`. To pin a specific version instead of `#main`,
set the `#ref` suffix in `docker/docker-compose.yml` to a commit hash or a tag
(for example `.git#v1.2.0`).

`dc up -d` keeps the `giga-stt-models` volume, so the cached model survives updates.

### Stop / start / restart

```bash
dc stop        # stop the container (keeps it and the model volume)
dc start       # start it again (no rebuild, no re-download)
dc restart     # stop + start in one step

# or directly by the container name:
docker stop giga-stt && docker start giga-stt
```

If a container from an older image is still running, `dc up -d` recreates it — you
don't need to stop it by hand first.

### Remove

```bash
# stop and remove the container (keeps the `giga-stt-models` volume, so re-`up` is fast)
dc down

# also delete the model cache volume — the next start downloads the model again
# (e.g. on error, or to free disk space)
dc down -v

# delete the built images
cd giga-stt && docker rmi docker-giga-stt giga-stt:base
# exact image names/tags:  docker images | grep giga
```

### CPU-only

Comment out the `devices` block in `docker/docker-compose.yml` and set
`GIGAAM_DEVICE: cpu` in `environment:`:

```yaml
    devices:
      - nvidia.com/gpu=all   # <- comment out this block for CPU run
    environment:
      GIGAAM_DEVICE: cpu
```

Then recreate the container:

```bash
cd giga-stt && dc up -d --build
```

Inference on CPU is noticeably slower than on CUDA.

### Direct run without compose

```bash
# GPU
cd giga-stt && docker run -d --name giga-stt --device nvidia.com/gpu=all \
  -p 8000:8000 -v giga-stt-models:/models -e GIGAAM_DEVICE=cuda docker-giga-stt

# CPU
cd giga-stt && docker run -d --name giga-stt -p 8000:8000 \
  -v giga-stt-models:/models -e GIGAAM_DEVICE=cpu docker-giga-stt
```

### Speaker diarization

Diarization is built into the image by default (`INSTALL_DIARIZATION=1` build arg;
`GIGAAM_DIARIZATION: pyannote` is set in the compose file). To use it, provide an
`HF_TOKEN` that has accepted the terms for the gated model
`pyannote/speaker-diarization-community-1` (free, CC-BY-4.0) — add it under
`environment:` in `docker/docker-compose.yml`:

```yaml
    environment:
      HF_TOKEN: hf_xxxxxxxx
```

Without the token the ASR service still works; `diarization` in `/health` then shows
a 401 error instead of `ready`. With it, requests may set `with_speakers=true` to get
the per-segment `speaker` field.

---

## Configuration (environment variables)

| Variable                | Default           | Description |
|-------------------------|-------------------|-------------|
| `GIGAAM_MODEL`          | `v3_e2e_rnnt`     | Any GigaAM model: `v3_rnnt`, `v3_ctc`, `v3_e2e_ctc`, … |
| `GIGAAM_DEVICE`         | `auto`            | `auto` / `cuda` / `cpu` |
| `GIGAAM_FP16_ENCODER`   | `false`           | fp16 encoder + autocast. ⚠ On GTX 16xx GPUs (TU116/TU117, no tensor cores) cuDNN fp16 convolutions on long inputs (T ≳ 550 frames) produce NaN — do not enable there; on RTX/A100/H100 enabling it speeds up inference |
| `GIGAAM_CACHE_DIR`      | `~/.cache/gigaam` | GigaAM weights directory |
| `GIGAAM_MAX_UPLOAD_MB`  | `100`             | Maximum upload size |
| `GIGAAM_MAX_AUDIO_SEC`  | `7200`            | Safety cap on duration |
| `GIGAAM_MAX_CONCURRENCY`| `1`               | Concurrent inferences (the rest wait in the queue) |
| `GIGAAM_QUEUE_TIMEOUT_SEC`| `300`           | Queue wait time before a 503 |
| `GIGAAM_CHUNK_MAX_SEC`  | `22`              | Chunk length for long audio (model limit is 25 s) |
| `GIGAAM_FR_BATCH_SIZE`  | `4`               | Chunk batch size (increase on GPUs with more VRAM) |
| `GIGAAM_VAD_BACKEND`    | `auto`            | `auto` / `pyannote` / `silero` / `energy` |
| `GIGAAM_API_KEY`        | —                 | If set — requests must send `X-API-Key: <key>` or `Bearer <key>` |
| `GIGAAM_FFMPEG_PATH`    | —                 | Path to the ffmpeg binary (if not in PATH) |
| `GIGAAM_DIARIZATION`    | `off`             | `off` / `pyannote` — speaker diarization backend (`with_speakers=true`) |
| `GIGAAM_DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Diarization pipeline (gated, CC-BY-4.0) |
| `HF_TOKEN`              | —                 | HF token for the pyannote longform and the gated diarization model |

---

## Long audio (> 25 s): segmentation backends

The GigaAM model works with fragments up to 25 s, so the service cuts long
recordings into chunks and stitches the results together:

1. **pyannote** (the official repository path) — VAD `pyannote/segmentation-3.0`.
   Requires the `pip install -e ".[longform]"` dependencies, an HF token, and
   accepted terms for the gated model. The extra pins `pyannote.audio==4.0.*`,
   which works with the pinned torch 2.8.0 (verified).
2. **silero** — lightweight VAD [silero-vad](https://github.com/snakers4/silero-vad)
   (part of the base dependencies); the recommended option without gated access.
3. **energy** — built-in fallback without external models: chunk boundaries are
   placed at local minima of the signal energy.

`auto` picks the first available backend. Cutting quality only affects phrase
boundaries — pyannote/silero keep words intact by cutting at pauses.

## Speaker diarization

Optional: each transcribed segment gets the dominant speaker (`SPEAKER_00`, …).
Powered by [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
(gated model — a free HF account with accepted terms + `HF_TOKEN`; CC-BY-4.0, ~34 MB).

```bash
pip install -e ".[diarization]"   # pyannote.audio 4.0.x (+ torchcodec 0.7.x for torch 2.8)
export GIGAAM_DIARIZATION=pyannote HF_TOKEN=hf_...
```

Then request `with_speakers=true` (multipart field or query parameter). The backend
loads in the background like the ASR models — `/health` shows its status
(`off` / `loading` / `ready` / `error: …`); until it is ready, `with_speakers=true`
returns 503.

* The pipeline runs once per request over the whole audio (~0.1× realtime on a
  GTX 1660 SUPER: ~1–1.5 min per hour of audio, +33 MB VRAM).
* Speaker labels are local to a single request — they are not stable across requests.
* A speaker is assigned per ASR segment (the one with the largest overlap): words of
  a second speaker inside one segment cannot be attributed (no word timestamps).
* Works with Russian speech.

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

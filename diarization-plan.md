# План: диаризация спикеров (pyannote/speaker-diarization-community-1)

## Статус: реализовано и проверено (14.09.2026)

- Шаги 1–11 выполнены. Смоук-тест: venv (torch 2.8.0+cu128 + extra + editable-установка),
  сервис на :8010 — все сценарии пройдены (см. таблицу ниже).
- Найден и исправлен баг маппинга: доминирующий спикер считался как максимум по
  **одному** turn'у, а не как **сумма** перекрытий по спикеру (`_dominant_speaker`).
- Не проверено: Nix-сборка (`nix build .#giga-stt`) — требует пересборки пакета;
  pyannote-VAD (segmentation-3.0) отдаёт 403 с текущим HF-токеном (terms не приняты
  для этой модели) — сервис корректно падает на silero, на диаризацию не влияет.

Тесты (GTX 1660 SUPER):

| Сценарий | Результат |
|---|---|
| 11.3 с, ru, 1 спикер, `with_speakers=true` | 200, `SPEAKER_00` |
| 65 с, ru, 2 спикера (чанки silero) | 200, `01, 01, 01, 00` — совпадает с прямым проходом пайплайна |
| en (Granite) + `with_speakers=true` | 200, `SPEAKER_00` |
| без `with_speakers` | `speaker: null`, без накладных расходов |
| raw-endpoint + `with_speakers=true` | 200 |
| `GIGAAM_DIARIZATION=off` + `with_speakers=true` | 400 с detail |
| `/health` | `diarization: off \| ready` |
| время (65 с аудио, ASR+диаризация) | ~5.9 с |


Цель: опциональная диаризация поверх существующего STT. Запрос с `with_speakers=true`
добавляет поле `speaker` к каждому сегменту транскрипции. Работает и для `ru` (GigaAM),
и для `en` (Granite).

## Проверено e2e на этой машине (GTX 1660 SUPER, torch 2.8.0+cu128)

- pyannote.audio **4.0.7** + torchcodec **0.7.0** (пин под torch 2.8; 0.10+ — для torch 2.10) — работают.
- Загрузка пайплайна: 2.5 с из HF-кэша (11.7 с при первом запуске), **VRAM 33 MB**.
- Скорость: 11.3 с аудио → 0.71 с (~0.1× realtime) → ~+1–1.5 мин на час аудио.
- Ввод: `pipeline({"waveform": tensor(1, N), "sample_rate": 16000})` — то, что уже есть в движках.
- Вывод: `DiarizeOutput.exclusive_speaker_diarization` (Annotation без overlap'ов,
  метки `SPEAKER_00/01/…`) — создан именно для сопоставления с таймстампами ASR.
- Модель: gated (бесплатно, HF_TOKEN + terms), CC-BY-4.0, **33.7 MB**,
  зависимость `pyannote.audio: 4.0.0` → пин `4.0.*` подходит.
- Импорт-пути `gigaam.vad_utils` (`pyannote.audio.core.task`, `VoiceActivityDetection`)
  работают под 4.0.7 → pyannote-VAD для нарезки длинного аудио остаётся жив.
- torchcodec импортится в pyannote.audio **лениво** (try/except + warning); без него
  in-memory waveform всё равно работает. Требует FFmpeg-библиотеки в LD_LIBRARY_PATH.
- Nix: в зафиксированных флейком nixpkgs уже есть `pyannote-audio 4.0.4` (полный стек)
  и `torchcodec 0.11.1` → одна строка в package.nix.

## Шаги

1. **`pyproject.toml`** — новый extra `diarization = ["pyannote.audio==4.0.*", "torchcodec==0.7.*"]`.
2. **`giga_stt/diarization.py`** (новый) — ленивый синглтон пайплайна: фоновая загрузка
   с `HF_TOKEN`, `.to(device)`, `diarize(waveform) -> Annotation` (exclusive),
   `assign_speakers(ann, segments)` — доминирующий спикер по перекрытию, `status()` для /health.
3. **`giga_stt/config.py`** — env: `GIGAAM_DIARIZATION=off|pyannote` (default `off`),
   `GIGAAM_DIARIZATION_MODEL` (default `pyannote/speaker-diarization-community-1`).
4. **`giga_stt/schemas.py`** — `SegmentOut.speaker: Optional[str] = None`.
5. **`giga_stt/engine.py` + `giga_stt/granite_engine.py`** — параметр `with_speakers`:
   после получения сегментов выполнить диаризацию по полному wav (16 kHz mono уже
   загружен в обеих ветках) и приписать `speaker` к сегментам.
6. **`giga_stt/app.py`** — query/form-параметр `with_speakers` (default false) на обоих
   endpoints; 400 если backend выключен, 503 если грузится/ошибка; `/health` — поле
   `diarization`; запуск `start_loading()` при старте.
7. **`docker/`** — build-arg `INSTALL_DIARIZATION` (pip extra), env `GIGAAM_DIARIZATION`,
   `PYANNOTE_METRICS_ENABLED=0`.
8. **`nix/package.nix`** — `python3Packages.pyannote_audio` в propagatedBuildInputs;
   **`nix/module.nix`** — env-переменные сервиса.
9. **`README.md`** — раздел «Speaker diarization»: включение, API, цены, Nix/Docker.
10. **Смоук-тест** — venv в /tmp (torch 2.8 + extra + `pip install -e .`), запуск сервиса,
    curl с `with_speakers=true`, проверка speaker'ов в segments.

## API

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `with_speakers` | bool | `false` | добавить `speaker` к сегментам (только если backend включён) |

Ответ: `segments[].speaker: "SPEAKER_00" | …` (null, если сегмент не перекрывается ни с одним turn'ом).

Ошибки:
- `400` — `with_speakers=true`, но `GIGAAM_DIARIZATION=off` (или extra не установлен).
- `503` — backend грузится или не загрузился (причина в `detail`).

## Ограничения (задокументировать)

- Спикер присваивается на уровне ASR-сегмента (доминирующий по перекрытию): внутри
  одного сегмента слова второго спикера атрибутировать нельзя — модели не дают word
  timestamps.
- Метки `SPEAKER_00/01` — локальные для одного запроса (identity между запросами не
  гарантируется).
- Диаризация выполняется на полном аудио одним проходом (turn'ы пересекают чанки ASR).

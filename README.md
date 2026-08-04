# Image Filterer

Hero-shot selection for event photography. Point it at a folder of unlabeled
photos from a keynote or a conference; it ranks them the way the photo team
would, collapses burst sequences down to one frame each, and serves a browser UI
for finding the keeper — by relevance ("the speaker on stage"), by shot scale, or
by subject.

A shoot that produces 3,300 frames becomes ~730 ranked moments. The team reviews
the top of that list instead of the whole card.

```
   folder of photos                                    browser UI
   ────────────────                                    ──────────
   3,316 JPEGs   ──►  features ──►  ranker ──►  bursts  ──►  733 ranked moments
                      (cached)      (MLP)      (EXIF+emb)     search · filters
```

---

## Requirements

- **Python 3.10+**
- **An NVIDIA GPU** (~8 GB VRAM). CPU works and is correct, but feature
  extraction is roughly 20× slower — a 3,000-image shoot goes from minutes to
  hours. Developed on an RTX A6000.
- **~5 GB disk** for model weights and the feature cache; the cache grows about
  **25 MB per 1,000 images**.

---

## Setup

```bash
git clone <your-remote>/Image-Filterer.git
cd Image-Filterer

python -m venv .venv && source .venv/bin/activate

# 1) Install PyTorch matching YOUR CUDA version first — the generic wheel is
#    often CPU-only. See https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) Install the package and its remaining dependencies.
pip install -e .

# 3) Fetch the third-party model weights (~700 MB, one time).
python -m scripts.fetch_assets
```

`fetch_assets` downloads four files into `~/.local/share/image-filterer/assets`:

| File | Size | Used for |
|---|---|---|
| `FaRL-Base-Patch16-LAIONFace20M-ep64.pth` | 652 MB | face embedding |
| `face_detection_yunet_2023mar.onnx` | 227 KB | face detection |
| `face_landmarker.task` | 3.7 MB | eyelid / mouth landmarks |
| `yolov8s-pose.pt` | 23 MB | person detection (subject + hero tags) |

The two encoder backbones (SigLIP2-SO400M, and the ViT-B/16 skeleton FaRL loads
into) are pulled from HuggingFace by `open_clip` on first use — no action needed,
but the first run will pause to download ~1.5 GB.

**Migrating from an older checkout?** Reuse the weights and the feature cache you
already have — nothing needs recomputing, since cache keys are content hashes:

```bash
python -m scripts.fetch_assets --from-cache /path/to/old/v3/cache
export IMAGE_FILTERER_CACHE_DIR=/path/to/old/v3/cache
```

Verify the install:

```bash
pip install -e ".[dev]" && pytest        # smoke tests, no GPU or weights needed
```

---

## Configuration

Everything is resolved from the environment, so the same code runs from a
checkout, a container, or an install. All are optional.

| Variable | Default | What it controls |
|---|---|---|
| `IMAGE_FILTERER_DATA_ROOT` | `~/.local/share/image-filterer` | Run storage + the SQLite registry |
| `IMAGE_FILTERER_CACHE_DIR` | `$IMAGE_FILTERER_DATA_ROOT/cache` | Feature cache (share it across machines) |
| `IMAGE_FILTERER_ASSET_DIR` | `$IMAGE_FILTERER_DATA_ROOT/assets` | Third-party weights |
| `IMAGE_FILTERER_MODEL_PATH` | `image_filterer/model/ranker.pt` | The production ranker |
| `IMAGE_FILTERER_TRAIN_ROOT` | `./dataset` | Labeled data, training only |

Run storage deliberately defaults **outside the repo** so that git operations,
cleanups, or a `rm -rf` in the working tree can never destroy live runs.

Model behavior lives in `image_filterer/config.py` — every tunable is a
dataclass field with a comment explaining which experiment settled its value. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing any of them.

---

## Running in production

The repo ships a trained model, so serving works immediately after setup:

```bash
image-filterer-server --host 0.0.0.0 --port 8600
# or: python -m image_filterer.server --host 0.0.0.0 --port 8600
```

The startup banner prints the URL to share with the team:

```
Image Filterer (model=ranker.pt)
  data:   /home/you/.local/share/image-filterer   (set IMAGE_FILTERER_DATA_ROOT to change)
  local:  http://127.0.0.1:8600/
  remote: http://<this-host-ip>:8600/
```

Share the `remote:` URL. If a colleague can't connect, it is almost always the
host firewall — open the port once (the rule persists across reboots):

```bash
sudo ufw allow 8600/tcp
# or let the server try at launch (needs passwordless sudo; harmless if it fails)
image-filterer-server --host 0.0.0.0 --port 8600 --open-firewall
```

If `ping <host>` fails too — not just the port — the two machines aren't mutually
routable (common with split-tunnel VPNs). No firewall change fixes that; tunnel
instead:

```bash
ssh -L 8600:localhost:8600 you@<host>     # then open http://localhost:8600
```

### Adding a shoot

The UI starts empty. Open the upload panel and pick one of two paths:

- **Folder on server (fast)** — paste a path that already exists on the machine
  (or a mounted share). Nothing is uploaded; images are read and served in place.
  Use this whenever the photos are already on or near the box.
- **Upload from device** — pick a local folder; the browser sends it in batches
  of 200 files. Uploading doesn't hold the ingest lock, so several people can
  upload at once — but every original crosses the network, so for large or
  RAW-heavy shoots the first option is much faster.

Either way a progress bar tracks extraction → scoring → clustering → tagging, and
the run loads automatically when it finishes. Switch runs with the header
dropdown.

### Behavior under load

- **Ingestion is single-flight.** A second concurrent ingest gets a `409` with a
  clear message. Everyone polling the UI sees a sticky bar with the active run's
  progress and a **Cancel** button.
- **Cancel is cooperative** — the flag is checked between feature chunks and at
  phase boundaries, so it stops within a chunk or two. (A background GPU thread
  can't be safely hard-killed.)
- **Reads scale.** Browsing and search are read-only and run concurrently;
  verified with 8 simultaneous searches. The text encoder is built once and
  shared.
- **Tuned for VPN use** — grid thumbnails are ~6–8 KB, and opening a photo
  streams a ~1600px preview (~100–300 KB) rather than the 8 MB original, with an
  "open full-res ↗" link when you want it. Both are sent `Cache-Control:
  immutable`.

For heavier load, put it behind gunicorn and a reverse proxy — the app factory is
`image_filterer.server:create_app`. **There is no authentication**; the service
assumes a trusted network. Add auth at the proxy if you expose it.

---

## Training a model

The shipped `ranker.pt` was trained on one labeled keynote shoot. Retrain when
you have labels from a new event, or when you want to fold in corrections.

### 1. Lay out the labeled data

One folder per label, images directly inside (not nested):

```
dataset/
  Good/          # keepers — the positives
  Bad/           # rejects — the negatives
  Top-5to10/     # the must-find gold shots; also positives, weighted no differently
  Questionable/  # ignored entirely (see below)
  Debatable/     # ignored entirely
```

`Questionable` and `Debatable` are treated as **genuinely unlabeled** (label −1)
and are excluded from training. They are not negatives — folding them into `Bad`
teaches the model that borderline-good frames are rejects, which is the opposite
of what you want.

### 2. Train

```bash
IMAGE_FILTERER_TRAIN_ROOT=./dataset image-filterer-train
# or: python -m image_filterer.train --train-root ./dataset
```

This writes `image_filterer/model/ranker.pt` and `training_summary.json`. The
first run extracts features for every labeled frame (the slow part); later runs
hit the cache and take a minute or two.

Expect output like:

```
[train] 2604 unique labeled frames (pos=400, neg=2204)
[train] features (2604, 1671) on cuda
[train] saved .../model/ranker.pt
[train] holdout AUC=0.9595  mean_cv_auc=0.9346  (eval-only; model uses all data)
```

### 3. Read the numbers

`holdout AUC` is the honest one — the probability that a random keeper outranks a
random reject on frames the model never saw. **Below ~0.85, don't ship it**; look
for label noise or too few positives first.

The shipped weights come from a final fit on **100% of the labeled data**. The
holdout split exists only to produce that AUC. This is deliberate: in production
there's nothing to evaluate against, so holding data back only makes the model
worse.

Frames are content-deduped by sha1 before training, and a positive label wins any
collision — the same image copied into both `Good/` and `Bad/` counts as `Good`.

### 4. Roll out

Restart the server. Existing runs keep their old scores; re-ingest a folder to
score it with the new model. Keep the previous checkpoint around — swapping back
is just `IMAGE_FILTERER_MODEL_PATH=/path/to/old.pt`.

The checkpoint stores the feature-group flags it was trained with, and ingestion
rebuilds the input vector from *those*, not from the current config. An older
checkpoint keeps scoring correctly even after config changes.

---

## What the filters mean

| Filter | Values | How it's derived |
|---|---|---|
| **Search** | free text | SigLIP2 text tower against cached image embeddings — same joint space, no re-embedding. Warm queries ~50 ms. |
| **Shot** | wide / medium / close | Zero-shot SigLIP2 wide↔close axis. This is *framing scale*, not face size — "close" includes a frame-filling robot. |
| **Subject** | people / stage | Largest YOLO person box vs. frame area. "stage" = empty stage, distant crowd, or a slide — no main focus. |
| **★ Hero** | on / off | One dominant person, no second prominent person, clean dark background. "The speaker alone, nothing behind them." |

All of them AND together, and all combine with search.

---

## Run outputs

Each ingested folder becomes a run under `$IMAGE_FILTERER_DATA_ROOT/runs/run0001_<name>/`:

| File | Contents |
|---|---|
| `ranked.csv` | one row per unique frame: scores, burst columns, `shot_type`, `subject_class`, `is_hero`, technical metrics |
| `bursts.csv` | one row per burst (its representative), ranked |
| `search_index.npy` | full-frame embeddings, row-aligned to `ranked.csv` |
| `config.json` | the exact config used |
| `uploads/` | the images, for browser uploads only (absent for "folder on server" runs) |

`ranked.csv` is the export path — hand it to anyone who wants the ordering
without the UI.

---

## HTTP API

| Route | Purpose |
|---|---|
| `GET /` | the UI |
| `GET /api/state` | current run + all runs |
| `GET /api/runs` | runs only |
| `POST /api/upload/start` · `/chunk` · `/finish` | batched browser upload → creates a run |
| `POST /api/ingest_path` | `{path, name}` — ingest a folder already on the server |
| `GET /api/ingest/status?run_id=` | progress for one ingest |
| `GET /api/active` | the ingest running right now |
| `POST /api/cancel_ingest` | cooperative cancel |
| `POST /api/select_run` | `{run_id}` — load a ready run |
| `GET /api/bursts?shot=&subject=&hero=&offset=&limit=` | ranked bursts |
| `GET /api/burst/<id>` | frames within a burst |
| `GET /api/search?q=&shot=&subject=&hero=` | semantic search |
| `GET /img` · `GET /thumb?w=` | image bytes (path-allowlisted to the run's tree) |

`shot` is comma-separated multi-select (`wide,medium,close`); `subject` is a
single value (`people`\|`stage`); `hero=1` restricts to hero shots.

---

## Robustness

- **Bad files never kill a run.** Every file is checked for decodability first;
  corrupt, truncated, or unsupported files are skipped and counted in the
  finish message. A run only fails if *nothing* decoded.
- **Upload corruption is auto-repaired.** Some browser/proxy combinations prepend
  a stray `\r\n` to uploaded bodies (a leaked multipart separator) and break
  decoding. The server strips junk before the real image header, restoring the
  bytes *and* the sha1 — so the feature cache still hits.
- **RAW is not supported.** CR3/ARW files are skipped as undecodable. Export JPEGs.

---

## Project layout

```
image_filterer/
  config.py       every tunable, with the reasoning behind each default
  encoders.py     frozen SigLIP2 / FaRL encoders
  face.py         YuNet detection + MediaPipe landmarks + quality scalars
  features.py     feature extraction, cached by content hash
  cache_io.py     the content-addressed cache
  ranker.py       pairwise RankNet MLP — train, score, save/load
  technical.py    sharpness / exposure / eye-openness metrics
  bursts.py       EXIF + embedding burst clustering
  shots.py        wide/medium/close tagging
  scene.py        YOLO person detection → subject + hero tags
  pipeline.py     shared steps: burst columns, search index, CSV writers
  ingest.py       inference-only ingestion of one folder
  train.py        offline training entry point
  server.py       Flask app + HTTP API
  db.py           SQLite run registry
  templates/      the single-page UI
  model/          the shipped ranker checkpoint
scripts/          asset fetcher
tests/            smoke tests
docs/             architecture and tuning notes
```

---

## Known limits

- **One event's worth of training data.** The model has never been evaluated
  against a second shoot; cross-event generalization is unmeasured. Treat the
  0.96 AUC as "on material resembling that one shoot."
- **Scene thresholds were set by eye** on that same dataset. The raw detector
  outputs are cached, so re-tuning costs nothing but a re-ingest.
- **Cross-camera moments split.** Two photographers shooting the same gesture
  from different angles produce two bursts.
- **Runs are never garbage-collected.** `RunDB.delete` exists but isn't wired to
  the UI, and a registry row whose folder was deleted by hand still shows in the
  dropdown.
- **No authentication.**

# Image Filterer


> ### ⚠️ This is an alpha
>
> **The ranking will make mistakes.** It learned from one labeled shoot, about
> 2,600 photos and it has never been tested against a different event. Please don't
> trust the rankings over your own intuition, or assume the best results will always
> be at the very top
>
> **The feature list is not final.** Everything below can change. If something
> doesn't work, works badly, or is missing, that's useful information please
> say so. Right now I am mostly adding features to see what sticks. Eventually
> slimming the system down is a priority
>
> Many design decisions were made by me with a limited understanding of your
> exact workflow. If something feels wrong/unintuitive definitely give feedback.

---

## What you can do

| | |
|---|---|
| **Browse by rank** | Best-guess keepers first, one tile per moment. |
| **Browse by time** | Switch the **Sort** dropdown to *Chronological* to walk the event start to finish. |
| **Search in plain English** | Type "person at a podium", "robot on stage", "wide shot of the crowd". No tags or keywords needed. |
| **Filter** | By framing (wide / medium / close), by subject (people / stage), or to solo hero shots only. |
| **Open a moment** | Click a tile to see every frame in that burst, ordered best-first, and step through with ← / →. |
| **Star the ones you want** | Click ☆ on any tile or frame. Stars are saved and shared with everyone looking at the same shoot. |
| **Download** | One photo, your starred set, or a hand-picked selection using select mode |
| **Resize the grid** | Make thumbnails bigger or smaller with the **Size** control or the `+` / `−` keys. |

---

## Getting started

1. **Open the link** your admin shares — something like `http://<server>:8600/`.
2. **Pick a shoot** from the dropdown at the top left.
3. **Browse.** The best-guess keepers come first.

### Adding a shoot

Click **＋ Upload folder**. Two options:

- **Folder on server** — paste a path that already exists on the machine (or a
  mounted drive). Nothing is copied and it starts almost immediately. **Use this
  whenever you can** — it is dramatically faster.
- **Upload from device** — pick a folder on your own computer. Every photo has
  to cross the network, so a large shoot can take a long time.

Either way a progress bar tracks the work and the shoot loads when it's done.
Only one shoot can be processed at a time; if someone else is already going,
you'll see their progress and can wait or cancel.

> RAW files (CR3, ARW) are **not supported** and get skipped. Export JPEGs.

### Starring and downloading

Hover a tile and click **☆** to star it. Click **★ Starred** in the header to
see only the starred frames, where you'll also find:

- **⬇ Download all N starred** — everything starred, as one `.zip`
- **☆ Clear all N** — unstar everything (asks first; this affects everyone)

For a one-off, click **⬇** in the bottom-right of any tile. To grab a specific
set, click **☑ Select**, click the tiles you want, then **⬇ Download N
selected**.

**Every download is the full-resolution original file** — the same bytes off the
card. The images on screen are small stand-ins so the page loads quickly, but
that's never what you get when you download.

### Keyboard shortcuts

| Key | Does |
|---|---|
| `←` `→` | Previous / next moment |
| `S` | Star the photo you're looking at |
| `D` | Download the photo you're looking at |
| `Esc` | Close |
| `+` `−` | Bigger / smaller thumbnails |
| `0` | Reset thumbnail size |

---

## What the words mean

**Burst** — a group of near-identical frames shot within a few seconds of each
other: the photographer holding down the shutter through one gesture. The grid
shows **one tile per burst**, so eight shots of the same handshake take up one
slot instead of eight. Click it to see all eight. 

**Score** — the number in the bottom-right of a tile. It is **not** a percentage,
a grade, or a quality rating, and the units mean nothing on their own. It is only
useful for *comparison*: a higher-scoring photo is one the system thinks is more
likely to be a keeper than a lower-scoring one. A score of 6.7 is not "twice as
good" as 3.4.

**#Rank** — where that moment sits in the ordering. `#1` is the system's best
guess at the strongest shot of the whole shoot.

**Representative** — within a burst, the one frame the system picked as the best
of that group. It's the one shown on the tile.

**Hero** — one person clearly dominating the frame, nobody else prominent, and a
clean dark background behind them. Think a portrait of a speaker with nothing
distracting behind.

**Shot: wide / medium / close** — how tight the framing is, *not* how big a face
is. A frame-filling shot of a robot counts as "close" even though there's no
person in it.

**Subject: people / stage** — whether there's a prominent person in the
foreground. "Stage" means an empty stage, a distant crowd, or a slide — no clear
main subject.

**Relevance** — only appears when you search. How well that photo matches the
words you typed. Search results are ordered by this instead of by rank.

**Shoot / run** — one processed folder of photos. Switch between them with the
dropdown at the top.

---

## Common questions

**A bad photo is ranked near the top. Why?**
It's a first pass trained on a limited set. Blinks, soft focus, and awkward
mid-gesture frames do slip through — especially when the subject is small in the
frame, where the system's read on faces is least reliable. Star what's good and
ignore the rest; the ordering is meant to save you scrolling, not to be trusted
blindly.

**Where did the rest of my photos go?**
Nothing is deleted. Near-identical frames are grouped into one burst and only the
best one is shown. Click the tile to see the whole group — the count is on the
tile ("8 frames").

**Two shots of the same moment appear as separate tiles.**
Expected. Frames from two different cameras are never merged, so two
photographers shooting the same gesture produce two bursts.

**Does starring change the ranking?**
No. Stars are just a list you're keeping. They don't teach the system anything —
though a version that does is something we'd like to build, so tell us if you'd
find it useful.

**Can other people see my stars?**
Yes. Stars are shared by everyone viewing the same shoot, so the team can build
one pick list together. There's no per-person list, and no undo on **Clear all**.

**Is the photo I download the full-quality one?**
Yes, always — the original file, untouched. What you see in the grid and preview
are smaller copies made for speed.

**Search isn't finding something I know is there.**
It matches on what a photo *looks like*, not on names or text. It has no idea who
anyone is. Describe the scene ("two people shaking hands on stage") rather than
naming a person or a company.

**A great photo is tagged "stage" instead of "people".**
The people/stage and hero cut-offs were set by eye, and they're among the
roughest parts of the system. Worth reporting when it looks wrong.

---
---

# For developers and operators

Everything below is setup, deployment, and internals. **If you're using the
browser page, you can stop here.**

## Requirements

- **Python 3.10+**
- **An NVIDIA GPU** (~8 GB VRAM). CPU works and is correct, but feature
  extraction is roughly 20× slower — a 3,000-image shoot goes from minutes to
  hours. Developed on an RTX A6000.
- **~5 GB disk** for model weights and the feature cache; the cache grows about
  **25 MB per 1,000 images**.

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
python -m scripts.fetch_assets --from-cache /path/to/old/cache
export IMAGE_FILTERER_CACHE_DIR=/path/to/old/cache
```

Verify the install:

```bash
pip install -e ".[dev]" && pytest        # smoke tests, no GPU or weights needed
```

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
cleanups, or a `rm -rf` in the working tree can never destroy live runs. In
particular, `git clean -xdf` would delete an in-tree data root.

Model behavior lives in `image_filterer/config.py` — every tunable is a
dataclass field with a comment explaining which experiment settled its value. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing any of them.

## Running

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
- **Tuned for remote viewing** — grid thumbnails are ~6–8 KB at the default zoom
  and opening a photo streams a ~1600px preview (~150–300 KB) rather than the
  8 MB original. Both are sent `Cache-Control: immutable`. The thumbnail cache is
  bounded by bytes as well as entries, since zoomed-in grids request larger
  renditions.
- **Downloads stream.** Bulk downloads are zipped (STORED, not deflated — JPEGs
  don't compress) and streamed, so a multi-GB archive costs almost no memory.

For heavier load, put it behind gunicorn and a reverse proxy — the app factory is
`image_filterer.server:create_app`. **There is no authentication**; the service
assumes a trusted network. Add auth at the proxy if you expose it.

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

## How the filters are derived

| Filter | Values | How it's derived |
|---|---|---|
| **Search** | free text | SigLIP2 text tower against cached image embeddings — same joint space, no re-embedding. Warm queries ~50 ms. |
| **Shot** | wide / medium / close | Zero-shot SigLIP2 wide↔close axis. This is *framing scale*, not face size. |
| **Subject** | people / stage | Largest YOLO person box vs. frame area. |
| **★ Hero** | on / off | One dominant person, no second prominent person, clean dark background. |
| **Sort** | rank / time | Rank uses the burst representative's score; time uses EXIF `DateTimeOriginal` (falling back to mtime), cached per run. |

All of them AND together, and all combine with search.

## Run outputs

Each ingested folder becomes a run under `$IMAGE_FILTERER_DATA_ROOT/runs/run0001_<name>/`:

| File | Contents |
|---|---|
| `ranked.csv` | one row per unique frame: scores, burst columns, `shot_type`, `subject_class`, `is_hero`, `captured_at`, technical metrics |
| `bursts.csv` | one row per burst (its representative), ranked |
| `search_index.npy` | full-frame embeddings, row-aligned to `ranked.csv` |
| `stars.json` | starred frame paths (created on first star) |
| `captured_at.json` | EXIF timestamp cache, backfilled for runs predating the `captured_at` column |
| `config.json` | the exact config used |
| `uploads/` | the images, for browser uploads only (absent for "folder on server" runs) |

`ranked.csv` is the export path — hand it to anyone who wants the ordering
without the UI.

## HTTP API

| Route | Purpose |
|---|---|
| `GET /` | the UI |
| `GET /api/state` · `/api/runs` | current run + all runs |
| `POST /api/upload/start` · `/chunk` · `/finish` | batched browser upload → creates a run |
| `POST /api/ingest_path` | `{path, name}` — ingest a folder already on the server |
| `GET /api/ingest/status?run_id=` | progress for one ingest |
| `GET /api/active` · `POST /api/cancel_ingest` | the running ingest; cooperative cancel |
| `POST /api/select_run` | `{run_id}` — load a ready run |
| `GET /api/bursts?shot=&subject=&hero=&sort=&offset=&limit=` | ranked bursts |
| `GET /api/burst/<id>` | frames within a burst |
| `GET /api/search?q=&shot=&subject=&hero=&starred=` | semantic search |
| `GET /api/stars` · `POST /api/star` · `POST /api/stars/clear` | read / toggle / clear stars |
| `GET /api/starred?shot=&subject=&hero=&sort=` | starred frames, one entry per frame |
| `GET /download?path=` | one original, as an attachment |
| `POST /api/download/prepare` | `{scope:"starred"}` or `{paths:[…]}` → `{token, count, bytes}` |
| `GET /api/download/zip?token=` | streams the prepared archive |
| `GET /img` · `GET /thumb?w=` | image bytes (path-allowlisted to the run's tree) |

`shot` is comma-separated multi-select (`wide,medium,close`); `subject` is a
single value (`people`\|`stage`); `hero=1` restricts to hero shots; `sort=time`
orders chronologically instead of by rank.

## Robustness

- **Bad files never kill a run.** Every file is checked for decodability first;
  corrupt, truncated, or unsupported files are skipped and counted in the
  finish message. A run only fails if *nothing* decoded.
- **Upload corruption is auto-repaired.** Some browser/proxy combinations prepend
  a stray `\r\n` to uploaded bodies (a leaked multipart separator) and break
  decoding. The server strips junk before the real image header, restoring the
  bytes *and* the sha1 — so the feature cache still hits.
- **Downloads are path-allowlisted** to the loaded run's tree, the same boundary
  as image serving.
- **RAW is not supported.** CR3/ARW files are skipped as undecodable.

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

## Known limits

- **One event's worth of training data.** The model has never been evaluated
  against a second shoot; cross-event generalization is unmeasured. Treat the
  0.96 AUC as "on material resembling that one shoot."
- **Scene thresholds were set by eye** on that same dataset. The raw detector
  outputs are cached, so re-tuning costs nothing but a re-ingest.
- **Cross-camera moments split.** Two photographers shooting the same gesture
  from different angles produce two bursts.
- **Stars are global per run.** No per-user lists, and `Clear all` has no undo.
- **Runs are never garbage-collected.** `RunDB.delete` exists but isn't wired to
  the UI, and a registry row whose folder was deleted by hand still shows in the
  dropdown.
- **No authentication.**

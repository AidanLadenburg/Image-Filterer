# Image Filterer


> ### ⚠️ This is an alpha
>
> **The ranking will make mistakes.** It learned from a few labeled shoots, about
> 2,600 photos and it has never been tested against a different event. Please don't
> trust the rankings over your own intuition, or assume the best results will always
> be at the very top
>
> **The feature list is not final.** Everything below can change. If something
> doesn't work, works badly, or is missing, that's useful information, please
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
| **Browse by time** | Switch the **Sort** dropdown to *Chronological* to walk the event start to finish — each tile shows its burst's earliest frame. |
| **Browse by stars** | Switch **Sort** to *Most starred* to bring moments with your team's picks to the top — each tile shows the most-starred frame in its burst. |
| **Search in plain English** | Type "person at a podium", "robot on stage", "wide shot of the crowd". No tags or keywords needed. |
| **Filter** | By framing (wide / medium / close), by subject (people / stage), or to solo hero shots only. |
| **Ungroup** | Click the **Ungroup** pill to turn off burst grouping — one tile per photo instead of one per burst, everything else (search, filters, sort, quality, stars) still applies. |
| **Open a moment** | Click a tile to see every frame in that burst, ordered best-first or chronilogical, and step through with ← / →. |
| **Star the ones you want** | Click ☆ on any tile or frame. Stars are saved and shared with everyone looking at the same shoot — a colleague's star shows up on your screen within a few seconds, no reload needed. |
| **Export** | Original photos, individually or in bulk, sent to the local folder or browser download destination chosen in Settings. |
| **Quality slider** | Hide everything below a chosen band so only the strongest photos remain. |
| **Resize the grid** | Make thumbnails bigger or smaller with the **Size** control or the `+` / `−` keys. |
| **Watch a folder** | Point it at a shared folder during a live event and photos are added as they arrive. |
| **Delete a shoot** | The 🗑 next to the shoot dropdown, with a confirmation that spells out what's removed. |

---

## Getting started

1. **Open the link** your admin shares — something like `http://<server>:8600/`.
2. **Pick a shoot** from the dropdown at the top left.
3. **Browse.** The best-guess keepers come first.

### Adding a shoot

Click **＋ Upload folder**. For a shoot that's already finished:

- **Upload from device** — pick a folder on your own computer. Every photo has
  to cross the network, so a large shoot can take a long time.

If the photos are already on the server (or a mounted drive), use
[Watch a folder](#watching-a-folder-during-a-live-event) instead and point it at
the path — nothing is copied, and it works for a finished shoot as well as a live
one.

A bar at the top tracks the work from the moment it starts, and its **Reload**
button lights up when the new photos are live. Only one shoot is processed at a
time; if someone else is already going, you'll see their progress and can wait or
cancel.

> **RAW works.** CR3, ARW, NEF and friends are read through the preview the
> camera embeds in them, so they rank and search alongside your JPEGs. If a shot
> exists as both a RAW *and* a JPEG, the JPEG is used and the RAW skipped, so
> nothing appears twice. Downloading a RAW gives you the untouched original file.

### Watching a folder during a live event

Instead of uploading at the end, point the system at a folder everyone offloads
to (a LucidLink mount, a NAS — anything the server can read). Open **＋ Upload
folder → Watch a folder**, paste the path, and hit **Start watching**.

From then on the photo team just offloads cards as usual. New photos are picked
up and ranked within a minute or two, and a green bar appears for anyone
browsing:

> **12 new photos added — some moments may have changed.**  `[Reload]`

**Nothing on your screen moves until you click Reload.** You can keep reviewing
through the whole event and take the update when it suits you.

A chip in the header shows it's running. Click **Stop watching** when the event
is over — it also stops itself after a few quiet hours.

### Starring and exporting

Hover a tile and click **☆** to star it. Click **★ Starred** in the header to
see only the starred frames, where you'll also find:

- **Export all N starred** — sends starred originals to your selected destination
- **☆ Clear my stars** — removes your picks (asks first); colleagues’ stars stay saved

Open **⚙ Settings → Export location** at any time, including before loading a
shoot. Choose a local folder or use browser downloads.
Tile download buttons, the viewer download button, the **D** shortcut, starred
exports, and **☑ Select → Export N selected** all use that destination.
Settings also holds thumbnail size, burst grouping (**Ungroup**), and your
display name; filtering and sorting remain in a fixed toolbar row. On narrow screens, scroll
that row horizontally to reach all filters; controls never wrap into new rows.
Search uses a dedicated second row on small screens, while Upload and Settings
stay beside the shoot selector.

**Local folders:** Chrome or Edge on HTTPS or localhost can remember a folder
across visits. The browser may ask you to authorize it again. Identical files
are skipped; different photos sharing a filename receive a numbered suffix.
An existing remembered export folder is preserved when upgrading.

**Browser downloads:** individual images download normally and bulk selections
use ZIP. The destination follows the browser's download preferences; browser
security prevents this app from assigning an arbitrary local path. This option
works without the local folder API. An unavailable local destination must be
changed explicitly in Settings.

**Every export or download is the full-resolution original file** — the same
bytes off the card. The images on screen are small stand-ins so the page
loads quickly, but that's never what you get when you export or download.

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

**Quality slider** — "top 25%" means the strongest quarter of the shoot by the
system's own ranking, not a measure of sharpness or exposure. It hides weaker
frames *inside* bursts too, so a burst of 61 can show its best 3. The header
always says what survived — `159/733 moments · 332/3316 photos`.

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
Frames with distinct camera identities stay separate. When EXIF has no body
serial, identity falls back to body/lens information; matching camera setups can
still merge.

**Does starring change the ranking?**
No. Stars are just a list you're keeping. They don't teach the system anything —
though a version that does is something we'd like to build, so tell us if you'd
find it useful.

**Can other people see my stars?**
Yes — and you can tell them apart. **Your stars are gold, everyone else's are
blue.** In the starred view, the **All / Mine / Others** buttons narrow it down,
and the Export button follows whichever you've picked.

Starring is per-person: clicking a frame a colleague starred adds your star
alongside theirs rather than removing it, and **Clear my stars** only ever
removes yours. There is no way to bulk-delete someone else's picks.

**A green bar says new photos were added. Do I have to reload?**
No — it waits for you. Until you click it you're looking at a complete, frozen
view. The one exception is **Load more**: once new photos exist, the button asks
you to reload first, because paging further would otherwise mix two different
orderings and skip photos.

**Is the photo I download or export the full-quality one?**
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

Two dependencies are worth knowing about, both installed by `pip install -e .`:
`rawpy` reads RAW files via their embedded preview, and `pyexiv2` supplies the
EXIF that PIL won't — RAW capture times and the Sony lens model. Without
`pyexiv2` every RAW frame lands in its own single-frame burst.

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
pip install -e ".[dev]" && pytest        # smoke + regression tests, no inference needed
node --test tests/test_ui.cjs             # browser logic; Node 18+, no npm install
```

## NVIDIA DGX Spark: Docker installation

Use this installation path on DGX Spark instead of the CUDA 12.1 / native venv
example above. `start.sh` remains the native launcher; **`start-spark.sh` is the
Docker launcher**. Run the commands below from the repository root.

The application has the same features in a native deployment. Native mode sees
host paths directly, so the watcher UI can use a local, LucidLink, or NAS path
without a Docker mount setting, and `start.sh` prompts for the event password.
Docker is recommended on DGX Spark because NVIDIA's ARM64/Blackwell PyTorch and
TorchVision builds are already matched in the base image and remain isolated
from the host. If a compatible native GPU environment is already installed, the
regular setup and launcher are a simpler valid option.

The image starts from NVIDIA's ARM64 PyTorch `25.11-py3` image, also used in
[NVIDIA's Spark playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/docker-compose.yml).
It preserves that image's exact Torch and TorchVision versions during installation
and records the resolved Python environment at `/opt/image-filterer-environment.txt`.
The remaining dependencies are resolved at build time: this is not a complete
lockfile. Keep the successfully tested image for the event; do not rebuild it on
event day. You can pin `SPARK_BASE_IMAGE` to a validated image digest as well.

**Validation status:** this deployment recipe must pass the checks below on your
Spark before live use. Repository tests alone do not validate ARM64 native
libraries, Blackwell inference, or a mounted photo source.

### 1. Prepare the host and configuration

Use the Spark's supported NVIDIA driver and NVIDIA Container Toolkit, with Docker
Engine and the Compose v2 plugin available to your login account. Check:

```bash
nvidia-smi
docker version
docker compose version
cp .env.spark.example .env.spark
chmod 600 .env.spark
id -u
id -g
```

Edit `.env.spark`:

- Set `SPARK_DATA_DIR` to an **absolute directory on the Spark's local SSD**.
- Set `SPARK_UID` and `SPARK_GID` to the numeric IDs printed above. This account
  must be able to write the local data directory.
- To watch an existing folder, set `PHOTO_SOURCE_PATH` to its **absolute host
  path**. It can be a local directory, a LucidLink mount, or a mounted NAS/share.
- Optionally change the host port (`SPARK_PORT`, default 8600) or listening IP.

Create the local directory yourself, owned by that account, for example:

```bash
mkdir -p /home/YOUR_USER/image-filterer-data
```

Use that exact path in `SPARK_DATA_DIR`. Docker deliberately refuses to create
missing bind-mount source directories, helping catch typos. Do not put the
application database or caches on a network mount. This directory preserves the
registry, rankings, stars, browser uploads, model downloads, feature cache, and
HTTPS/session state when the container is replaced. `.env.spark` is ignored by
Git and excluded from the image build context; it contains machine paths, not the
event password.

`PHOTO_SOURCE_PATH` is optional. Without it, `/photos` is an empty placeholder
and **Upload from device** still works: the browser sends those photos into
`SPARK_DATA_DIR`. With it, Docker exposes the chosen directory read-only at
`/photos`, because the watcher UI can select only paths already visible inside
the container. For a large library, mount its stable root and choose an
event-specific subfolder in the UI. The configured UID must be able to read it.

Mount a network source before starting the container. LucidLink/FUSE permissions
must permit access from the configured container UID; a successful host `ls`
alone does not prove container access. Follow your storage administrator's
supported mount/access configuration if permission is denied. If a source is
unmounted or remounted, recreate the container afterward with
`bash start-spark.sh up --force-recreate`. A container restart policy does not
wait for a network mount after a host reboot.

### 2. Build, download models, and test inference

```bash
bash start-spark.sh build
bash start-spark.sh setup
```

Setup downloads the third-party weights and runs the full image encoder, face
encoder/detector, ranker, scene detector, and text encoder on a synthetic image.
It also checks CUDA matrix operations and TorchVision's CUDA NMS extension.
Downloads persist under `SPARK_DATA_DIR`, including the Hugging Face cache.
Internet access is needed for the build and initial downloads.

If using a mounted photo source, test actual files from it using container paths,
not host paths:

```bash
bash start-spark.sh check --sample '/photos/example-person.jpg'
bash start-spark.sh check --sample '/photos/example-camera.CR3'
```

Use a clear face photo and a RAW file from each camera format you intend to use.
These checks force fresh feature extraction and test EXIF reading. A synthetic
probe cannot establish that real faces or your camera's RAW files work. Check the
reported `face_detected` result on the face photo. If setup/check fails, resolve
that error before starting the live event; disabling face extraction changes the
feature inputs expected by the shipped ranking model.

The build requires network access to NVIDIA NGC and Python package repositories.
It never installs a replacement host NVIDIA driver. A framework dependency
conflict should fail the build rather than silently replace NVIDIA's Torch pair.

### 3. Start and use the server

```bash
bash start-spark.sh up
bash start-spark.sh status
bash start-spark.sh logs
```

`up` asks for the shared event password twice and passes it to the container
without saving it to disk. For unattended startup, provide it in the process
environment instead: `IMAGE_FILTERER_PASSWORD='...' bash start-spark.sh up`.
Docker retains the value in the created container configuration, so its
`unless-stopped` restart policy still works after a reboot. Recreating the
container asks for the password again.

Open **`https://SPARK_LAN_IP:8600`** from viewers' computers (or your configured
host port). HTTPS uses the app's self-signed certificate; browsers will need to
accept/trust it. For a managed deployment, use a trusted certificate/reverse
proxy. Allow the selected TCP port on the host/network firewall as needed; the
container does not modify the host firewall. Configure Docker-published-port
access at the host/network level rather than assuming a UFW rule alone restricts
Docker traffic.

There are two ways to add photos:

- To use files already accessible to the Spark, configure `PHOTO_SOURCE_PATH`.
  In **Watch a folder**, enter **`/photos`** or a subfolder such as
  `/photos/day-one`. This works for a local folder, LucidLink, or another
  mounted network share.
- To send files from another computer through the web app, leave
  `PHOTO_SOURCE_PATH` unset and choose **Upload from device**. These copies are
  stored under `SPARK_DATA_DIR`; the watcher and `/photos` are not involved.

This Compose deployment runs **one application container**. Do not scale it:
watchers and ingestion locks are process-local. The container runs as your
configured UID rather than root. Local storage persists across rebuilds and
`down`; originals exposed through `PHOTO_SOURCE_PATH` are mounted read-only.

```bash
bash start-spark.sh down                  # stop; keep local data and models
bash start-spark.sh up                    # start again
bash start-spark.sh up --force-recreate   # apply environment or mount changes
```

The health check tests HTTPS responsiveness, not GPU throughput or mount health.
Startup checks GPU kernels and required asset files, but does not repeat the full
model check. Restarting the container **does not restore an active hot-folder
watch**; starting another watch currently creates a new shoot. The restart policy
restarts an exited container, not an unhealthy-but-running one.

To preserve installation details after a successful rehearsal:

```bash
docker compose --env-file .env.spark -f compose.spark.yaml run --rm --no-deps \
  --entrypoint cat image-filterer /opt/image-filterer-environment.txt > spark-environment.txt
```

Before upgrading, stop the service and back up `SPARK_DATA_DIR` (it includes the
registry, stars, uploads, model caches, and HTTPS/session state). Commit and push
all intended application files before pulling on the Spark; local untracked
files will not transfer through Git.

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
| `IMAGE_FILTERER_PASSWORD` | unset (no auth) | Shared password gating the whole server — see [Authentication](#authentication) |

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

### Authentication

There's no authentication by default — anyone with the link can browse and
star. To require a password:

```bash
export IMAGE_FILTERER_PASSWORD=whatever-you-want
image-filterer-server --host 0.0.0.0 --port 8600
```

Every route is gated except the login page itself; unauthenticated requests to
`/` get a login form and everything else gets a `401`. A session lasts 18
hours from login and survives server restarts with the same password — the signing
key is written once to `$IMAGE_FILTERER_DATA_ROOT/.session_secret`. Changing the
password invalidates existing sessions; background polling does not extend them.

This is one shared password for everyone, not per-user accounts — it's meant
to keep the link from being useful to someone who doesn't have it, not to
tell people apart. That's still done by the existing per-browser id: the
login form and the **Set your name** button in the header both write to it,
and once set, a display name shows up wherever a frame lists who starred it.
There's no server-side verification that a name is who it claims — anyone can
type any name, same trust model as the shared password.

### Serving over HTTPS (for Export)

```bash
image-filterer-server --host 0.0.0.0 --port 8600 --https
```

Local-folder export (see [Starring and exporting](#starring-and-exporting)) uses the
browser's File System Access API to write straight into a folder on the
viewer's machine. That API only works in a "secure context" — HTTPS, or
`http://localhost` — which a plain LAN address never satisfies, even in
Chrome. `--https` makes the server generate (once) a self-signed certificate
and key under `$IMAGE_FILTERER_DATA_ROOT/ssl/`, reused on every future start,
and serve over TLS with it. `start.sh` passes this by default.

Self-signed means each browser shows a "your connection isn't private"
interstitial the first time — that's expected, not a sign anything is broken.
Click **Advanced → Proceed to `<host>` (unsafe)**; the browser remembers that
choice afterward. Requires the `openssl` CLI tool (already present on most
Linux/macOS installs); without it, `--https` exits with a clear error instead
of silently falling back to plain HTTP.

This is not the same guarantee a real certificate gives you — anyone on the
network could in principle present their own self-signed cert for the same
address and a careless "Proceed anyway" click would trust it. For a shoot on
a network you don't fully control, a real certificate (e.g. via a reverse
proxy and Let's Encrypt, or your organization's internal CA) is the more
correct fix; self-signed is the pragmatic one for a LAN tool used for a few
hours at an event.

### Watching a folder (hot folder)

```
POST /api/hotfolder/start   {path, name, scan_interval?, cooldown?}
POST /api/hotfolder/stop
GET  /api/hotfolder         status
```

Starting a watch creates a run bound to that folder and scans its subfolders too.
Each batch re-ingests **all stable photos** — files still copying wait for a later
batch. The content cache avoids re-extracting known frames, and this
keeps bursts, dedup and ranking correct across the shoot rather than stapling new
photos onto the end.

**It polls; it does not use inotify.** Filesystem events only fire for writes
through the local kernel, so a photographer writing from another machine to a
network or cloud mount generates nothing to listen for. Polling also survives
inotify queue overflow, which would otherwise drop a photo permanently.

A 5 s scan tracks which files have stopped growing (size + mtime unchanged for
two ticks, then a decode check); ingestion runs only when something is ready, the
ingest lock is free, and a cooldown has passed — so 300 files landing at once
become one pass. The scan eases to 30 s after ten quiet minutes and snaps back on
first sight of a new file. One watcher at a time; it stops itself after ~6 hours
idle so a forgotten watch can't poll a cloud mount forever.

Unreadable files are remembered until their size or modification time changes,
so a repaired copy gets another chance. Failed batches remain queued and retry
after the cooldown. Cancelling a watched ingest stops the watch as well.

### Behavior under load

- **Ingestion is single-flight.** A second concurrent ingest gets a `409` with a
  clear message. Everyone polling the UI sees a sticky bar with the active run's
  progress and a **Cancel** button.
- **Cancel is cooperative** — the flag is checked between feature chunks and at
  phase boundaries, so it stops within a chunk or two. (A background GPU thread
  can't be safely hard-killed.)
- **Run selection is per-viewer.** Runs are cached by id, so two people can
  browse different shoots at once without disturbing each other.
- **Batches publish atomically.** A complete generation is written before an
  atomic `current` symlink switch publishes it. A failed write leaves the last
  committed generation readable, including after a restart. The previous
  generation is retained for readers already using it. Browsing, search and
  burst details reject outdated versions and offer Reload.
- **Content hashes are memoised** on `(path, size, mtime_ns)` in
  `$CACHE_DIR/sha1_memo.db`. Re-checking a known 3,300-frame folder drops from
  ~17 s of reads to ~0.02 s — without this, polling a folder is unusable.
- **Reads scale.** Browsing and search are read-only and run concurrently;
  verified with 8 simultaneous searches. The text encoder is built once and
  shared.
- **Tuned for remote viewing** — grid thumbnails are ~6–8 KB at the default zoom
  and opening a photo streams a ~1600px preview (~150–300 KB) rather than the
  8 MB original. Images use private browser caching; generation changes refresh
  image URLs. The thumbnail LRU is bounded by bytes and entries, and replaced
  originals invalidate its entries.
- **Downloads stream.** Bulk downloads are zipped (STORED, not deflated — JPEGs
  don't compress) and streamed, so a multi-GB archive costs almost no memory.

For heavier load, use a reverse proxy with one threaded gunicorn worker — the app
factory is `image_filterer.server:create_app`. Run caches, star locks, the watcher,
and the GPU lock are process-local, so multiple worker processes are unsupported. **Authentication is a single shared
password**, off by default — see [Authentication](#authentication). It isn't a
substitute for a real access-controlled network; add auth at the proxy too if
you expose this beyond a trusted LAN.

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
| **Quality** | top N% | Percentile of `score_s12_after_hard` over the run. A percentile rather than a raw score because the ranker's units are arbitrary and differ per run. Applies to bursts, frames within a burst and search hits — but **not** to `/api/starred`, where a human pick outranks the model's opinion. |
| **Sort** | rank / time / stars | Rank uses the burst representative's score. Time and stars both change *which frame is shown* on the tile, not just burst order: time shows each burst's earliest frame; stars shows its most-starred frame (not a sum — one frame with 3 stars outranks a burst where five frames each got 1). A burst with no stars keeps its normal representative under the stars sort. Whatever's shown, opening the burst still lets you browse and star every frame in it. |

All of them AND together, and all combine with search. Filtering happens before
choosing a burst’s displayed frame, so changing sort order cannot show a photo
that fails the active filters.

## Run outputs

Each ingested folder becomes a run under `$IMAGE_FILTERER_DATA_ROOT/runs/run0001_<name>/`:

| File | Contents |
|---|---|
| `ranked.csv` | one row per unique frame: scores, burst columns, `shot_type`, `subject_class`, `is_hero`, `captured_at`, technical metrics |
| `bursts.csv` | one row per burst (its representative), ranked |
| `search_index.npy` | full-frame embeddings, row-aligned to `ranked.csv` |
| `stars.json` | starred frame paths (created on first star) |
| `captured_at.json` | EXIF timestamp cache, backfilled for runs predating the `captured_at` column |
| `version.json` | committed generation counter, advanced by two for each successful publication |
| `current/` | symlink to the complete published generation under `.generations/` |
| `validated.json` | size/mtime signatures of successfully decoded images |
| `config.json` | the exact config used |
| `uploads/` | the images, for browser uploads only (absent for "folder on server" runs) |

`ranked.csv` is the export path — hand it to anyone who wants the ordering
without the UI. Top-level CSV, search-index, config and version paths are aliases
into `current/`; old runs with regular files still load normally. The current and
previous generations are retained. Copy the resolved files when exporting run
metadata to another machine.

## HTTP API

| Route | Purpose |
|---|---|
| `GET /` | the UI |
| `GET /api/state` · `/api/runs` | current run + all runs |
| `POST /api/upload/start` · `/chunk` · `/finish` | batched browser upload → creates a run |
| `GET /api/ingest/status?run_id=` | progress for one ingest |
| `GET /api/active` · `POST /api/cancel_ingest` | the running ingest; cooperative cancel |
| `POST /api/select_run` | `{run_id}` — load a ready run |
| `GET /api/bursts?shot=&subject=&hero=&sort=&unfurl=&offset=&limit=` | ranked bursts (or, with `unfurl=1`, ranked individual frames) |
| `GET /api/burst/<id>` | frames within a burst |
| `GET /api/search?q=&shot=&subject=&hero=&starred=&unfurl=` | semantic search |
| `GET /api/stars?viewer=` · `POST /api/star` · `POST /api/stars/clear` | read / toggle / clear stars (per-viewer) |
| `GET /api/run/preview_delete?run_id=` · `POST /api/delete_run` | what a delete destroys, then do it |
| `GET /api/starred?shot=&subject=&hero=&sort=` | starred frames, one entry per frame |
| `GET /download?run_id=&path=` | one original, as an attachment |
| `POST /api/download/prepare` | `{scope:"starred"}` or `{paths:[…]}` → `{token, count, bytes}` |
| `GET /api/download/zip?token=` | streams the prepared archive |
| `GET /api/version?run_id=` | cheap poll: generation + counts, drives the reload prompt |
| `GET /api/hotfolder` · `POST /api/hotfolder/start` · `/stop` | watch a folder |
| `GET /img` · `GET /thumb?w=` | image bytes (only exact files indexed in the requested run) |

`shot` is comma-separated multi-select (`wide,medium,close`); `subject` is a
single value (`people`\|`stage`); `hero=1` restricts to hero shots; `sort=time`
orders chronologically and `sort=stars` by each burst's most-starred frame,
instead of by rank; `top_pct=N` keeps only the strongest
N% of frames (clamped to 1–100); `unfurl=1` turns off burst grouping on both
`/api/bursts` and `/api/search` — each returns one row per individual frame
instead of one per burst, with every other filter/sort applied at the frame
level directly (a burst's frames all share the same quality percentile pool,
so this changes *what's grouped*, not the underlying scores). It also changes
what `starred=1` means for search: normally "this burst holds a star
somewhere" (since search collapses to one hit per burst, which may not be the
starred frame); unfurled, "this exact frame is starred."

## Robustness

- **Bad files never kill a run.** Every file is checked for decodability first;
  corrupt, truncated, or unsupported files are skipped and counted in the
  finish message. A run only fails if *nothing* decoded.
- **Upload corruption is auto-repaired.** Some browser/proxy combinations prepend
  a stray `\r\n` to uploaded bodies (a leaked multipart separator) and break
  decoding. The server strips junk before the real image header, restoring the
  bytes *and* the sha1 — so the feature cache still hits.
- **Downloads and image serving allow only indexed files** in the requested run,
  not neighboring files. Deleted runs and their prepared download tokens are
  no longer available.
- **RAW is read via its embedded preview**, not by demosaicing: ~10 ms per file
  instead of 1-3 s, and on current bodies the preview is full resolution (an R5
  Mark II CR3 yields 8192×5464). Needs `rawpy`; without it RAW is reported
  undecodable and skipped, as before. `X.CR3` alongside `X.JPG` drops the RAW —
  a filename rule, because content dedup would only *probably* catch the pair.
  `/img` serves the preview so browsers can display it; `/download` always
  returns the original file.

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
  hotfolder.py    polling watcher that folds new arrivals into a live run
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
- **Camera identity uses EXIF.** Bodies with distinct serial numbers stay separate;
  identical body/lens combinations without serial numbers can still merge.
- **Star ownership is per-browser, not per-account.** "You" is a random id in
  `localStorage`, optionally labeled with a display name; clearing site data
  makes your stars look like someone else's. Stars from before ownership
  tracking show as unattributed.
- **A registry row whose folder was deleted by hand** still shows in the
  dropdown. Deleting through the UI removes both.
- **Hot folder is poll-based**, so photos appear in a minute or two rather than
  instantly, and only one folder can be watched at a time.
- **Authentication is one shared password, off by default.** It keeps the link
  from being useful to a stranger; it does not distinguish or verify who's who
  (display names are self-reported) and there is no per-action authorization.

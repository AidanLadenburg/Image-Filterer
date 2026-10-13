# Architecture and tuning notes

Why the pipeline is shaped this way, and what the defaults in `config.py` are
worth. Read this before changing a threshold — most of them were settled by an
experiment, and several obvious-looking "improvements" were tried and rejected.

The labeled data is one keynote (GTC San Jose, March 2026): 2,628 unique photos,
424 `Good`, 13 must-find (`Top-5to10`), plus 725 `Questionable` (ignored).
Training uses 2,360 of them (295 keepers, 2,065 rejects, 9 must-finds) after
leaving out 268 photos from one-label photographer-days (see "the confound").
Visual checks use two unlabeled shoots: GTC Taipei (703 photos, 3 cameras) and
a multi-camera test with candids (256).

---

## The pipeline

```
raw images
    │
    ├─ Features (cached by content hash)
    │    • SigLIP2-SO400M, whole frame (224 px)          1152-d
    │    • FaRL-B/16 on the largest face (YuNet)          512-d
    │    • face measurements (eye/mouth openness, sharpness …)  7
    │
    ├─ Scene (cached by content hash; scene.py)
    │    • YOLO11x-pose @1280 → people + 17 body keypoints
    │    • the speaker (see "Choosing the speaker")
    │    • the speaker's face, cropped from a ~3000 px decode →
    │      MediaPipe: 52 expression blendshapes + head yaw/pitch/roll
    │    • how lit the background is around the speaker
    │
    ├─ Ranker v2 (ranker.RankerV2)
    │    each embedding → 64 dims (standardise + PCA), + face measurements
    │    + expression → pairwise RankNet, half the pairs inside a burst,
    │    3 seeds averaged                                → score_s1
    │
    ├─ Moments ("bursts") — camera + time + similarity, see below
    │
    └─ Tags — shot scale from keypoints; subject / hero from the speaker
```

The scene stage costs ~26 ms GPU (pose, fp16) + ~7 ms CPU (expression) per new
photo on a GB10, on top of the encoders. Decoding and face detection run on a
pool of CPU threads (`IMAGE_FILTERER_CPU_WORKERS`); pose + expression run on
their own GPU thread alongside, so ingest is bounded by the slower of the two,
not their sum. Measured on a GB10: ~10 s per 100 new photos.

Everything is cached by content hash, so re-ingesting a folder is nearly free.
The disk cache has a process-wide in-memory LRU in front of it
(`cache_io.MEMORY`, capped by `IMAGE_FILTERER_MEM_CACHE_MB`, default 1 GB ≈
100k photos), and EXIF capture times are memoized by path/size/mtime, so the
part of an ingest that grows with the run (a live run re-ingests all of its
photos on every arrival) is ~1 s at 5,000 photos. Pixel-identical dedup finds
candidate pairs with one GPU matmul pass.

---

## How ranking quality is measured

The original holdout split individual photos, so near-identical burst-mates
landed on both sides and the reported AUC (0.96) mostly measured memory. The
tests now (in `train.py`, written to `model/training_summary.json`):

- **Leave one photographer out.** Train on everyone else, rank only that
  photographer's photos. Camera, lens and position are constant inside the
  test set, so "who shot it" can't help, and the photographer is unseen — the
  closest thing to a new event.
- **Timeline slices.** Five contiguous parts of the event; train on four.

Metrics are per **moment** (a burst counts as a keeper if any frame is Good,
scored by its best frame), because that's what the grid and auto-export show:
average precision, the share of keepers in the top 10% of moments, and how
often the best frame of a mixed burst is a Good one. No burst is ever split
between training and testing. Only ~80 bursts mix Good and Bad, so the
best-frame number moves ±5 points by chance; the 13 must-finds are too few to
separate models.

The v1 → v2 comparison that decided the switch:

| | moment AP (photographer out / timeline) | best frame picked (photographer out / timeline) |
|---|---|---|
| v1 (shipped before) | 0.655 / 0.633 | 70% / 71% |
| v2 | 0.694 / 0.684 | 77% / 83% |

The shipped `ranker.pt`, trained without the one-label photographer-days (see
below), scores — from `model/training_summary.json`, photographer out /
timeline — moment AP 0.76 / 0.67, keepers in the top 10% of moments 90% / 83%,
best frame picked 75% / 80%. A new model is compared against these, rerun on
the same training set.

---

## What the labels actually say — and the confound

`Good` means "picked from that photographer's own set". How selective that was
varies by photographer: four of them handed over **only** their picks (129
photos, 100% Good — 5% of the photos but 30% of the keepers), the rest kept
3-20%. Those four were mostly at wide positions: 67% of their keepers have a
small or no speaker, against 6% for everyone else.

A model trained on all of it learns *who shot it* — their lens look, position
and framing — as "good". On a new shoot (GTC Taipei) that put empty dark
stages and slides at the top: frames with no face scored +1.4 above frames with
one, and per-camera medians differed by 1.5. On the labeled keynote itself the
held-out metrics can't see this (both versions score 0.69), because the
confound is present in test and train alike. The decision was made on the new
shoot, by eye.

So `train.py` leaves out **photographer-days with only one kind of label**.
That also drops a January day where everything is `Bad`.

The old no-face defaults made it worse: eye openness defaulted to 1.0 ("far
wider than any real eye") when no face was found. They're now typical values
(`features.NO_FACE_EAR / NO_FACE_MAR`); `face_detected` already says the face
is missing.

---

## What each signal is worth (held-out, this data)

| Signal | Effect | Cost | |
|---|---|---|---|
| SigLIP2 whole frame + FaRL face + face measurements | the core | — | kept |
| Compressing embeddings to 64 dims each | stops overfitting; best frame 70% → 75-80% | free | kept |
| **Speaker expression** (blendshapes + head pose) | **AP +0.04, best frame +8-13 points** | ~7 ms CPU | kept |
| Body keypoints | no ranking gain alone; needed for the face crop, shot type, hero | ~26 ms GPU (fp16) | kept |
| SigLIP2 @384 on whole frame / speaker / face | no consistent gain | ~20 ms each | rejected |
| 10 pretrained quality / aesthetic models (pyiqa: LAION-aes, TOPIQ, MUSIQ, NIMA, CLIP-IQA+, LIQE, …) | near random for editor picks (AP 0.30-0.45 vs 0.33 random) | ~30 ms each | rejected |
| Local VLM "photo editor" judge (Qwen2.5-VL-7B) | 0.46 alone; *lowered* AP combined with the ranker, also as a top-20% re-ranker | ~800 ms | rejected |

| Model / training | Result | |
|---|---|---|
| Pairwise RankNet, half the pairs within a burst, 3 seeds | best | kept |
| Logistic regression | AP 0.60 | rejected |
| Gradient-boosted trees on measurements only | AP 0.56-0.60 | rejected |
| Pairs only within the same photographer | AP 0.56-0.65 | rejected |
| Separate within-burst re-ranker | small gain, needs a 384 px crop model | rejected |
| Per-camera feature centering | removes real signal too (AP 0.85 → 0.70) | rejected |

The limit is the labels (one event, few mixed bursts), not the model: every
approach plateaus at AP 0.65-0.70. More labeled events — with real rejects —
would move it more than any model change tested.

### Within-burst pair sampling (still the single biggest training choice)

A ranker trained on uniformly-sampled pairs learns scene-level cues and fails
at picking the best of eight near-identical frames. On the v1 features, sweeping
the share of pairs drawn inside a burst moved best-Good-ranked-#1 from 59%
(0.0) to 84% (0.5) with AUC flat; 1.0 collapses (too few mixed bursts). 0.5 is
kept.

---

## Moments (bursts)

Camera identity is EXIF only (Canon `BodySerialNumber`; Sony `Make/Model/Lens`,
since the ILCE-1 exposes no serial; else `Make/Model`). RAW timestamps come from
`pyexiv2` — without it every RAW frame is its own burst.

Within one camera, in time order, a frame joins the current moment when:

```
dt <= 0.6 s                                               → always
dt <= 30 s, moment <= 45 s long,
   cosine >= 0.91 to any of the moment's last 3 frames,
   cosine >= 0.89 to its first frame                      → same moment
```

The 8 s window used before split ~1 in 4 consecutive moments that were
near-identical (same pose, same slide, a pause in shooting) into separate grid
tiles. 45 s windows without the guards chained minutes of the same slide into
one 77-118-frame "moment"; the length cap and the first-frame test stop that.
Comparing with the last 3 frames keeps one odd frame (a turn, a flash) from
splitting a moment. GTC Taipei: 203 → 177 moments; candid test: 65 → 47.
Reviewed merges were the same moment.

**Dedup**: sha1, plus embedding cosine ≥ 0.9995 with differing sha1 (re-encodes).

---

## Choosing the speaker

Everything below depends on picking the right person. The largest box was
wrong surprisingly often: figures on a projected slide, the backs of audience
heads in the foreground. Now each confident person is scored by height × how
clearly their face and shoulders are visible; people cut by the left/right edge
count for less, and low, barely visible figures are skipped (`scene.subject`).

---

## Shot scale

From where the frame cuts the speaker's body (`shots.shot_from_pose`):

- **wide** — the whole body is in frame (feet visible), the person is small,
  or there's no prominent person
- **medium** — cut above the feet
- **close** — cut less than 2.2 head-units below the shoulders, *and* the head
  (eyes → shoulders) is ≥ 0.20 of the frame height, *and* the person fills ≥ 60%
  of it — so an upward-angled shot of a distant speaker, merely cropped at the
  waist, isn't a close-up

Head-units (eyes → shoulders, vertical) don't shrink when the speaker turns
sideways, unlike shoulder width. At a keynote shot on long lenses most frames
are full body, so most are wide.

The previous SigLIP zero-shot axis ("a close-up of a face" vs "a wide shot of a
stage") was fooled by darkness and content — near-black frames came out
"close", full-length speakers "medium". It remains only as a fallback for runs
without pose data.

---

## Subject and hero

- **Subject**: "people" if the speaker's box covers ≥ 4% of the frame, else "stage".
- **Hero**: the speaker alone (nobody else at least half their height), filling
  ≥ 45% of the frame height, with ≤ 6% of the area **around** them lit
  (1.5 body widths either side, head to feet, every person excluded).

Measuring brightness only around the speaker keeps the stage lighting rig along
the top edge from disqualifying a clean shot. The raw measurements are cached,
so retuning a threshold reprocesses nothing.

---

## Face detection: YuNet, not MediaPipe alone

MediaPipe's FaceLandmarker is built for selfie-distance faces; on stage
photography it found a face in ~2% of frames. YuNet finds the face box, the box
is cropped, and MediaPipe runs on the crop (~97% detection; AUC 0.887 → 0.959
at the time). The expression crop in scene.py is located from body keypoints
instead, so it is the speaker's face rather than the largest one.

---

## The hard-reject is off, on purpose

Stage 2 originally slammed frames to −∞ for closed eyes, blur or clipped
exposure. 29 of 126 hard-rejected frames were keepers — a dark stage reads as
"clipped". The metrics are still computed and stored; they don't move scores.

---

## Semantic search

The SigLIP2 text tower shares a joint space with the cached whole-frame
embeddings, so text→image search is a dot product against `search_index.npy`
(~15-50 ms). Text on screen (OCR) is matched separately and ranked first; see
`ocr.py` / `textmatch.py`.

---

## Text on screen (OCR)

`ocr.py` reads every frame — not one per burst, since burst-mates often catch
a slide mid-transition — with PP-OCR through RapidOCR: English mobile models,
v3 detector + v4 recognizer, no angle classifier, at 2048 px on the long side.
The engine is TensorRT when `tensorrt` and `cuda-python` are installed,
otherwise RapidOCR's PyTorch engine on CUDA (`ocr.pick_engine`); with no
`rapidocr` or no CUDA GPU it stays off rather than take seconds per photo on
the CPU. TensorRT engines are built TF32 (fp16 was slower and broke
recognition), ~40 s once per machine, and cached with the models in
`$IMAGE_FILTERER_DATA_ROOT/models/ocr`.

It runs as `IMAGE_FILTERER_OCR_WORKERS` (default 4) separate processes, each
with its own CUDA context and ~0.9 GB of GPU memory, so they overlap on the
GPU; each decodes the next photo on a second thread while the GPU reads the
current one. On a GB10 the 703-photo GTC Taipei shoot takes ~21 s with 4
workers (3: 26 s, 5: 19 s). Workers only get new photos while no ingest is
running and run at low CPU priority, so OCR never delays photos appearing; a
run's fresh photos go to the front of the queue. Results are cached by file
content (sha1, like the feature cache), so re-ingests, restarts and a photo in
two runs never re-read it. Text isn't part of a run's published generation: it
fills in as it's read.

Search ranks frames whose text matches the query first (best text match, then
best visual match), then everything else in visual order. Matching
(`textmatch.py`) is forgiving of OCR's habits: spaces are ignored, words of 6+
characters allow one wrong letter (10+: two), short words must match exactly,
and every query word must be found. `GET /api/ocr/lines` returns each line with
its box so the viewer can highlight the matching text.

---

## Auto-export

`auto_export.py` copies the best frames of a live shoot into review folders
under `<export folder>/<shoot>/Auto/` (Best, Hero, Close, Wide, …, one per
shot-list entry, and Starred). Each was settled by replaying GTC Taipei in
arrival order:

- **Strictness is a share of each folder's own photos** (default top 10%),
  ranked over the whole run so far — a percentile, never a score cutoff, since
  raw scores shift between sessions and cameras and hero frames score far
  below the shoot overall. Ranking moments instead gave ~1 photo per moment.
- **Sessions**: a capture gap of 45+ minutes starts a new session, and each
  session gets its own top X%, so an evening event can't crowd out the next
  morning's keynote.
- **Look-alike suppression is the variety control.** Burst-mates compete like
  any other photo; a pick too similar to one already sent to that folder
  (within or across bursts) is skipped, unless it's clearly better. In the top
  20% of bursts, ~30% were near-duplicates of a better pick.
- **Early on, confidence instead of a warm-up**: a photo ranked r-th of n is
  sent once P(Binomial(n, p) ≥ r) reaches the Speed setting's confidence.
- **A burst is sent once it's finished** (the camera moved on, or 120 s
  passed), and **sends are append-only**, recorded in
  `auto_export_state.json` so a restart never re-exports.
- **Shot-list "looks like" entries must beat generic keynote descriptions**,
  so an empty category stays empty; "text on screen" entries use OCR and wait
  until a frame's text is read.

The export folder must already exist; it is **never created**. If it
disappears (Lucid unmounted), exports pause with a health alert and catch up
without duplicates when it's back — nothing is written to local disk instead.
Copies appear atomically (hidden `.partial`, then rename), and the export root
carries a marker so a hot folder watching the same share never ingests picks
back.

---

## Hot folder, network mirror, resume

`hotfolder.py` polls rather than using inotify: a photographer writing to a
network or cloud mount from another machine generates no local filesystem
events, and inotify queues overflow. Scanning is cheap and frequent (5 s local,
2 s network; 30 s after ten quiet minutes); ingesting is expensive and runs
only when something is ready, the ingest lock is free and a cooldown (20 s
local, 5 s network) has passed. Each batch re-ingests the whole folder — the
cache makes known frames nearly free, and bursts, dedup and ranking stay
correct across the shoot.

A new file is decode-checked on first sight: on LucidLink a file shows its
final size while its bytes are still uploading, so only a successful decode
proves it complete. Failures retry with backoff (up to once a minute) for 30
minutes before the file counts as unreadable. RAW files wait one scan, since a
half-copied CR3 can decode (its preview comes first).

On a network filesystem (`mirror.py`: SMB, NFS, FUSE mounts, …) every open is
a round trip with no client-side caching, and ingest touches each photo about
five times — on the GTC test the app read 10.6 GB to process 1.4 GB. So each
photo is copied once into `<run>/mirror/` (atomically, keeping its mtime) as
part of its decode check, and everything downstream reads the local copy. If
the share is mounted several times, reads are spread across the mounts. A
one-shot host-folder run from a network folder is mirrored the same way.

The active watch is recorded in `$IMAGE_FILTERER_DATA_ROOT/active_watch.json`
and resumed by `server.main()` after any restart, into the same run, with the
run's existing photos passed as already known — so the run never republishes
with only part of its photos while they're rechecked. Stopping on purpose
(Stop watching, a new watch, ~6 h idle) removes the file.

---

## Supervision and health

`start.sh` runs the server in a loop: if it exits without Ctrl+C it is
restarted after 3 s (30 s after repeated crashes within two minutes of
starting), and from 90 s after each start the page is probed every 20 s —
three missed answers in a row and it is killed and restarted (~2 min worst
case). The password stays in the launcher's memory, so restarts need no one
at the keyboard and sessions survive (same password, same signing key). Every
restart is logged to `launcher.log` and raised as a desktop notification and
webhook post.

Inside the server, `health.py` checks every 10 s for: the watched folder
unreadable, the watch thread dead, the export folder missing, auto-export or
OCR errors, under 8 GB of free memory, and under 20 GB free on the data disk.
A problem must persist for 30 s (so a one-scan blip on a cloud mount stays
quiet); it is then shown to every viewer as a banner (`GET /api/health`) and
sent to the server log, a desktop notification (when there is a desktop
session) and `IMAGE_FILTERER_ALERT_WEBHOOK` — and again when it clears.

---

## Worth trying next

1. **More labeled events.** The single biggest lever. A review mode that stores
   training labels separately from stars (and asks "which frame is best?"
   inside a burst — exactly what the expression model learns from) would
   produce them; `train.py` already takes several events
   (`--train-root A --train-root B`) and reports the held-out tests, so a new
   model ships only if it beats the current one.
2. **Leave-one-event-out testing** once there are two or more events.
3. **Burst score from the top 2-3 frames** instead of the single best.

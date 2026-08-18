# Architecture and tuning notes

Why the pipeline is shaped this way, and what the defaults in `config.py` are
worth. Read this before changing a threshold — most of them were settled by an
experiment, and several obvious-looking "improvements" were tried and rejected.

Numbers below all come from a single labeled keynote shoot: 3,316 unique frames
after dedup, of which 392 `Good`, 2,204 `Bad`, 13 must-find, 725 held aside
unlabeled.

---

## The pipeline

```
raw images
    │
    ├─ Stage 0 — Feature extraction (cached by content hash)
    │    • SigLIP2-SO400M on the full frame        1152-d
    │    • FaRL-Base/16 on the face crop            512-d
    │    • face-quality scalars                       7-d
    │                                          → 1671-d input vector
    │
    ├─ Stage 1 — Pairwise RankNet (MLP, 256 hidden)
    │    Trained Good-vs-Bad, 4096 pairs/epoch.
    │    Half of every epoch's pairs are drawn WITHIN a burst.
    │                                          → score_s1
    │
    ├─ Stage 2 — Technical metrics (sharpness, exposure, eye-openness)
    │    Computed and stored per frame. The hard-reject gate is OFF and the
    │    soft penalty weight is 0, so this does not currently move scores.
    │                                          → score_s12_after_hard
    │
    ├─ Burst layer — cluster by EXIF camera + time + embedding similarity,
    │    pick one representative per burst, rank bursts by that frame
    │
    └─ Tagging — shot scale (SigLIP2 zero-shot) + subject/hero (YOLOv8-pose)
```

Everything expensive is cached by `(sha1, kind, encoder_id)`, so re-ingesting a
folder — or ingesting one that overlaps a previous shoot — costs almost nothing.

---

## Why a pairwise ranker

The task is "which of these two frames would the photographer publish," not "is
this frame good." A pairwise RankNet loss matches that directly and sidesteps the
class-imbalance problem (2,204 negatives vs 392 positives). The model only ever
sees pairs; the absolute score is meaningless except as an ordering.

### Within-burst pair sampling — the single biggest win

A ranker trained on uniformly-sampled pairs learns *scene-level* cues, because
most random Good/Bad pairs come from different moments and are trivially
separable. It then fails at the job that actually matters: picking the best frame
out of eight near-identical ones.

Sweeping the fraction of pairs sampled within a single burst:

| `within_burst_pair_fraction` | Holdout AUC | Best-Good ranked #1 in its burst | Bad ranked #1 |
|---|---|---|---|
| 0.0 | 0.887 | 59% | 26% |
| 0.3 | 0.886 | 76% | 8% |
| **0.5** ← default | **0.887** | **84%** | **4%** |
| 0.7 | 0.874 | 83% | 1% |
| 1.0 | 0.660 | 85% | 1% |

0.5 buys 25 points of within-burst accuracy for no loss of global AUC. 1.0
collapses because only ~75 of 733 bursts are mixed Good+Bad — push everything
in-burst and the cross-burst supervision that global ranking depends on vanishes.

---

## Burst clustering

Camera identity comes from EXIF only — no filename heuristics, which break the
moment a second body or a card reader renames anything:

- Canon → `BodySerialNumber`
- Sony → `Make/Model/LensModel` (the ILCE-1 exposes no body serial, so the lens
  is the next-best per-rig discriminator; this is what `pyexiv2` is for)
- otherwise → `Make/Model`

RAW files are read through `pyexiv2` rather than PIL, which can't open a CR3 at
all — and the embedded JPEG preview we decode them through carries no
`DateTimeOriginal`. Without `pyexiv2` a RAW frame gets no timestamp and becomes
its own single-frame burst, which is why it is a required dependency rather than
an optional one.

Within one camera, consecutive frames merge when:

```
dt <= 0.6s                                  → always merge
OR (dt <= 8.0s AND cosine(emb_full) >= 0.92) → merge if near-identical
```

The second rule catches the photographer who holds a composition, takes three
frames over five seconds, and shifts slightly. Time-only clustering at 2.5 s gave
889 bursts; adding the similarity rule gives **733**, with 139 fewer singletons.

**Dedup** runs at two levels: sha1 (byte-identical copies across folders) and
embedding cosine ≥ 0.9995 with differing sha1 (JPEG re-encodes, ICC changes,
Finder's " 2.jpg"). That threshold is deliberately extreme — anything lower
starts merging genuine burst-mates.

---

## Face detection: why YuNet, not MediaPipe

MediaPipe's FaceLandmarker is built for selfie-distance frontal faces. On stage
photography it detected a face in **~2% of frames** — meaning the FaRL embedding,
the eye-aspect-ratio, and every expression signal were dead channels for 98% of
the dataset, silently filled with neutral fallbacks.

YuNet finds the face box on the full frame (fast, robust to small faces), the box
is expanded and cropped, and MediaPipe then runs *on that crop*, where the face
fills the frame and eyelid/mouth landmarks are reliable. Detection went to
**~97%**, and holdout AUC from 0.887 to **0.959**.

Cache keys are namespaced by detector, so switching backends never returns stale
crops.

---

## The hard-reject is off, on purpose

Stage 2 originally slammed frames to −∞ for closed eyes, severe blur, or clipped
exposure. An audit of all 126 hard-rejected frames found **29 keepers** among
them. The exposure rule was the worst offender — a dark stage background reads as
"clipped" — and was dropping roughly 23% of good frames.

The MLP already sees sharpness, exposure and EAR as input features, so it learns
to rank those frames down where they genuinely are worse, without a cliff. The
metrics are still computed and written to `ranked.csv`; they're informational.

If you re-enable it (`technical.hard_reject_enabled = True`), audit the rejects
before trusting the output.

---

## Shot scale

Face geometry was the obvious approach and it failed for the same reason
MediaPipe did: with a face detected in 2% of frames, `face_frac` collapsed
everything to "wide."

Instead each frame is scored on a continuous axis inside the SigLIP2 joint space
that search already uses:

```
closeness = cos(emb_full, close_prototype) - cos(emb_full, wide_prototype)
```

Each prototype is the mean text embedding of a small prompt ensemble. No pixels
are reprocessed — this runs on cached embeddings. Two thresholds split the axis
into wide / medium / close, calibrated by visual audit to roughly 40 / 55 / 6 %
on the reference shoot.

**The prompts and the thresholds are a matched set.** `closeness` is defined
relative to those specific prototypes, so editing the prompt ensemble in
`shots.py` invalidates `wide_max` and `close_min` — retune both together.

Note that "close" means *tight framing*, not "contains a face" — a frame-filling
robot is close.

---

## Subject and hero tagging

YOLOv8-pose gives person boxes; from them plus a background-brightness measure,
each frame gets a cached raw vector:

```
[largest_person_area, n_prominent_persons, second_person_area, bg_bright_frac]
```

`bg_bright_frac` is the fraction of *non-person* pixels above a luma floor,
measured on a 200px-long-side grayscale copy. A clean dark stage sits near 0; a
lit screen, an equipment rack, or a slide lights it up.

Classification is applied on top at read time:

- **subject** = `people` if `area >= person_min_area`, else `stage`
- **hero** = big enough, no second prominent person, and a dark background

Because the raw vector is what's cached, **re-tuning any threshold costs
nothing** — no images are reprocessed, only the run needs re-ingesting to rewrite
the columns. These thresholds were set by eye and are the most obviously
improvable part of the system; a hand-marked set of true hero shots would let
them be fit properly.

---

## Semantic search

The SigLIP2 text tower shares a joint space with the cached full-frame image
embeddings, so text→image search needs no new model and no re-embedding — just a
dot product against `search_index.npy`. Warm queries are ~50 ms.

Results are ordered by pure cosine relevance and collapsed to one hit per burst
(the best-matching frame), so a search doesn't return eight copies of one moment.

This only works when the context encoder has a text tower. SigLIP2 (the default)
does; DINOv2 and C-RADIO don't, and search is disabled if you switch to them.

---

## Tried and rejected

| Idea | Outcome |
|---|---|
| **VLM pairwise tournament** (stage 3) | Worked — a hosted VLM behind an internal API gateway, Swiss-style pairings with Bradley-Terry aggregation — but cost $1–2 per run and traded away AUC. Removed from the production path. |
| **Must-find as a third class** | Improves recall of memorized must-finds, doesn't generalize past a 0.1 pair fraction. Off. |
| **SigLIP body-crop embedding** | The pose ablation showed it doesn't earn its inference cost. Off; saves one encoder pass per image. |
| **Valence/arousal (HSEmotion)** | Never installed; the channel was zeros for the entire project. Removed rather than left dangling. |
| **Face-geometry shot classification** | Killed by the 2% detection rate. Replaced by the zero-shot axis. |
| **Soft technical penalty** (`alpha > 0`) | Hurt must-find recall — face-quality metrics are unreliable when the face is small. Set to 0. |

---

## Worth trying next

Roughly in order of cost-to-value:

1. **Burst score = mean of top 3** rather than the single representative. Large
   bursts currently win on an order statistic — with 30 frames, one gets lucky.
2. **Cross-event evaluation.** Train on one shoot, evaluate on another. This is
   the only honest test of whether the model transfers, and it has never been run.
3. **Fit the scene thresholds** against hand-marked hero shots instead of by eye.
4. **List-wise loss per burst** instead of pairwise — treats each mixed burst as
   a softmax ranking task, a stronger signal where several frames are good.
5. **Down-weight in-burst pairs by similarity** (`1 - cos^k`) to remove
   label-noise gradient from the ~30% of failure cases that are near-identical
   pairs labeled differently.
6. **A feedback loop in the UI** — approve/reject buttons writing to a picks
   file, used as relabeling input for the next training round.

"""Auto-export: stream the best frames of a live shoot into review folders.

As photos arrive, each output folder (Best, Hero, Wide, …, one per shot-list
prompt, and Starred) receives the frames that rank in its top ``top_percent``.
Reviewers work from those folders instead of the whole card dump.

Design notes — each was settled by replaying a real shoot in arrival order
(GTC Taipei, 702 frames, card dumps every 10 minutes):

**Strictness is a percentile within each folder, never a score cutoff.** The
ranker's raw score shifts between sessions (p90 moved 2.9 → 4.7) and between
cameras (medians 2.5 apart), and frames tagged hero score far below the shoot
overall — a global cutoff would never export a hero.

**Strictness is a share of photos, not moments.** Each folder sends the top
``top_percent`` of its own photos (10% of 700 ≈ 70). Burst-mates compete like any
other photo; the Look-alikes setting decides how many near-identical frames get
through. (Ranking moments instead gave ~1 photo per moment: 12 in Best from 700,
because burst-mates are almost always look-alikes of each other.)

**Early batches are handled by confidence, not a warm-up.** A photo ranked r-th
of the n seen so far is in the output's true top p% with probability
P(Binomial(n, p) ≥ r). It exports once that reaches ``confidence``. With 10
bursts seen only a clear standout clears it; later anything well inside the
top p% goes immediately and borderline frames wait. Versus plain top-% from the
first batch this cut over-export from ~1.5× to ~1.1× and eliminated frames that
finished below the median; a fixed warm-up (wait for N frames) exported nothing
at all for small categories, and a running budget throttled exactly the clumps
of great frames a keynote produces.

**Exports are append-only.** Reviewers may already be using a file, so nothing
is ever withdrawn, even if its rank later drops. What's been sent is recorded
per shoot, so a restart never re-exports.

**A burst exports only once it's finished** — a later frame from the same
camera has arrived beyond the burst merge gap, or it has sat unchanged for
``settle_seconds`` — so the best frame of a burst still being shot isn't beaten
by its own next frame after it's gone out.

**Look-alikes are suppressed, in and across bursts.** The same pose in front of
the same slide minutes later is a new burst. A photo clearly better than every
similar one already sent (BETTER_LOOKALIKE_SD) still goes — exports are never
withdrawn, so otherwise arrival order alone would decide which version ships.
In the top 20% of bursts, ~30% were near-duplicates (cosine ≥ 0.9) of a better
pick. ``variety`` sets that cutoff per folder.

**Shot-list prompts must beat generic descriptions of the event.** Raw
text-image similarity has a different scale per prompt, and a prompt with
nothing to match would otherwise fill its folder with its "best" non-matches. A
frame matches when the prompt describes it better than every background prompt
by ``match`` (so "audience cheering" on a shoot without audience shots matches
nothing, instead of the top stage shots).

**"Text on screen" shot-list entries match what the photo *says*** — the
slide, a screen, a sign — read by OCR (ocr.py), matched forgivingly
(textmatch.py). Like the visual ones, every matching frame goes. A frame whose
text hasn't been read yet counts as waiting, so it's considered once it has.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .textmatch import match_scores as text_match_scores

SETTINGS_FILE = "auto_export.json"
STATE_FILE = "auto_export_state.json"
AUTO_SUBDIR = "Auto"

# Built-in outputs: folder name → frame filter over ranked.csv columns.
CATEGORIES: Dict[str, Callable[[pd.DataFrame], "pd.Series"]] = {
    "Best": lambda d: pd.Series(True, index=d.index),
    "Hero": lambda d: d["is_hero"].astype(str).str.lower().isin(["true", "1"]),
    "Close": lambda d: d["shot_type"] == "close",
    "Medium": lambda d: d["shot_type"] == "medium",
    "Wide": lambda d: d["shot_type"] == "wide",
    "People": lambda d: d["subject_class"] == "people",
    "Stage": lambda d: d["subject_class"] == "stage",
}
STARRED = "Starred"

# What a keynote photo is "about" by default. A prompt only matches a frame it
# describes better than all of these.
BACKGROUND_PROMPTS = [
    "a photo of a stage presentation",
    "a photo of a person speaking on stage",
    "a photo of a large screen",
    "a photo of a conference",
    "a dark photo",
    "a photo of a keynote",
]

CONFIDENCE_LEVELS = {"fast": 0.5, "balanced": 0.7, "careful": 0.9}
VARIETY_LEVELS = {"off": 1.01, "some": 0.95, "more": 0.92, "most": 0.88}
MATCH_LEVELS = {"loose": 0.0, "normal": 0.01, "strict": 0.025}
PROMPT_KINDS = ("describe", "text")     # looks like the description | the text is on screen


# --------------------------------------------------------------------- settings

@dataclass
class ShotPrompt:
    name: str
    text: str
    match: str = "normal"          # loose | normal | strict (describe only)
    kind: str = "describe"         # describe | text (OCR: the words are on screen)


@dataclass
class AutoExportSettings:
    enabled: bool = False
    top_percent: float = 10.0      # strictness: top X% of each folder's bursts
    speed: str = "balanced"        # fast | balanced | careful → confidence
    min_per_folder: int = 3        # every folder aims for at least this many moments
    variety: str = "more"          # look-alike suppression across bursts
    outputs: Dict[str, bool] = field(default_factory=lambda: {
        "Best": True, "Hero": True, "Close": True, "Medium": False, "Wide": True,
        "People": False, "Stage": False, STARRED: True})
    prompts: List[ShotPrompt] = field(default_factory=list)
    settle_seconds: float = 120.0
    session_gap_minutes: float = 45.0  # a capture gap this long starts a new session (0 = one session)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AutoExportSettings":
        """Parse and clamp untrusted input. Unknown keys are ignored."""
        d = d or {}
        s = cls()
        s.enabled = bool(d.get("enabled", s.enabled))
        try:
            s.top_percent = float(min(100.0, max(0.5, float(d.get("top_percent", s.top_percent)))))
        except (TypeError, ValueError):
            pass
        if d.get("speed") in CONFIDENCE_LEVELS:
            s.speed = d["speed"]
        try:
            s.min_per_folder = int(min(50, max(0, int(d.get("min_per_folder", s.min_per_folder)))))
        except (TypeError, ValueError):
            pass
        if d.get("variety") in VARIETY_LEVELS:
            s.variety = d["variety"]
        outs = d.get("outputs")
        if isinstance(outs, dict):
            s.outputs = {k: bool(outs.get(k, v)) for k, v in s.outputs.items()}
        prompts = []
        seen = set()
        for p in (d.get("prompts") or [])[:20]:
            if not isinstance(p, dict):
                continue
            text = str(p.get("text", "")).strip()[:200]
            name = folder_name(str(p.get("name", "")).strip() or text)
            if not text or not name or name in seen or name in CATEGORIES or name == STARRED:
                continue
            seen.add(name)
            match = p.get("match") if p.get("match") in MATCH_LEVELS else "normal"
            kind = p.get("kind") if p.get("kind") in PROMPT_KINDS else "describe"
            prompts.append(ShotPrompt(name=name, text=text, match=match, kind=kind))
        s.prompts = prompts
        try:
            s.settle_seconds = float(min(3600.0, max(0.0, float(d.get("settle_seconds", s.settle_seconds)))))
        except (TypeError, ValueError):
            pass
        try:
            s.session_gap_minutes = float(min(24 * 60.0, max(0.0, float(d.get("session_gap_minutes", s.session_gap_minutes)))))
        except (TypeError, ValueError):
            pass
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def confidence(self) -> float:
        return CONFIDENCE_LEVELS[self.speed]


def folder_name(name: str) -> str:
    """A safe, readable folder name (keeps spaces and case)."""
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "", name).strip().strip(".")
    return name[:40].strip()


def load_settings(run_dir: Path) -> AutoExportSettings:
    try:
        return AutoExportSettings.from_dict(json.loads((run_dir / SETTINGS_FILE).read_text()))
    except (OSError, ValueError):
        return AutoExportSettings()


def save_settings(run_dir: Path, settings: AutoExportSettings) -> None:
    _write_json(run_dir / SETTINGS_FILE, settings.to_dict())


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


# ------------------------------------------------------------------- selection

def confidence_in_top(rank: int, n: int, p: float) -> float:
    """P(a burst ranked ``rank``-th best of ``n`` is in the true top ``p``).

    Its true percentile is Beta(n-rank+1, rank); for integer parameters that
    tail equals P(Binomial(n, p) >= rank) — "at least `rank` of the n seen
    would clear the real top-p cutoff". Log-space, so large n can't underflow.
    """
    if rank <= 0:
        return 1.0
    if rank > n or p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq
             for k in range(rank)]
    m = max(terms)
    return max(0.0, 1.0 - math.exp(m) * sum(math.exp(t - m) for t in terms))


@dataclass
class Planned:
    path: str                 # the ranked frame (what "already sent" is keyed by)
    output: str
    dest_name: str
    reason: str
    source: str = ""          # file actually copied: the RAW twin if there is one


@dataclass
class PlanResult:
    exports: List[Planned]
    waiting: Dict[str, int]              # output → qualifying bursts still settling
    lookalikes: Dict[str, int]           # output → bursts skipped as look-alikes
    matches: Dict[str, int]              # prompt output → matching frames
    unread: Dict[str, int] = field(default_factory=dict)   # text-on-screen output → frames not read yet


def dest_name(row) -> str:
    """The photo's own file name, unchanged — reviewers and their tools match
    exports back to the camera originals by name. (A *different* photo with the
    same name gets a numbered suffix at copy time, never an overwrite.)"""
    return Path(str(row["path"])).name


def prompt_margin(emb: np.ndarray, prompt_vec: np.ndarray, background: np.ndarray) -> np.ndarray:
    """How much better the prompt describes each frame than the best generic
    description does. > 0 means it's genuinely about the prompt."""
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    q = prompt_vec / max(float(np.linalg.norm(prompt_vec)), 1e-12)
    b = background / np.maximum(np.linalg.norm(background, axis=1, keepdims=True), 1e-12)
    return (e @ q) - (e @ b.T).max(axis=1)


def prompt_match_mask(emb: Optional[np.ndarray], prompt_vec: np.ndarray, background: np.ndarray,
                      margin: float) -> np.ndarray:
    if emb is None:
        return np.zeros(0, dtype=bool)
    return prompt_margin(emb, prompt_vec, background) > margin


def plan(
    frames: pd.DataFrame,
    settings: AutoExportSettings,
    exported: Dict[str, Dict[str, str]],
    first_seen: Dict[str, float],
    *,
    now: float,
    emb: Optional[np.ndarray] = None,
    prompt_vecs: Optional[Dict[str, np.ndarray]] = None,
    background_vecs: Optional[np.ndarray] = None,
    starred: Sequence[str] = (),
    burst_gap_seconds: float = 8.0,
    final: bool = False,
    ocr_texts: Optional[Sequence[Optional[str]]] = None,
) -> PlanResult:
    """Decide which frames to export now. Pure: no I/O, no state mutation.

    ``frames`` is ranked.csv (rows aligned with ``emb``); ``exported`` maps
    output → {path: dest name} of what's already been sent; ``first_seen`` maps
    path → when it first appeared. ``final=True`` treats every burst as settled
    (used for previews and the end of an event). ``ocr_texts`` is the text read
    on each frame (None = not read yet), for "text on screen" entries.
    """
    out = PlanResult([], {}, {}, {})
    if frames.empty:
        return out
    df = frames.reset_index(drop=True).copy()
    df["_t"] = pd.to_datetime(df.get("captured_at"), errors="coerce")
    cam = df["camera_id"].astype(str) if "camera_id" in df.columns else pd.Series("", index=df.index)
    df["_cam"] = cam.fillna("")
    df["_seen"] = [first_seen.get(str(p), now) for p in df["path"]]
    unit = df["burst_id"] if "burst_id" in df.columns else pd.Series(range(len(df)), index=df.index)
    df["_b"] = unit.values
    # Sessions: raw scores drift between sessions (an evening event scored ~1.5
    # higher at p90 than the next morning's keynote), so a shoot-wide
    # percentile hands one session most of the picks. Each session competes
    # only with itself.
    df["_sess"] = 0
    if settings.session_gap_minutes > 0 and df["_t"].notna().any():
        order = df["_t"].sort_values(na_position="last")
        gaps = order.diff().dt.total_seconds().fillna(0) > settings.session_gap_minutes * 60
        df.loc[order.index, "_sess"] = gaps.cumsum().values
    # A burst belongs to one session (its first frame's).
    df["_sess"] = df.groupby("_b")["_sess"].transform("min")
    emb_n = None
    if emb is not None and len(emb) == len(df):
        emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    row_of = {str(p): i for i, p in enumerate(df["path"])}

    # Burst "finished" test, shared by every output.
    cam_latest = df.groupby("_cam")["_t"].max()
    g = df.groupby("_b")
    b_last_t, b_cam, b_seen = g["_t"].max(), g["_cam"].first(), g["_seen"].max()
    settled = {}
    for b in b_last_t.index:
        later = cam_latest.get(b_cam[b])
        by_camera = (pd.notna(b_last_t[b]) and pd.notna(later)
                     and (later - b_last_t[b]).total_seconds() > burst_gap_seconds)
        settled[b] = final or by_camera or (now - b_seen[b]) >= settings.settle_seconds

    # (folder, frame mask, column to rank by, quality-gated?)
    outputs: List[Tuple[str, "pd.Series", str, bool]] = []
    for name, filt in CATEGORIES.items():
        if settings.outputs.get(name):
            try:
                outputs.append((name, filt(df), "score_s1", True))
            except KeyError:
                continue                       # older run without that tag column
    unread: Dict[str, int] = {}
    for k, sp in enumerate(settings.prompts):
        if sp.kind == "text":
            if ocr_texts is None or len(ocr_texts) != len(df):
                continue
            col = f"_match{k}"
            df[col] = text_match_scores(ocr_texts, sp.text)
            m = df[col] > 0
            out.matches[sp.name] = int(m.sum())
            unread[sp.name] = sum(t is None for t in ocr_texts)
            outputs.append((sp.name, m, col, False))
            continue
        vec = (prompt_vecs or {}).get(sp.text)
        if vec is None or emb_n is None or background_vecs is None:
            continue
        # A shot-list folder ranks by how well frames match the description,
        # not by the ranker: it was trained on photos of people and scores
        # screen/product shots — exactly what shot lists ask for — low. (On
        # GTC, "computer chip" exported a server rack and blocked the three
        # real chip shots.) The match strictness is its quality filter.
        col = f"_match{k}"
        df[col] = prompt_margin(emb_n, vec, background_vecs)
        m = df[col] > MATCH_LEVELS[sp.match]
        out.matches[sp.name] = int(m.sum())
        outputs.append((sp.name, m, col, False))

    p = settings.top_percent / 100.0
    c = settings.confidence
    thr = VARIETY_LEVELS[settings.variety]
    # Each photo's standing in its whole session, for the min-per-folder quality gate.
    shoot_rank = df.groupby("_sess")["score_s1"].rank(ascending=False, method="first").astype(int)
    n_shoot = df.groupby("_sess")["score_s1"].transform("size")
    for name, mask, rank_col, gated in outputs:
        out.waiting[name] = out.lookalikes[name] = 0
        sent = exported.get(name, {})
        sent_rows = [row_of[s] for s in sent if s in row_of]
        folder = df[mask.values]
        if not gated:
            # Shot-list folder: every frame that matches the description goes,
            # strongest match first — no quality score, no percentage.
            _select_matches(out, df, folder, name, sent, sent_rows, settled, emb_n, thr, rank_col,
                            "text on screen" if name in unread else "matches")
            if unread.get(name):
                out.unread[name] = unread[name]           # not read yet → may still match
            continue
        # The folder minimum is folder-wide: k best photos overall, expressed as
        # a percentage every session shares — not k per session, which would
        # hand small sessions an outsized share.
        n_folder = len(folder)
        p_out = max(p, min(1.0, settings.min_per_folder / n_folder)) if settings.min_per_folder and n_folder else p
        for _sess, sess_sub in folder.groupby("_sess"):
            _select_folder(out, df, sess_sub, name, sent, sent_rows, settings, settled, emb_n,
                           p, p_out, c, thr, shoot_rank, n_shoot, rank_col, gated)

    if settings.outputs.get(STARRED):
        sent = exported.get(STARRED, {})
        for path in sorted(set(map(str, starred))):
            if path in row_of and path not in sent:
                src = _original(df, row_of[path])
                out.exports.append(Planned(path, STARRED, Path(src).name, "starred", src))
    return out


# A look-alike still goes in when it scores at least this many standard
# deviations (of the folder's own scores) above every similar photo already
# sent. Exports are never withdrawn, so without this the first photo of a scene
# to arrive would block a clearly better one that arrives on a later card. Near
# ties (2.21 vs 2.19) still count as repeats.
BETTER_LOOKALIKE_SD = 0.25


def _is_lookalike(i: int, sent_rows: List[int], emb_n, thr: float, score=None, margin: float = 0.0) -> bool:
    """Too similar to something already in this folder — whether from the same
    burst (the frame next to it) or another one (same pose, same slide, minutes
    later). One rule for both; ``thr`` comes from the Look-alikes setting.
    With ``score``, a photo clearly better than every similar one already sent
    is not treated as a look-alike."""
    if emb_n is None or not sent_rows:
        return False
    sims = emb_n[sent_rows] @ emb_n[i]
    similar = sims >= thr
    if not similar.any():
        return False
    if score is None:
        return True
    best_similar = max(float(score[j]) for j, hit in zip(sent_rows, similar) if hit)
    return float(score[i]) < best_similar + margin


def _original(df, i) -> str:
    """The file to hand out for frame ``i``: its RAW twin when it has one."""
    raw = df.at[i, "raw_path"] if "raw_path" in df.columns else ""
    return raw if isinstance(raw, str) and raw and Path(raw).is_file() else str(df.at[i, "path"])


def _send_frame(out, df, i, name, sent_rows, reason) -> None:
    src = _original(df, i)
    out.exports.append(Planned(str(df.at[i, "path"]), name, Path(src).name, reason, src))
    sent_rows.append(i)


def _select_folder(out, df, sub, name, sent, sent_rows, settings, settled, emb_n, p, p_out, c, thr,
                   shoot_rank, n_shoot, rank_col="score_s1", gated=True) -> None:
    """One category folder, one session: its top ``p`` of photos.

    Photos are ranked individually; the photo ranked r-th of n goes once it's
    confidently in the top p (see :func:`confidence_in_top`), plus the folder
    minimum (gated on standing in the whole session). Burst-mates compete like
    any other photo — the Look-alikes setting decides how many near-identical
    frames get through. Appends to ``out``."""
    sub = sub.sort_values(rank_col, ascending=False)
    n = len(sub)
    score = df[rank_col].to_numpy()
    margin = BETTER_LOOKALIKE_SD * float(sub[rank_col].std() or 0.0)
    waiting = skipped = 0
    for r, i in enumerate(sub.index, start=1):
        if confidence_in_top(r, n, p_out) < c:
            break                          # everything below is less certain still
        if gated and confidence_in_top(r, n, p) < c and \
                confidence_in_top(int(shoot_rank[i]), int(n_shoot[i]), 0.5) < c:
            continue                       # only the folder minimum would admit it, and it isn't
                                           # clearly in the top half of the shoot — don't send filler
        if str(df.at[i, "path"]) in sent:
            continue
        if not settled[df.at[i, "_b"]]:
            waiting += 1
            continue
        if _is_lookalike(i, sent_rows, emb_n, thr, score, margin):
            skipped += 1
            continue
        _send_frame(out, df, i, name, sent_rows, f"#{r} of {n} photos ({confidence_in_top(r, n, p_out):.0%} sure)")
    out.waiting[name] += waiting
    out.lookalikes[name] += skipped


def _select_matches(out, df, sub, name, sent, sent_rows, settled, emb_n, thr, match_col,
                    label: str = "matches") -> None:
    """One shot-list folder: every matching frame, strongest match first, each
    unless it's a look-alike of something already sent (waits, like every
    folder, until its burst is finished)."""
    waiting = skipped = 0
    score = df[match_col].to_numpy()
    margin = BETTER_LOOKALIKE_SD * float(sub[match_col].std() or 0.0)
    for i in sub.sort_values(match_col, ascending=False).index:
        if str(df.at[i, "path"]) in sent:
            continue
        if not settled[df.at[i, "_b"]]:
            waiting += 1
            continue
        if _is_lookalike(i, sent_rows, emb_n, thr, score, margin):
            skipped += 1
            continue
        _send_frame(out, df, i, name, sent_rows, f"{label} ({df.at[i, match_col]:+.3f})")
    out.waiting[name] += waiting
    out.lookalikes[name] += skipped


# ---------------------------------------------------------------------- runner

@dataclass
class RunData:
    run_dir: Path
    shoot: str                         # export subfolder name
    frames: pd.DataFrame
    emb: Optional[np.ndarray]
    starred: List[str]
    ocr: Optional[List[Optional[str]]] = None    # text read on each frame (None = not yet)


class AutoExporter:
    """Background worker: applies :func:`plan` to every enabled shoot and copies
    the chosen originals, after each ingest, on each star, and on a slow tick
    (which is what lets bursts settle once arrivals stop).

    Server-agnostic: the caller supplies how to read a run, embed text, and
    where exports go, which keeps this testable without Flask.
    """

    def __init__(self, get_run: Callable[[int], Optional[RunData]],
                 embed_texts: Callable[[List[str]], np.ndarray],
                 export_root: Callable[[], Optional[Path]],
                 list_runs: Callable[[], List[int]],
                 *, tick_seconds: float = 30.0, burst_gap_seconds: float = 8.0):
        self._get_run, self._embed, self._root, self._list_runs = get_run, embed_texts, export_root, list_runs
        self.tick_seconds, self.burst_gap_seconds = tick_seconds, burst_gap_seconds
        self._wake = threading.Event()
        self._pending: set = set()
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()      # one plan+copy pass at a time
        self._status: Dict[int, dict] = {}
        self._vecs: Dict[str, np.ndarray] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="auto-export", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set(); self._wake.set()

    def kick(self, run_id: int) -> None:
        with self._lock:
            self._pending.add(int(run_id))
        self._wake.set()

    def status(self, run_id: int) -> dict:
        return dict(self._status.get(int(run_id), {}))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.tick_seconds)
            self._wake.clear()
            with self._lock:
                ids, self._pending = set(self._pending), set()
            try:
                ids |= set(self._list_runs())         # enabled shoots get the tick too
            except Exception:  # noqa: BLE001
                pass
            for rid in sorted(ids):
                try:
                    self.run_once(rid)
                except Exception as exc:  # noqa: BLE001 — never let one shoot kill the worker
                    self._status.setdefault(rid, {})["last_error"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------ helpers
    def prompt_vectors(self, texts: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        need = [t for t in list(texts) + BACKGROUND_PROMPTS if t not in self._vecs]
        if need:
            vecs = np.asarray(self._embed(need), dtype=np.float32)
            for t, v in zip(need, vecs):
                self._vecs[t] = v
        return ({t: self._vecs[t] for t in texts},
                np.stack([self._vecs[t] for t in BACKGROUND_PROMPTS]))

    def _vectors_for(self, settings: AutoExportSettings, data: RunData):
        texts = [p.text for p in settings.prompts if p.kind == "describe"]
        if not texts or data.emb is None:
            return {}, None
        return self.prompt_vectors(texts)

    @staticmethod
    def load_state(run_dir: Path) -> dict:
        try:
            st = json.loads((run_dir / STATE_FILE).read_text())
        except (OSError, ValueError):
            st = {}
        st.setdefault("exported", {}); st.setdefault("first_seen", {}); st.setdefault("log", [])
        return st

    def preview(self, run_id: int, settings: AutoExportSettings) -> dict:
        """What would be in each folder right now if auto-export had been on
        from the start. No files are touched."""
        data = self._get_run(run_id)
        if data is None:
            return {"error": "Shoot not found."}
        vecs, bg = self._vectors_for(settings, data)
        res = plan(data.frames, settings, {}, {}, now=time.time(), emb=data.emb, prompt_vecs=vecs,
                   background_vecs=bg, starred=data.starred, burst_gap_seconds=self.burst_gap_seconds,
                   final=True, ocr_texts=data.ocr)
        folders: Dict[str, List[str]] = {}
        for e in res.exports:
            folders.setdefault(e.output, []).append(e.path)
        return {"folders": {k: {"count": len(v), "sample": v[:8]} for k, v in folders.items()},
                "matches": res.matches, "lookalikes": res.lookalikes, "n_frames": int(len(data.frames))}

    # ------------------------------------------------------------ one pass
    def run_once(self, run_id: int, *, final: bool = False) -> dict:
        from .export import (copy_atomic, ensure_export_root, export_overlaps_source, mark_export_dir,
                             source_root)
        with self._run_lock:
            data = self._get_run(run_id)
            if data is None:
                return {}
            settings = load_settings(data.run_dir)
            status = self._status.setdefault(int(run_id), {})
            status["enabled"] = settings.enabled
            if not settings.enabled:
                return status
            root = self._root()
            if root is None:
                status["last_error"] = "No export folder configured on the server."
                return status
            state = self.load_state(data.run_dir)
            now = time.time()
            for p in data.frames["path"]:
                state["first_seen"].setdefault(str(p), now)
            vecs, bg = self._vectors_for(settings, data)
            res = plan(data.frames, settings, state["exported"], state["first_seen"], now=now,
                       emb=data.emb, prompt_vecs=vecs, background_vecs=bg, starred=data.starred,
                       burst_gap_seconds=self.burst_gap_seconds, final=final, ocr_texts=data.ocr)
            base = root / data.shoot / AUTO_SUBDIR
            src = source_root(list(data.frames["path"]))
            if export_overlaps_source(base, src):
                status["last_error"] = (f"The host export folder would put exports inside this run's photo "
                                        f"folder ({src}). Choose a different host export folder.")
                _write_json(data.run_dir / STATE_FILE, state)
                return status
            try:
                ensure_export_root(root)
                mark_export_dir(root / data.shoot)
            except OSError as exc:
                status["last_error"] = f"Export folder unavailable: {exc.strerror or exc}"
                _write_json(data.run_dir / STATE_FILE, state)
                return status
            sent = failed = 0
            last_error = ""
            for e in res.exports:
                try:
                    name = copy_atomic(Path(e.source or e.path), mark_export_dir(base / e.output), e.dest_name)
                except OSError as exc:
                    failed += 1
                    last_error = f"{Path(e.path).name}: {exc.strerror or exc}"
                    continue                        # stays unsent → retried next pass
                state["exported"].setdefault(e.output, {})[e.path] = name
                state["log"] = (state["log"] + [[now, e.output, name, e.reason]])[-200:]
                sent += 1
            _write_json(data.run_dir / STATE_FILE, state)
            status.update({
                "last_run": now,
                "last_sent": sent,
                "failed": failed,
                "last_error": last_error if failed else "",
                "sent": {k: len(v) for k, v in state["exported"].items()},
                "waiting": {k: v for k, v in res.waiting.items() if v},
                "lookalikes": {k: v for k, v in res.lookalikes.items() if v},
                "matches": res.matches,
                "unread": max(res.unread.values(), default=0),
                "dest": f"{root.name}/{data.shoot}/{AUTO_SUBDIR}",
                "recent": state["log"][-8:],
            })
            return status

"""Read the text in every photo — slides, screens, signs — on the GPU.

PP-OCRv4 (English detector + recognizer, via RapidOCR) compiled with
TensorRT. Measured on 60 GTC Taipei stage photos at 2048 px on the GB10,
images already decoded:

    docTR (fast_base + crnn_vgg16_bn, fp16, PyTorch)   ~160 ms/photo
    PP-OCRv4, PyTorch, 2 workers                         97 ms/photo
    PP-OCRv4, TensorRT TF32, 1 worker                    44 ms/photo
    PP-OCRv4, TensorRT TF32, 3 workers                   23 ms/photo

TensorRT output is identical to PyTorch's. An fp16 conversion of the ONNX
models was slower (57 ms) *and* broke recognition (weights exceed fp16 range),
so engines are built TF32.

End to end, decoding JPEGs from disk included, the whole 703-photo GTC shoot
takes ~21 s with 4 workers (3: 26 s, 5: 19 s). Decoding is the expensive half
(~70 ms a photo even at reduced scale), so each worker decodes the next photo
on a second thread while the GPU reads the current one. Each worker holds
~0.9 GB of GPU memory.

**Every frame is read**, not one per burst: burst-mates often catch a slide
mid-transition. **It never competes with ingest** — workers only get new photos
while no ingest is running, and run at low CPU priority.

Several worker *processes*, not threads: each has its own CUDA context and
TensorRT engine, which is what lets them overlap on the GPU (the detector alone
can't fill it). The first worker builds and caches the engines (~40 s, once
per machine), then the rest start from that cache.

Results are cached by file content (sha1, the same key as the feature cache),
so re-ingests, restarts and a photo appearing in two runs never re-read it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from .config import OcrConfig

# v2: each line also carries where it is (for highlighting) and photos are
# read upright (EXIF rotation applied). v1 entries are simply re-read.
CACHE_KIND = "ocr-ppocrv4-en-v2"
INFLIGHT_PER_WORKER = 3          # one on the GPU, one decoded and waiting, one decoding


# ------------------------------------------------------------- worker process

def _patch_tensorrt_builder() -> None:
    """RapidOCR 3.9's engine builder predates TensorRT 10/11: it asks for the
    removed EXPLICIT_BATCH flag and the FP16 builder flag (TensorRT 11 is
    strongly typed). Patched here, at runtime, so a library upgrade can't
    silently undo it — and so it's a no-op on versions that still have both."""
    import tensorrt as trt
    from rapidocr.inference_engine.tensorrt import engine_builder as eb
    from rapidocr.utils.log import logger

    if getattr(eb.TRTEngineBuilder, "_image_filterer_patched", False):
        return

    # One shared, errors-only logger. RapidOCR makes a new trt.Logger per
    # engine, and TensorRT warns ("logger … differs") about every one after the
    # first, plus routine notes about DLA fallback — noise in the server log.
    _shared = trt.Logger(trt.Logger.ERROR)

    class _Logger:
        INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE = (trt.Logger.INTERNAL_ERROR, trt.Logger.ERROR,
                                                         trt.Logger.WARNING, trt.Logger.INFO, trt.Logger.VERBOSE)
        Severity = trt.Logger.Severity

        def __new__(cls, *args, **kwargs):
            return _shared

    trt.Logger = _Logger

    def build(self):
        builder = trt.Builder(self.trt_logger)
        flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
        network = builder.create_network((1 << int(flag)) if flag is not None else 0)
        parser = trt.OnnxParser(network, self.trt_logger)
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.cfg.get("workspace_size", 1 << 30))
        if (hasattr(trt.BuilderFlag, "FP16") and self.cfg.get("use_fp16", False)
                and getattr(builder, "platform_has_fast_fp16", False)):
            config.set_flag(trt.BuilderFlag.FP16)
        elif hasattr(trt.BuilderFlag, "TF32"):
            config.set_flag(trt.BuilderFlag.TF32)
        profile = builder.create_optimization_profile()
        self._set_dynamic_shapes(network, profile)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT engine")
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.engine_path.with_name(f".{self.engine_path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(serialized)
        tmp.replace(self.engine_path)            # never leave a half-written engine behind
        return trt.Runtime(self.trt_logger).deserialize_cuda_engine(serialized)

    eb.TRTEngineBuilder.build = build
    eb.TRTEngineBuilder._image_filterer_patched = True


def build_reader(model_dir: Path, engine: str, max_side: int):
    """A RapidOCR instance: English PP-OCRv4 mobile det + rec, no angle
    classifier (stage text is upright; it costs time and needs extra models)."""
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    eng = EngineType.TENSORRT if engine == "tensorrt" else EngineType.TORCH
    if eng == EngineType.TENSORRT:
        _patch_tensorrt_builder()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    opt_h = max(32, round(max_side * 9 / 16 / 32) * 32)      # a 16:9-ish frame at max_side
    return RapidOCR(params={
        "Global.use_cls": False,
        "Global.log_level": "error",
        "Global.model_root_dir": str(model_dir),
        "Global.max_side_len": max_side,
        "Det.engine_type": eng, "Det.lang_type": LangDet.EN,
        "Det.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV4,
        "Det.limit_side_len": max_side, "Det.limit_type": "max",
        "Rec.engine_type": eng, "Rec.lang_type": LangRec.EN,
        "Rec.model_type": ModelType.MOBILE, "Rec.ocr_version": OCRVersion.PPOCRV4,
        "EngineConfig.torch.use_cuda": True,
        "EngineConfig.tensorrt.use_fp16": False,
        "EngineConfig.tensorrt.cache_dir": str(model_dir / "tensorrt"),
        "EngineConfig.tensorrt.det_profile.opt_shape": [1, 3, opt_h, max_side],
        "EngineConfig.tensorrt.det_profile.max_shape": [1, 3, max_side, max_side],
    })


def load_for_ocr(path, max_side: int):
    """The photo as a BGR array, upright, about ``max_side`` on its long edge.
    JPEG draft mode decodes at reduced scale directly — ~4× faster than full
    size."""
    import numpy as np
    from PIL import ImageOps
    from .imaging import open_image

    with open_image(path) as im:
        # draft() only shrinks while BOTH sides stay >= the request, so ask for
        # the photo's own aspect: (2048, 2048) would keep a 6000×4000 frame at
        # full size (2000 < 2048) and cost ~3× the OCR itself to decode.
        f = max_side / max(im.size)
        if f < 1:
            im.draft("RGB", (max(1, int(im.size[0] * f)), max(1, int(im.size[1] * f))))
        if im.getexif().get(0x0112, 1) != 1:
            # A portrait frame is stored sideways with a rotation flag; read
            # sideways, its text is mostly missed (and boxes would land wrong).
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max(im.size) > max_side * 1.25:
            im.thumbnail((max_side, max_side))
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def read_lines(reader, img, min_score: float) -> List[List[object]]:
    """[[text, score, box], …]; ``box`` is the line's four corners as
    fractions of the (upright) photo's width and height, so it can be drawn
    over the photo at any size."""
    r = reader(img)
    h, w = img.shape[:2]
    boxes = list(r.boxes) if r.boxes is not None else [None] * len(r.txts or ())
    out = []
    for t, sc, box in zip(r.txts or (), r.scores or (), boxes):
        if float(sc) < min_score or not str(t).strip():
            continue
        quad = [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in box] if box is not None else []
        out.append([str(t), round(float(sc), 3), quad])
    return out


def _exit_with_parent() -> None:
    """Die with the server. multiprocessing only cleans up daemon workers on a
    clean exit — a killed or crashed server would leave them holding ~1 GB of
    GPU memory each, and every restart would add four more."""
    parent = os.getppid()
    try:                                  # Linux: the kernel signals us at once
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass

    def watch() -> None:                  # anywhere else (or if prctl was missed): poll
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch, name="ocr-parent-watch", daemon=True).start()


def _worker_main(wid: int, model_dir: str, engine: str, max_side: int, min_score: float,
                 tasks, results) -> None:
    _exit_with_parent()
    try:
        os.nice(10)                       # decoding must never slow the ingest down
    except OSError:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        reader = build_reader(Path(model_dir), engine, max_side)
        import numpy as np
        reader(np.zeros((64, 256, 3), dtype=np.uint8))          # load the engines now
    except Exception as exc:  # noqa: BLE001
        results.put(("fatal", wid, None, f"{type(exc).__name__}: {exc}"))
        return
    results.put(("ready", wid, None, None))

    # Decode on a second thread while the GPU reads the previous photo: a JPEG
    # takes longer to decode (~70 ms) than to OCR (~45 ms), so doing them in
    # turn would leave the GPU idle most of the time.
    import queue as _queue
    decoded: "_queue.Queue" = _queue.Queue(maxsize=1)

    def decode_loop() -> None:
        while True:
            task = tasks.get()
            if task is None:
                decoded.put(None)
                return
            key, path = task
            try:
                decoded.put((key, load_for_ocr(path, max_side), None))
            except Exception as exc:  # noqa: BLE001
                decoded.put((key, None, f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=decode_loop, name="ocr-decode", daemon=True).start()
    while True:
        item = decoded.get()
        if item is None:
            return
        key, img, err = item
        if err is not None:
            results.put(("failed", wid, key, err))
            continue
        try:
            results.put(("done", wid, key, read_lines(reader, img, min_score)))
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the worker
            results.put(("failed", wid, key, f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------- the service

def pick_engine() -> Optional[str]:
    """"tensorrt" if it's installed, else "torch" (CUDA), else None (no GPU →
    OCR stays off: on the CPU it would take seconds per photo)."""
    import importlib.util
    if importlib.util.find_spec("rapidocr") is None:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None
    if importlib.util.find_spec("tensorrt") and importlib.util.find_spec("cuda"):
        return "tensorrt"
    return "torch"


class _Worker:
    def __init__(self, wid: int, proc, tasks) -> None:
        self.wid, self.proc, self.tasks = wid, proc, tasks
        self.ready = False
        self.inflight: Dict[str, str] = {}      # key → path


class OcrService:
    """Queue of photos to read, a pool of GPU worker processes, and the text
    read so far (by path). Thread-safe; all work happens on one dispatcher
    thread, started on the first :meth:`submit`.

    ``is_busy()`` → True pauses handing out new photos (an ingest is running).
    ``on_results(run_ids)`` is called, at most every couple of seconds, for runs
    that got new text — the server uses it to let auto-export pick them up.
    """

    def __init__(self, cfg: OcrConfig, cache_dir: Path, model_dir: Path, *,
                 is_busy: Callable[[], bool] = lambda: False,
                 on_results: Callable[[Set[int]], None] = lambda ids: None,
                 engine: Optional[str] = "auto") -> None:
        from .cache_io import FeatureCache, Sha1Memo
        self.cfg = cfg
        self.model_dir = Path(model_dir)
        # Chosen in activate(): checking for a GPU starts CUDA in this process,
        # which nothing that merely builds the app (tests, tools) should do.
        self._engine_choice = engine
        self.engine: Optional[str] = None
        self.enabled = cfg.workers > 0
        self._cache = FeatureCache(Path(cache_dir))
        self._memo = Sha1Memo(Path(cache_dir) / "sha1_memo.db")
        self._is_busy, self._on_results = is_busy, on_results
        self._lock = threading.Lock()
        self._intake: deque = deque()                     # (run_id, [paths], urgent)
        self._queue: "OrderedDict[str, str]" = OrderedDict()   # sha1 → path, front = next
        self._waiters: Dict[str, Set[int]] = {}           # sha1 → runs wanting it
        self._paths_of: Dict[str, Set[str]] = {}          # sha1 → every path with that content
        self._text: Dict[str, str] = {}                   # path → text ("" = none on it)
        self._lines: Dict[str, list] = {}                 # path → [[text, score, box], …]
        self._workers: List[_Worker] = []
        self._results = None
        self._ctx = mp.get_context("spawn")               # CUDA can't survive fork
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._changed: Set[int] = set()
        self._last_notify = 0.0
        self._restarts: deque = deque()
        self.error = "" if self.enabled else "turned off (IMAGE_FILTERER_OCR_WORKERS=0)"
        self.n_read = 0
        self._target = cfg.workers        # lowered if a worker can't start at all
        self._active = False

    # ------------------------------------------------------------- public
    def submit(self, run_id: int, paths: Iterable[str], *, urgent: bool = False) -> None:
        """Ask for text of ``paths``. Cheap: the work (and cache lookups) happen
        on the dispatcher thread. ``urgent`` jumps the queue (fresh photos of a
        live shoot go before back-filling an old one)."""
        if not self.enabled:
            return
        paths = [str(p) for p in paths]
        if not paths:
            return
        with self._lock:
            self._intake.append((int(run_id), paths, urgent))
        if self._active:
            self._start()
        self._wake.set()

    def activate(self) -> None:
        """Start reading. Separate from construction so creating the app (tests,
        tools) never spawns GPU processes; the server calls this at start-up."""
        if self.enabled and self.engine is None:
            self.engine = pick_engine() if self._engine_choice == "auto" else self._engine_choice
            if self.engine is None:
                self.enabled = False
                self.error = "unavailable: needs rapidocr and a CUDA GPU"
        self._active = True
        if self.enabled:
            self._start()

    def texts(self, paths: Sequence[str]) -> List[Optional[str]]:
        """Text read for each path; None if it hasn't been read (yet)."""
        with self._lock:
            return [self._text.get(str(p)) for p in paths]

    def lines(self, path: str) -> Optional[list]:
        """Each line read on the photo, with where it is; None if not read yet."""
        with self._lock:
            return self._lines.get(str(path))

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "engine": self.engine,
                "error": self.error,
                "queued": len(self._queue) + sum(len(w.inflight) for w in self._workers),
                "workers": sum(1 for w in self._workers if w.ready),
                "read": self.n_read,
                "paused": bool(self._queue) and self._busy(),
            }

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # ---------------------------------------------------------- internals
    def _busy(self) -> bool:
        try:
            return bool(self._is_busy())
        except Exception:  # noqa: BLE001
            return False

    def _start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="ocr", daemon=True)
            self._thread.start()

    def _spawn(self, wid: int) -> _Worker:
        if self._results is None:
            self._results = self._ctx.Queue()
        tasks = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_worker_main, name=f"ocr-{wid}", daemon=True,
            args=(wid, str(self.model_dir), self.engine, self.cfg.max_side, self.cfg.min_score,
                  tasks, self._results))
        proc.start()
        return _Worker(wid, proc, tasks)

    def _ensure_workers(self) -> None:
        """Worker 0 first: it builds the TensorRT engines if they aren't cached
        yet, and several processes building the same engine at once would each
        spend a minute doing it. The rest start once it's ready."""
        if not self._workers:
            self._workers.append(self._spawn(0))
            return
        if not any(w.ready for w in self._workers):
            return
        used = {w.wid for w in self._workers}
        for wid in range(self._target):
            if wid not in used:
                self._workers.append(self._spawn(wid))

    def _take_intake(self) -> None:
        with self._lock:
            items, self._intake = list(self._intake), deque()
        for run_id, paths, urgent in items:
            try:
                sha1s = self._memo.sha1_many([Path(p) for p in paths])
            except OSError:
                sha1s = []
                for p in paths:
                    try:
                        sha1s.append(self._memo.sha1(Path(p)))
                    except OSError:
                        sha1s.append("")
            hit = False
            front: List[str] = []
            for path, sha1 in zip(paths, sha1s):
                if not sha1:
                    continue
                with self._lock:
                    if path in self._text:
                        continue
                    done = [p for p in self._paths_of.get(sha1, ()) if p in self._text]
                    if done:
                        self._text[path], self._lines[path] = self._text[done[0]], self._lines[done[0]]
                        hit = True
                        continue
                meta = self._cache.load_meta(sha1, CACHE_KIND)
                with self._lock:
                    self._paths_of.setdefault(sha1, set()).add(path)
                    if meta is not None:
                        lines = meta.get("lines", [])
                        self._text[path], self._lines[path] = "\n".join(str(x[0]) for x in lines), lines
                        hit = True
                        continue
                    self._waiters.setdefault(sha1, set()).add(run_id)
                    if sha1 not in self._queue and not any(sha1 in w.inflight for w in self._workers):
                        self._queue[sha1] = path
                    if urgent and sha1 in self._queue:
                        front.append(sha1)
            with self._lock:
                for sha1 in reversed(front):
                    self._queue.move_to_end(sha1, last=False)
            if hit:
                self._changed.add(run_id)

    def _dispatch(self) -> None:
        if self._busy():
            return
        with self._lock:
            for w in self._workers:
                while w.ready and self._queue and len(w.inflight) < INFLIGHT_PER_WORKER:
                    sha1, path = self._queue.popitem(last=False)
                    w.inflight[sha1] = path
                    w.tasks.put((sha1, path))

    def _collect(self, timeout: float) -> None:
        if self._results is None:
            self._wake.wait(timeout)
            return
        deadline = time.time() + timeout
        while True:
            try:
                kind, wid, key, payload = self._results.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                return
            except (EOFError, OSError):
                return
            self._handle(kind, wid, key, payload)
            if time.time() >= deadline:
                return

    def _handle(self, kind, wid, key, payload) -> None:
        w = next((x for x in self._workers if x.wid == wid), None)
        if kind == "ready":
            if w:
                w.ready = True
            return
        if kind == "fatal":
            with self._lock:
                self.error = f"OCR worker failed to start: {payload}"
                if w:
                    self._workers.remove(w)
                # Run with however many did start; none at all → give up.
                self._target = sum(1 for x in self._workers if x.ready)
                if not self._target and not self._workers:
                    self.enabled = False
                    self._queue.clear()
            return
        if w is None or key not in w.inflight:
            return
        path = w.inflight.pop(key)
        lines = payload if kind == "done" else []
        meta = {"lines": lines} if kind == "done" else {"lines": [], "error": str(payload)}
        try:
            self._cache.save_meta(key, CACHE_KIND, meta)
        except OSError:
            pass
        text = "\n".join(str(x[0]) for x in lines)
        with self._lock:
            for p in self._paths_of.get(key, {path}):
                self._text[p], self._lines[p] = text, lines
            self._changed |= self._waiters.pop(key, set())
            if kind == "done":
                self.n_read += 1

    def _reap(self) -> None:
        """Restart crashed workers and give their photos to someone else. If
        they keep dying, stop trying (something's wrong with the GPU setup)."""
        for w in list(self._workers):
            if w.proc.is_alive():
                continue
            with self._lock:
                for sha1, path in w.inflight.items():
                    self._queue[sha1] = path
                    self._queue.move_to_end(sha1, last=False)
                self._workers.remove(w)
            now = time.time()
            self._restarts.append(now)
            while self._restarts and now - self._restarts[0] > 600:
                self._restarts.popleft()
            if len(self._restarts) > 5:
                with self._lock:
                    self.error = "OCR workers keep crashing; text on screen is paused."
                    self.enabled = False
                return
            # _ensure_workers starts the replacement.

    def _notify(self, force: bool = False) -> None:
        if not self._changed or (not force and time.time() - self._last_notify < 2.0):
            return
        ids, self._changed = set(self._changed), set()
        self._last_notify = time.time()
        try:
            self._on_results(ids)
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001 — keep going; a dead thread would stop OCR silently
                with self._lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(5)
        for w in self._workers:
            try:
                w.tasks.put(None)
            except Exception:  # noqa: BLE001
                pass

    def _step(self) -> None:
        self._wake.clear()
        self._take_intake()
        with self._lock:
            work = bool(self._queue) or any(w.inflight for w in self._workers)
        if work and self.enabled:
            self._reap()
            self._ensure_workers()
            self._dispatch()
        self._collect(0.25 if work else 1.0)
        self._notify(force=not work)

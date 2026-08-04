"""Image Filterer — hero-shot selection for event photography.

Ingest a folder of unlabeled event photos, rank them the way the photo team
would, and browse the result: bursts collapsed to one frame, semantic search,
and filters for shot scale, subject, and hero shots.

Entry points::

    python -m image_filterer.train    # train the production ranker (offline, once)
    python -m image_filterer.server   # serve the browser UI

Programmatic use::

    from image_filterer import default_config, ingest_folder
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import Config, default_config
from .ingest import ingest_folder
from .scene import SceneConfig, classify_scene

__all__ = [
    "__version__",
    "Config",
    "default_config",
    "ingest_folder",
    "SceneConfig",
    "classify_scene",
]

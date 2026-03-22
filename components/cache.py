"""Cache for detections and embeddings to avoid recomputation on repeated runs."""

from __future__ import annotations

import os
import pickle
import json
from pathlib import Path
from dataclasses import asdict, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from common.classes.player import Player, PlayersDetections


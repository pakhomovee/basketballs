"""Basketball analysis components: detection, court detection, team clustering, visualization."""

import sys
from pathlib import Path

# Ensure components dir is first in path (avoids shadowing by system 'common' package)
_components_dir = Path(__file__).resolve().parent
if str(_components_dir) not in sys.path:
    sys.path.insert(0, str(_components_dir))

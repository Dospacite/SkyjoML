import hashlib as _hashlib
import os as _os
from pathlib import Path as _Path

__version__ = "0.2.0"

# Numba invalidates an on-disk cache entry only when the function's own source file changes, not when a constant it
# imports from another module does (e.g. MAX_PLAYERS). Key the cache directory by the files that define compile-time
# constants so that such a change can never silently reuse stale machine code.
_here = _Path(__file__).resolve().parent
_h = _hashlib.sha256()
for _f in ("game/rules.py", "game/obs_layout.py", "game/actions.py", "game/rng.py"):
    _h.update((_here / _f).read_bytes())
_os.environ.setdefault("NUMBA_CACHE_DIR", str(_here.parents[1] / ".numba_cache" / _h.hexdigest()[:12]))

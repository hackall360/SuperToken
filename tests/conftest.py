import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency for CI
    from tests._stubs import install_torch_stub

    install_torch_stub()

import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_package_stub() -> None:
    package_name = "gpu_tokenizer"
    if package_name in sys.modules:
        return

    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package


def _install_torch_stub() -> None:
    for name in ["torch", "torch.distributed", "torch.multiprocessing"]:
        sys.modules.pop(name, None)

    torch_stub = types.ModuleType("torch")

    class _Device:
        def __init__(self, type_name: str, index: int | None = None) -> None:
            self.type = type_name
            self.index = index

    class _DeviceContext:
        def __init__(self, cuda_mod: "_CudaStub", index: int) -> None:
            self._cuda = cuda_mod
            self._index = index

        def __enter__(self) -> "_DeviceContext":
            self._cuda.set_device(self._index)
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    class _CudaStub:
        def __init__(self) -> None:
            self._available = False
            self._count = 0
            self._current = 0
            self.enabled_pairs: list[tuple[int, int]] = []

        def is_available(self) -> bool:
            return self._available

        def device_count(self) -> int:
            return self._count

        def current_device(self) -> int:
            return self._current

        def set_device(self, index: int) -> None:
            self._current = int(index)

        def device(self, index: int) -> _DeviceContext:
            return _DeviceContext(self, int(index))

        def device_enable_peer_access(self, peer: int) -> None:
            self.enabled_pairs.append((self._current, int(peer)))

        def device_can_access_peer(self, *_: int) -> bool:
            return True

        def Stream(self, *_: object, **__: object) -> object:
            return object()

        def stream(self, stream_obj: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(__enter__=lambda *args: stream_obj, __exit__=lambda *a: False)

        def memcpy_peer_async(self, *_: object, **__: object) -> None:
            return None

    cuda_stub = _CudaStub()
    torch_stub.cuda = cuda_stub
    torch_stub.device = lambda type_name, index=None: _Device(type_name, index)

    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.is_available = lambda: False
    dist_stub.is_initialized = lambda: False

    mp_stub = types.ModuleType("torch.multiprocessing")
    mp_stub.start_processes = lambda *args, **kwargs: None
    spawn_stub = types.ModuleType("torch.multiprocessing.spawn")
    spawn_stub.ProcessContext = object
    mp_stub.spawn = spawn_stub

    torch_stub.distributed = dist_stub
    torch_stub.multiprocessing = mp_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.multiprocessing"] = mp_stub
    sys.modules["torch.multiprocessing.spawn"] = spawn_stub


def _reload_dist_runtime():
    sys.modules.pop("gpu_tokenizer.dist_runtime", None)
    return importlib.import_module("gpu_tokenizer.dist_runtime")


_install_package_stub()
_install_torch_stub()


def test_enable_peer_access_invokes_peer_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_dist_runtime()
    cuda_stub = module.torch.cuda
    cuda_stub._available = True
    cuda_stub._count = 3
    cuda_stub.enabled_pairs.clear()

    monkeypatch.setattr(module.utils, "can_peer", lambda src, dst: True)

    module._enable_peer_access_for_devices([0, 1, 2], logger=None)

    expected = {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }
    assert set(cuda_stub.enabled_pairs) == expected


def test_enable_peer_access_skips_without_connectivity(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_dist_runtime()
    cuda_stub = module.torch.cuda
    cuda_stub._available = True
    cuda_stub._count = 2
    cuda_stub.enabled_pairs.clear()

    monkeypatch.setattr(module.utils, "can_peer", lambda *_: False)

    module._enable_peer_access_for_devices([0, 1], logger=None)

    assert cuda_stub.enabled_pairs == []

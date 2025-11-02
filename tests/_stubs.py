"""Helper utilities for installing lightweight dependency stubs in tests."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def install_package_stub() -> None:
    """Ensure ``gpu_tokenizer`` is importable without running its ``__init__``."""

    if "gpu_tokenizer" in sys.modules:
        return

    package = types.ModuleType("gpu_tokenizer")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "gpu_tokenizer", loader=None, is_package=True
    )
    sys.modules["gpu_tokenizer"] = package


def install_torch_stub() -> None:
    """Provide a very small ``torch`` stub for CPU-only environments.

    If a real PyTorch installation is available on the system, prefer it over
    the stub to avoid poisoning the test environment for GPU-enabled tests.
    """

    if "torch" in sys.modules:
        try:
            Path('.artifacts').mkdir(exist_ok=True)
            (Path('.artifacts')/ 'stub_log.txt').write_text('torch present in sys.modules\n', encoding='utf-8')
        except Exception:
            pass
        return
    # Prefer importing the real library if it is present and importable.
    try:  # pragma: no cover - exercised across environments
        import torch as _real_torch  # type: ignore
        # If the import succeeded, do not install the stub
        if hasattr(_real_torch, "__version__"):
            try:
                Path('.artifacts').mkdir(exist_ok=True)
                (Path('.artifacts')/ 'stub_log.txt').write_text('import torch ok\n', encoding='utf-8')
            except Exception:
                pass
            return
    except Exception:
        # If import fails, try a lightweight spec probe; otherwise fall back to stub
        try:
            import importlib.util as _ilus
            if _ilus.find_spec("torch") is not None:  # type: ignore[attr-defined]
                try:
                    Path('.artifacts').mkdir(exist_ok=True)
                    (Path('.artifacts')/ 'stub_log.txt').write_text('find_spec torch ok\n', encoding='utf-8')
                except Exception:
                    pass
                return
        except Exception:
            pass

    torch_stub = types.ModuleType("torch")

    class _DType(str):
        pass

    torch_stub.uint16 = _DType("uint16")
    torch_stub.int16 = _DType("int16")
    torch_stub.int32 = _DType("int32")
    torch_stub.int64 = _DType("int64")
    torch_stub.int8 = _DType("int8")
    torch_stub.uint8 = _DType("uint8")

    def _iinfo(dtype: _DType) -> types.SimpleNamespace:
        max_val = (1 << 16) - 1 if dtype == torch_stub.uint16 else (1 << 31) - 1
        return types.SimpleNamespace(max=max_val)

    torch_stub.iinfo = _iinfo

    class _Device:
        def __init__(self, spec: str, index: int | None = None) -> None:
            self.type = spec
            self.index = index

        def __str__(self) -> str:  # pragma: no cover - debug helper
            return f"{self.type}:{self.index}" if self.index is not None else self.type

    torch_stub.device = lambda spec, index=None: _Device(str(spec), index)

    class _Tensor:
        def __init__(self, *_, device: _Device | None = None, **__):
            self._device = device or _Device("cpu")

        @property
        def device(self) -> _Device:
            return self._device

        def to(self, *, device: _Device | None = None, **__) -> "_Tensor":
            if device is not None:
                self._device = device
            return self

        def contiguous(self) -> "_Tensor":
            return self

        def unfold(self, *_, **__) -> "_Tensor":
            return self

        def numel(self) -> int:
            return 0

        def __bool__(self) -> bool:  # pragma: no cover - defensive
            return False

    torch_stub.Tensor = _Tensor

    def _make_tensor(*_, **__) -> _Tensor:
        return _Tensor()

    for name in ["empty", "arange", "bitwise_left_shift", "sum", "ones", "cat"]:
        setattr(torch_stub, name, _make_tensor)

    torch_stub.bool = _DType("bool")
    torch_stub.long = _DType("long")

    class _Cuda:
        def is_available(self) -> bool:
            return False

        def device_count(self) -> int:
            return 0

        def set_device(self, *_: object, **__: object) -> None:
            return None

        def device(self, index: int) -> types.SimpleNamespace:
            return types.SimpleNamespace(__enter__=lambda: self, __exit__=lambda *args: False)

        def stream(self, *_: object, **__: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: False)

        def Event(self, *_: object, **__: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(record=lambda *args, **kwargs: None, wait=lambda: None)

        def mem_get_info(self, *_: object, **__: object) -> tuple[int, int]:
            return (0, 1)

        def device_can_access_peer(self, *_: object, **__: object) -> bool:
            return False

    torch_stub.cuda = _Cuda()

    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.is_available = lambda: False
    dist_stub.is_initialized = lambda: False
    dist_stub.gather_object = lambda *args, **kwargs: None
    dist_stub.broadcast_object_list = lambda *args, **kwargs: None
    dist_stub.init_process_group = lambda **kwargs: None
    dist_stub.destroy_process_group = lambda: None

    utils_stub = types.ModuleType("torch.utils")
    cpp_stub = types.ModuleType("torch.utils.cpp_extension")
    cpp_stub.load_inline = lambda *args, **kwargs: None
    utils_stub.cpp_extension = cpp_stub

    torch_stub.jit = types.SimpleNamespace(script=lambda fn: fn)
    torch_stub.__super_token_stub__ = True

    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.cpp_extension"] = cpp_stub
    sys.modules.setdefault("torch.multiprocessing", types.ModuleType("torch.multiprocessing"))
    try:
        Path('.artifacts').mkdir(exist_ok=True)
        (Path('.artifacts')/ 'stub_log.txt').write_text('installed stub torch\n', encoding='utf-8')
    except Exception:
        pass


__all__ = ["install_package_stub", "install_torch_stub"]

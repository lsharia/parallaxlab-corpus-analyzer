"""Verify the Python environment required by the RAG project."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


REQUIRED_DEPENDENCIES = (
    ("pandas", "pandas"),
    ("sentence_transformers", "sentence-transformers"),
    ("chromadb", "chromadb"),
    ("spacy", "spacy"),
)


def check_import(module_name: str, display_name: str) -> bool:
    """Import one dependency and print a readable PASS or FAIL result."""
    try:
        importlib.import_module(module_name)
    except Exception as error:
        print(f"[FAIL] {display_name}: {type(error).__name__}: {error}")
        return False

    print(f"[PASS] {display_name}")
    return True


def check_torch_and_cuda() -> None:
    """Report PyTorch and CUDA availability without making GPU support mandatory."""
    try:
        torch: ModuleType = importlib.import_module("torch")
    except Exception as error:
        print(
            f"[INFO] PyTorch: unavailable ({type(error).__name__}: {error}); "
            "CUDA check skipped"
        )
        return

    torch_version = getattr(torch, "__version__", "unknown")
    print(f"[PASS] PyTorch: installed (version {torch_version})")

    if not torch.cuda.is_available():
        print("[INFO] CUDA: unavailable; the project can run on CPU")
        return

    cuda_version = getattr(torch.version, "cuda", "unknown")
    print(f"[PASS] CUDA: available (version {cuda_version})")
    print(f"[PASS] GPU: {torch.cuda.get_device_name(0)}")


def main() -> int:
    """Run all checks and return non-zero only when a required import fails."""
    print(f"Python: {sys.version}")
    print("\nPyTorch and hardware checks:")
    check_torch_and_cuda()

    print("\nRequired dependency imports:")
    dependency_results = [
        check_import(module_name, display_name)
        for module_name, display_name in REQUIRED_DEPENDENCIES
    ]

    if all(dependency_results):
        print("\nEnvironment verification PASSED")
        return 0

    print("\nEnvironment verification FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

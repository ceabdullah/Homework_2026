"""Preflight check. Run this FIRST, before downloading 10 GB of anything.

    python scripts/check_env.py

Reports what is ready and what is missing, with the exact fix for each.
Exit code 0 means you can start Phase 0.
"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, BAD = "[ ok ]", "[warn]", "[FAIL]"
problems: list[str] = []
warnings: list[str] = []


def fail(msg: str, fix: str) -> None:
    print(f"{BAD} {msg}")
    problems.append(f"{msg}\n         fix: {fix}")


def warn(msg: str, fix: str) -> None:
    print(f"{WARN} {msg}")
    warnings.append(f"{msg}\n         fix: {fix}")


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        fail(f"Python {v.major}.{v.minor} is too old",
             "install Python 3.11 (3.10+ required; the code uses `str | None` syntax)")
    else:
        print(f"{OK} Python {v.major}.{v.minor}.{v.micro}")


def check_packages() -> None:
    required = ["torch", "torchvision", "timm", "albumentations", "numpy", "pandas",
                "sklearn", "cv2", "PIL", "matplotlib", "seaborn", "yaml", "tqdm"]
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"missing packages: {', '.join(missing)}",
             "pip install -r requirements.txt")
    else:
        print(f"{OK} all {len(required)} required packages import")


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        return  # already reported
    print(f"{OK} torch {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{OK} CUDA available: {name} ({vram:.1f} GB)")
        if vram < 8:
            warn(f"only {vram:.1f} GB VRAM",
                 "use --batch_size 16; do not lower image_size, resolution is not "
                 "the interesting variable")
    else:
        fail("CUDA not available -- training will run on CPU and take weeks",
             "pip install torch==2.5.1 torchvision==0.20.1 "
             "--index-url https://download.pytorch.org/whl/cu124")


def check_pretrained_weights() -> None:
    """timm fetches ImageNet weights from the HF hub on first use. Failing
    here at setup beats failing 40 minutes into a run."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")
    print("       ... fetching a small test model, first run may take a minute")
    try:
        import timm
        timm.create_model("resnet18", pretrained=True, num_classes=2)
        print(f"{OK} pretrained weights reachable (timm/HF hub)")
    except ImportError:
        return
    except Exception as e:  # noqa: BLE001 - any network/hub error is the same story
        fail(f"cannot fetch pretrained weights: {type(e).__name__}: {str(e)[:120]}",
             "check internet/proxy access to huggingface.co, or run with "
             "--no-pretrained and say so in the write-up (it changes the numbers)")


def check_kaggle() -> None:
    token = Path.home() / ".kaggle" / "kaggle.json"
    env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if token.exists() or env:
        print(f"{OK} Kaggle credentials found")
    else:
        warn("no Kaggle credentials (needed for HAM10000)",
             f"Kaggle -> Settings -> Create New Token, save to {token}")
    if shutil.which("kaggle") is None:
        warn("`kaggle` CLI not on PATH", "pip install kaggle==1.6.17")


def check_disk() -> None:
    free = shutil.disk_usage(Path.cwd()).free / 1e9
    if free < 15:
        fail(f"only {free:.1f} GB free on this drive",
             "free up space; the two datasets need ~10 GB plus room for results/")
    else:
        print(f"{OK} {free:.1f} GB free disk")


def check_git() -> None:
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        print(f"{OK} git branch: {branch}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn("not a git checkout", "clone the repo rather than copying files")


def check_layout() -> None:
    missing = [p for p in ("src/train.py", "configs/phase1_balanced.yaml",
                           "requirements.txt") if not Path(p).exists()]
    if missing:
        fail(f"missing project files: {missing}",
             "run this from the repository root, not from scripts/")
    else:
        print(f"{OK} project layout looks right (cwd={Path.cwd()})")


def main() -> int:
    print("=" * 68)
    print("Preflight check -- cross-domain skin lesion classification")
    print("=" * 68)
    for fn in (check_layout, check_python, check_packages, check_torch_cuda,
               check_pretrained_weights, check_kaggle, check_disk, check_git):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fail(f"{fn.__name__} crashed: {type(e).__name__}: {e}", "report this")
    print("=" * 68)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {WARN} {w}")
    if problems:
        print(f"\n{len(problems)} blocking problem(s):")
        for p in problems:
            print(f"  {BAD} {p}")
        print("\nFix these, then re-run this script.")
        return 1

    print("\nReady. Next:")
    print("  python -m src.data.download")
    print("  python -m src.data.build_splits")
    print("  bash scripts/run_phase1.sh      (Windows: .\\scripts\\run_phase1.ps1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

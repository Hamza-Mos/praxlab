"""
Run autoresearch training on a cloud GPU via Modal.

One-time setup (downloads data + trains tokenizer on Modal):
    modal run modal_run.py --setup

Run training (drop-in replacement for `uv run train.py`):
    modal run modal_run.py

Select GPU (default H100):
    AUTORESEARCH_GPU=A100-80GB modal run modal_run.py
"""

import os
import sys

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

GPU_TYPE = os.environ.get("AUTORESEARCH_GPU", "H100")
VOLUME_NAME = "autoresearch-data"

app = modal.App("autoresearch")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.9.1",
        "kernels>=0.11.7",
        "tiktoken>=0.11.0",
        "rustbpe>=0.1.0",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "numpy>=2.2.6",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={"/root/.cache/autoresearch": volume},
    timeout=1800,
)
def setup_data(prepare_py: str):
    """Download training data and train tokenizer on Modal."""
    import subprocess

    with open("/tmp/prepare.py", "w") as f:
        f.write(prepare_py)

    proc = subprocess.Popen(
        ["python", "/tmp/prepare.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    volume.commit()


@app.function(
    gpu=GPU_TYPE,
    image=image,
    volumes={"/root/.cache/autoresearch": volume},
    timeout=720,
)
def run_training(train_py: str, prepare_py: str) -> int:
    """Execute train.py on a cloud GPU, streaming output back."""
    import subprocess

    workdir = "/tmp/autoresearch"
    os.makedirs(workdir, exist_ok=True)
    for name, src in [("train.py", train_py), ("prepare.py", prepare_py)]:
        with open(os.path.join(workdir, name), "w") as f:
            f.write(src)

    proc = subprocess.Popen(
        ["python", "train.py"],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(setup: bool = False):
    if setup:
        prepare_py = open("prepare.py").read()
        setup_data.remote(prepare_py)
    else:
        train_py = open("train.py").read()
        prepare_py = open("prepare.py").read()
        returncode = run_training.remote(train_py, prepare_py)
        if returncode != 0:
            sys.exit(returncode)

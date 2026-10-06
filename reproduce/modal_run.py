"""Run reproduce/run.py on Modal: the same single script, in one GPU container.

    modal run reproduce/modal_run.py                        (every stage, on an H100)
    modal run reproduce/modal_run.py --only grid --gpu A100

The run directory and the Hugging Face cache live in a Modal volume, named by the environment
variable ATTENTION_RELAY_VOLUME (default attention-relay). The gated Llama 3.1 checkpoints need a
Modal secret named huggingface that holds an HF_TOKEN. The image pins the software of the paper's
GPU runs (torch 2.8.0, transformers 5.17.0) and the NumPy and scikit-learn versions the
reproduction was checked with (2.5.3, 1.9.1), since k-means' results can change with them.
"""

import os
import pathlib

import modal

# the repository on this machine; inside the container the image already holds it
REPOSITORY = pathlib.Path(__file__).resolve().parents[1] if modal.is_local() else None
VOLUME = os.environ.get("ATTENTION_RELAY_VOLUME", "attention-relay")

app = modal.App("attention-relay-reproduce")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.17.0",
        "numpy==2.5.3",  # k-means' results can change with the versions of NumPy and scikit-learn
        "datasets",
        "scikit-learn==1.9.1",
        "scipy",
        "threadpoolctl",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
)
if REPOSITORY is not None:
    image = image.add_local_dir(REPOSITORY / "src", remote_path="/root/repository/src")
    image = image.add_local_dir(REPOSITORY / "reproduce", remote_path="/root/repository/reproduce")


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=131072,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=24 * 3600,
)
def reproduce(arguments: list[str]) -> None:
    import subprocess
    import sys
    import threading

    stopped = threading.Event()

    def commit_every_ten_minutes():
        while not stopped.wait(600):
            volume.commit()

    threading.Thread(target=commit_every_ten_minutes, daemon=True).start()
    environment = {**os.environ, "PYTHONPATH": "/root/repository/src:/root/repository"}
    try:
        subprocess.run(
            [sys.executable, "-m", "reproduce.run", *arguments],
            cwd="/root/repository",
            env=environment,
            check=True,
        )
    finally:
        stopped.set()
        volume.commit()


@app.local_entrypoint()
def main(out: str = "/vol/runs/paper", only: str = "", limit: int = 0, gpu: str = "H100"):
    arguments = ["--out", out, "--device", "cuda"]
    if only:
        arguments += ["--only", only]
    if limit:
        arguments += ["--limit", str(limit)]
    reproduce.with_options(gpu=gpu).remote(arguments)

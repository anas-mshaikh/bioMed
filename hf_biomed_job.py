# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=1.0.0",
#   "pip",
#   "setuptools",
#   "wheel",
# ]
# ///

"""
HF Jobs bootstrap runner for BioMed training.

Purpose:
- Download a private local-source tarball from a HF dataset repo.
- Extract it on the HF Job machine.
- Install project dependencies.
- Run your existing local training command.
- Upload training outputs back to a HF model/dataset repo.

This lets you train local project files on HF Jobs without GitHub.

Example:
  hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN hf_biomed_job.py \
    --src-repo YOUR_USER/biomed-train-src \
    --src-filename biomed_src.tgz \
    --result-repo YOUR_USER/biomed-training-runs \
    --result-repo-type dataset \
    --run-name smoke_10 \
    --requirements requirements-hf-train.txt \
    -- \
    python training/training_unsloth.py \
      --model-id Qwen/Qwen3-0.6B \
      --training-mode single_action_curriculum \
      --dataset-episodes 8 \
      --rollout-steps 2 \
      --trainer-max-steps 10 \
      --num-generations 2 \
      --output-dir outputs/hf_smoke_10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi, hf_hub_download


def log(message: str) -> None:
    print(f"[hf-biomed-job] {message}", flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    pretty = " ".join(str(x) for x in cmd)
    log(f"RUN: {pretty}")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """
    Safe-ish tar extraction: blocks path traversal.
    """
    extract_dir = extract_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if not str(target).startswith(str(extract_dir)):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(extract_dir)


def find_project_root(extract_dir: Path, preferred_subdir: str | None) -> Path:
    if preferred_subdir:
        candidate = extract_dir / preferred_subdir
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"--project-subdir not found: {candidate}")

    # If archive extracted directly with pyproject/training folder.
    if (extract_dir / "pyproject.toml").exists() or (extract_dir / "training").exists():
        return extract_dir.resolve()

    # If tar preserved one root folder.
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    for child in children:
        if (child / "pyproject.toml").exists() or (child / "training").exists():
            return child.resolve()

    # Fall back to extract dir.
    return extract_dir.resolve()


def install_dependencies(
    project_root: Path,
    *,
    requirements: str | None,
    install_editable: bool,
    extra_pip: list[str],
) -> None:
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if requirements:
        req_path = project_root / requirements
        if not req_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {req_path}")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_path)])

    if extra_pip:
        run_cmd([sys.executable, "-m", "pip", "install", *extra_pip])

    if install_editable:
        if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
            run_cmd([sys.executable, "-m", "pip", "install", "-e", str(project_root)])
        else:
            log("No pyproject.toml/setup.py found; skipping editable install.")


def create_repo_if_needed(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )


def upload_outputs(
    *,
    api: HfApi,
    project_root: Path,
    output_dir: str,
    result_repo: str,
    result_repo_type: str,
    run_name: str,
    private_results: bool,
) -> None:
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    if not out_path.exists():
        log(f"Output directory not found, skipping upload: {out_path}")
        return

    create_repo_if_needed(
        api=api,
        repo_id=result_repo,
        repo_type=result_repo_type,
        private=private_results,
    )

    log(f"Uploading outputs from {out_path} to {result_repo_type}:{result_repo}/{run_name}")
    api.upload_folder(
        folder_path=str(out_path),
        repo_id=result_repo,
        repo_type=result_repo_type,
        path_in_repo=run_name,
        commit_message=f"Upload BioMed training outputs: {run_name}",
    )


def write_run_metadata(project_root: Path, run_name: str, command: list[str]) -> None:
    meta_dir = project_root / "outputs" / "hf_job_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / f"{run_name}.txt"

    lines = [
        f"run_name={run_name}",
        f"timestamp={int(time.time())}",
        f"cwd={project_root}",
        f"python={sys.version}",
        "command=" + " ".join(command),
        "",
        "environment:",
    ]

    for key in sorted(os.environ):
        if key.upper().endswith("TOKEN") or "SECRET" in key.upper():
            continue
        if key.startswith(("HF_", "CUDA", "ACCELERATE", "TRANSFORMERS", "TRL", "WANDB")):
            lines.append(f"{key}={os.environ.get(key)}")

    meta_file.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote metadata: {meta_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run local BioMed training source on Hugging Face Jobs.")

    parser.add_argument(
        "--src-repo",
        required=True,
        help="HF dataset repo containing source tarball, e.g. user/biomed-train-src",
    )
    parser.add_argument(
        "--src-repo-type",
        default="dataset",
        choices=["dataset", "model"],
        help="Repo type for source bundle.",
    )
    parser.add_argument(
        "--src-filename", default="biomed_src.tgz", help="Filename inside source repo."
    )
    parser.add_argument("--src-revision", default="main", help="Source repo revision.")

    parser.add_argument(
        "--project-subdir", default=None, help="Optional subdir inside extracted tarball."
    )
    parser.add_argument("--workdir", default="/tmp/biomed_job_work", help="Remote work directory.")

    parser.add_argument(
        "--requirements",
        default=None,
        help="Requirements file inside project root, e.g. requirements-hf-train.txt",
    )
    parser.add_argument(
        "--install-editable",
        action="store_true",
        default=True,
        help="pip install -e project root if pyproject/setup exists.",
    )
    parser.add_argument("--no-install-editable", action="store_false", dest="install_editable")
    parser.add_argument(
        "--extra-pip",
        action="append",
        default=[],
        help="Extra pip packages. Can be repeated, e.g. --extra-pip trl --extra-pip bitsandbytes",
    )

    parser.add_argument("--result-repo", required=True, help="HF repo to upload outputs to.")
    parser.add_argument(
        "--result-repo-type",
        default="dataset",
        choices=["dataset", "model"],
        help="Repo type for outputs.",
    )
    parser.add_argument("--run-name", required=True, help="Subfolder name in result repo.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output dir to upload. If omitted, tries to infer from command.",
    )
    parser.add_argument("--private-results", action="store_true", default=True)
    parser.add_argument("--public-results", action="store_false", dest="private_results")

    parser.add_argument("--skip-install", action="store_true", help="Skip pip installation.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip output upload.")

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Training command after --, e.g. -- python training/training_unsloth.py ...",
    )

    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        raise SystemExit("No training command provided. Add it after --")

    return args


def infer_output_dir(command: list[str]) -> str | None:
    for i, item in enumerate(command):
        if item == "--output-dir" and i + 1 < len(command):
            return command[i + 1]
        if item.startswith("--output-dir="):
            return item.split("=", 1)[1]
    return None


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("WARNING: HF_TOKEN not found. Private repo download/upload may fail.")

    api = HfApi(token=hf_token)

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    extract_dir = workdir / "src"
    extract_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading source bundle: {args.src_repo}/{args.src_filename}")
    bundle_path = hf_hub_download(
        repo_id=args.src_repo,
        repo_type=args.src_repo_type,
        filename=args.src_filename,
        revision=args.src_revision,
        token=hf_token,
    )

    log(f"Extracting source bundle: {bundle_path}")
    safe_extract_tar(Path(bundle_path), extract_dir)

    project_root = find_project_root(extract_dir, args.project_subdir)
    log(f"Project root: {project_root}")

    if not args.skip_install:
        install_dependencies(
            project_root=project_root,
            requirements=args.requirements,
            install_editable=args.install_editable,
            extra_pip=args.extra_pip,
        )
    else:
        log("Skipping dependency install.")

    write_run_metadata(project_root, args.run_name, args.command)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "1")

    exit_code = 0
    try:
        result = run_cmd(args.command, cwd=project_root, env=env, check=False)
        exit_code = int(result.returncode)
        log(f"Training command exited with code {exit_code}")
    finally:
        if not args.skip_upload:
            output_dir = args.output_dir or infer_output_dir(args.command)
            if output_dir:
                try:
                    upload_outputs(
                        api=api,
                        project_root=project_root,
                        output_dir=output_dir,
                        result_repo=args.result_repo,
                        result_repo_type=args.result_repo_type,
                        run_name=args.run_name,
                        private_results=args.private_results,
                    )
                except Exception as exc:
                    log(f"Output upload failed: {exc!r}")
                    if exit_code == 0:
                        exit_code = 2
            else:
                log("No output dir provided/inferred; skipping output upload.")
        else:
            log("Skipping output upload.")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

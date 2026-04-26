# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=1.0.0",
#   "pip",
#   "setuptools",
#   "wheel",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi


def log(message: str) -> None:
    print(f"[hf-github-job] {message}", flush=True)


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    secrets: list[str] | None = None,
) -> subprocess.CompletedProcess:
    secrets = secrets or []
    pretty = " ".join(str(x) for x in cmd)
    log(f"RUN: {redact(pretty, secrets)}")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def python_has_pip() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def ensure_pip() -> None:
    if python_has_pip():
        return

    log("pip not found. Trying ensurepip...")
    result = subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], text=True)

    if result.returncode != 0 or not python_has_pip():
        raise RuntimeError(
            "pip is not available. Keep pip/setuptools/wheel in the UV dependency block."
        )


def pip_install(args: list[str]) -> None:
    ensure_pip()
    run_cmd([sys.executable, "-m", "pip", "install", *args])


def install_dependencies(
    project_root: Path,
    *,
    requirements: str | None,
    install_editable: bool,
    extra_pip: list[str],
) -> None:
    pip_install(["--upgrade", "pip", "setuptools", "wheel"])

    if requirements:
        req_path = project_root / requirements
        if not req_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {req_path}")
        pip_install(["-r", str(req_path)])

    if extra_pip:
        pip_install(extra_pip)

    if install_editable:
        if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
            pip_install(["-e", str(project_root)])
        else:
            log("No pyproject.toml/setup.py found. Skipping editable install.")


def github_auth_header(token: str) -> str:
    # GitHub git-over-HTTPS works reliably with basic auth:
    # username = x-access-token, password = token
    raw = f"x-access-token:{token}".encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"AUTHORIZATION: basic {encoded}"


def clone_github_repo(
    *,
    github_repo: str,
    branch: str,
    commit: str | None,
    destination: Path,
    token: str | None,
) -> None:
    clone_url = f"https://github.com/{github_repo}.git"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    cmd = ["git"]

    secrets = []
    if token:
        header = github_auth_header(token)
        cmd += ["-c", f"http.https://github.com/.extraheader={header}"]
        secrets.append(header)
        secrets.append(token)

    cmd += [
        "clone",
        "--depth",
        "1",
        "--branch",
        branch,
        clone_url,
        str(destination),
    ]

    run_cmd(cmd, secrets=secrets)

    if commit:
        run_cmd(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
        run_cmd(["git", "checkout", commit], cwd=destination)


def create_repo_if_needed(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
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
        log(f"Output directory not found. Skipping upload: {out_path}")
        return

    create_repo_if_needed(api, result_repo, result_repo_type, private_results)

    log(f"Uploading {out_path} to {result_repo_type}:{result_repo}/{run_name}")
    api.upload_folder(
        folder_path=str(out_path),
        repo_id=result_repo,
        repo_type=result_repo_type,
        path_in_repo=run_name,
        commit_message=f"Upload BioMed training outputs: {run_name}",
    )


def infer_output_dir(command: list[str]) -> str | None:
    for i, item in enumerate(command):
        if item == "--output-dir" and i + 1 < len(command):
            return command[i + 1]
        if item.startswith("--output-dir="):
            return item.split("=", 1)[1]
    return None


def write_metadata(project_root: Path, run_name: str, command: list[str]) -> None:
    meta_dir = project_root / "outputs" / "hf_job_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_file = meta_dir / f"{run_name}.txt"
    meta_file.write_text(
        "\n".join(
            [
                f"run_name={run_name}",
                f"timestamp={int(time.time())}",
                f"project_root={project_root}",
                f"python={sys.version}",
                "command=" + " ".join(command),
            ]
        ),
        encoding="utf-8",
    )
    log(f"Wrote metadata: {meta_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Clone GitHub repo and run BioMed training on HF Jobs.")

    parser.add_argument(
        "--github-repo", required=True, help="GitHub repo as owner/name, e.g. theRake/bioMed"
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--commit", default=None, help="Optional exact commit SHA for reproducible runs."
    )

    parser.add_argument("--workdir", default="/tmp/biomed_github_job")
    parser.add_argument("--requirements", default="requirements-hf-train.txt")
    parser.add_argument("--install-editable", action="store_true", default=True)
    parser.add_argument("--no-install-editable", action="store_false", dest="install_editable")
    parser.add_argument("--extra-pip", action="append", default=[])

    parser.add_argument("--result-repo", required=True, help="HF repo to upload outputs to.")
    parser.add_argument("--result-repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--private-results", action="store_true", default=True)
    parser.add_argument("--public-results", action="store_false", dest="private_results")

    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")

    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        raise SystemExit("No training command provided. Add it after --")

    return args


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    github_token = os.environ.get("GITHUB_TOKEN")

    api = HfApi(token=hf_token)

    workdir = Path(args.workdir)
    project_root = workdir / "repo"

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    log(f"Cloning GitHub repo: {args.github_repo} branch={args.branch}")
    clone_github_repo(
        github_repo=args.github_repo,
        branch=args.branch,
        commit=args.commit,
        destination=project_root,
        token=github_token,
    )

    if not args.skip_install:
        install_dependencies(
            project_root=project_root,
            requirements=args.requirements,
            install_editable=args.install_editable,
            extra_pip=args.extra_pip,
        )

    write_metadata(project_root, args.run_name, args.command)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")

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
                    log(f"Upload failed: {exc!r}")
                    if exit_code == 0:
                        exit_code = 2
            else:
                log("No output dir provided/inferred. Skipping upload.")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

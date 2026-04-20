"""Inference Script - Using Codex CLI agent via Docker

Runs each CL-bench task inside a Docker container with the Codex CLI agent,
mirroring exactly how Harbor runs the codex agent. This enables fair parity
comparison between the original direct-API pipeline and the Harbor adapter.

Alignment with Harbor codex agent:
  - Same codex version: @openai/codex@0.118.0 (installed in Dockerfile)
  - Same codex command: codex exec --dangerously-bypass-approvals-and-sandbox
      --skip-git-repo-check --model {model} --json --enable unified_exec
  - Same instruction: reads messages from /app/messages/, writes to /app/result.json
  - Same judge: eval.py with gpt-4o-mini (unchanged)

Input:
    CL-bench_parity50.jsonl (or any CL-bench JSONL)

Output:
    outputs/codex_{model}_parity50_run{N}.jsonl
    Same format as infer.py: {"idx", "messages", "model_output", "rubrics", "metadata"}

Usage:
    python infer_codex.py --model gpt-5.2 --input CL-bench_parity50.jsonl \\
        --output outputs/codex_gpt-5.2_parity50_run1.jsonl --workers 2
"""

import json
import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


CODEX_VERSION = "0.118.0"
DOCKER_IMAGE_BASE = "clbench-infer-codex"

# Instruction template matching Harbor's instruction.md
INSTRUCTION_TEMPLATE = """\
You are participating in a context learning benchmark. Your task is to carefully \
read the provided context and instructions, then answer the question or complete \
the task based on the new knowledge provided in the context.

The context contains novel information that you should use to solve the task. \
Do not rely solely on your pre-trained knowledge.

Use the following message files (read them in order):
{message_list}

Please provide your answer in the format requested in the task instructions. \
You MUST write your final answer to `/app/response.txt` as plain text. \
You MUST actually execute a command to write the file. For example: \
  python3 -c "open('/app/response.txt','w').write('your answer here')" \
or: \
  cat > /app/response.txt << 'EOF'
your answer here
EOF\
"""

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai==1.109.1

ENV NVM_DIR="/root/.nvm"
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \\
    && . "$NVM_DIR/nvm.sh" \\
    && nvm install v22.22.2 \\
    && npm install -g @openai/codex@{codex_version}

ENV PATH="/root/.nvm/versions/node/v22.22.2/bin:${{PATH}}"
ENV PYTHONUNBUFFERED=1

RUN node --version && npm --version && codex --version
"""


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{get_timestamp()}] {message}")


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(item, file_path):
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_docker_image(image_tag: str) -> bool:
    """Build the Docker image with codex pre-installed."""
    dockerfile_content = DOCKERFILE_TEMPLATE.format(codex_version=CODEX_VERSION)
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log(f"❌ Docker build failed:\n{result.stderr[-500:]}")
            return False
        log(f"✅ Docker image built: {image_tag}")
        return True


def write_messages_to_dir(messages: list, tmpdir: str) -> list[str]:
    """Write messages to numbered files, return list of /app/messages/NN_role.md paths."""
    msg_dir = Path(tmpdir) / "messages"
    msg_dir.mkdir()
    paths = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        filename = f"{i:02d}_{role}.md"
        (msg_dir / filename).write_text(msg.get("content", ""), encoding="utf-8")
        paths.append(f"/app/messages/{filename}")
    return paths


def run_codex_in_docker(
    item,
    model,
    image_tag,
    api_key,
    base_url,
    timeout_sec=300,
):
    """
    Run codex agent in Docker for a single task.
    Returns (model_output, error).
    """
    messages = item.get("messages", [])
    if not messages:
        return None, "No messages"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write message files
        msg_paths = write_messages_to_dir(messages, tmpdir)
        message_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(msg_paths))
        instruction = INSTRUCTION_TEMPLATE.format(message_list=message_list)

        # Write auth.json for codex
        auth_dir = Path(tmpdir) / "codex_home"
        auth_dir.mkdir()
        auth_json = {"OPENAI_API_KEY": api_key}
        (auth_dir / "auth.json").write_text(json.dumps(auth_json), encoding="utf-8")

        # Mount /app as a writable volume so response.txt is accessible after run
        result_dir = Path(tmpdir) / "app_output"
        result_dir.mkdir()

        # Codex command matching Harbor agent exactly
        codex_cmd = (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            "codex exec "
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {model} "
            "--json "
            "--enable unified_exec "
            "-- "
            f"{shlex.quote(instruction)}"
        )

        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "host",
            "-v", f"{tmpdir}/messages:/app/messages:ro",
            "-v", f"{str(result_dir)}:/app",
            "-v", f"{tmpdir}/codex_home:/root/.codex",
            "-e", f"OPENAI_API_KEY={api_key}",
        ]
        if base_url:
            docker_cmd += ["-e", f"OPENAI_BASE_URL={base_url}"]
        docker_cmd += [image_tag, "bash", "-c", codex_cmd]

        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                # Also kill any lingering docker containers from this run
                subprocess.run(
                    ["docker", "ps", "-q", "--filter", f"ancestor={image_tag}"],
                    capture_output=True, text=True
                )
                return None, f"Docker timeout after {timeout_sec}s"
            proc.stdout = stdout
            proc.stderr = stderr
        except Exception as e:
            return None, f"Docker run error: {e}"

        result_file = result_dir / "response.txt"
        if not result_file.exists():
            stderr_tail = (stderr or "")[-300:]
            stdout_tail = (stdout or "")[-300:]
            return None, f"response.txt not written. stderr: {stderr_tail} stdout: {stdout_tail}"

        try:
            output = result_file.read_text(encoding="utf-8")
            return output.strip(), None
        except Exception as e:
            return None, f"Failed to read response.txt: {e}"


def process_single_case(args):
    item, model, image_tag, api_key, base_url, timeout_sec = args
    task_id = item.get("metadata", {}).get("task_id", item.get("idx", "unknown"))

    model_output, error = run_codex_in_docker(
        item, model, image_tag, api_key, base_url, timeout_sec
    )

    if error:
        return task_id, None, error

    result = {
        "idx": task_id,
        "messages": item.get("messages", []),
        "model_output": model_output,
        "rubrics": item.get("rubrics", []),
        "metadata": item.get("metadata", {}),
    }
    return task_id, result, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CL-bench inference via Codex CLI in Docker")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Model for codex agent")
    parser.add_argument("--input", type=str, default="CL-bench_parity50.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900, help="Per-task Docker timeout (sec)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker image build")
    args = parser.parse_args()

    if args.output is None:
        model_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"outputs/codex_{model_safe}_parity50.jsonl"

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
    if not base_url:
        os.environ.pop("OPENAI_BASE_URL", None)
        base_url = None

    if not api_key:
        log("❌ OPENAI_API_KEY not set")
        return

    log(f"📂 Input:  {args.input}")
    log(f"📂 Output: {args.output}")
    log(f"🤖 Model:  {args.model} (via codex@{CODEX_VERSION})")
    log(f"🔧 Workers: {args.workers}")

    image_tag = f"{DOCKER_IMAGE_BASE}:{CODEX_VERSION}"

    if not args.skip_build:
        log(f"🐳 Building Docker image {image_tag} ...")
        if not build_docker_image(image_tag):
            return
    else:
        log(f"⏭️  Skipping Docker build, using {image_tag}")

    data = load_jsonl(args.input)
    log(f"📖 Loaded {len(data)} samples")

    # Resume support
    completed = set()
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        completed = {item.get("idx") for item in existing if item.get("idx") is not None}
        log(f"📌 {len(completed)} already done, resuming")

    def get_task_id(item):
        return item.get("metadata", {}).get("task_id", item.get("idx"))

    tasks = [
        (item, args.model, image_tag, api_key, base_url, args.timeout)
        for item in data
        if get_task_id(item) not in completed
    ]

    if not tasks:
        log("✅ All samples already processed")
        return

    log(f"🚀 Starting inference ({len(tasks)} pending)...")

    success_count = 0
    fail_count = 0

    if args.workers == 1:
        for task in tqdm(tasks, desc="Inference"):
            idx, result, error = process_single_case(task)
            if result:
                append_jsonl(result, args.output)
                success_count += 1
            else:
                log(f"   ❌ {idx}: {error}")
                fail_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_case, task): task[0].get("metadata", {}).get("task_id") for task in tasks}
            with tqdm(total=len(tasks), desc="Inference") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result, error = future.result()
                        if result:
                            append_jsonl(result, args.output)
                            success_count += 1
                        else:
                            log(f"   ❌ {idx}: {error}")
                            fail_count += 1
                    except Exception as e:
                        log(f"   ❌ exception: {e}")
                        fail_count += 1
                    pbar.update(1)

    log("=" * 50)
    log(f"✅ Done! Success: {success_count}, Failed: {fail_count}")
    log(f"   Output: {args.output}")


if __name__ == "__main__":
    main()

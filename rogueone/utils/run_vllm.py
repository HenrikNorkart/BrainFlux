import shlex
from pathlib import Path
import subprocess
import os
import signal
from typing import Optional
from time import sleep, time

from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

_EMBEDDING_SERVER_PID = None  # Stores the process ID of the embedding server


def start_daemon(
    cmd: str,
    pidfile: Optional[str] = None,
    logfile: Optional[str] = None,
    cwd: Optional[str] = None,
) -> int:
    """
    Start a command as a detached daemon, optionally writing a pidfile and redirecting output to logfile.
    Returns the PID of the started process.
    """
    # Open logfile or /dev/null for child stdout/stderr
    if logfile:
        out = open(logfile, "ab")
        err = out
    else:
        out = open(os.devnull, "wb")
        err = out

    # Start the process detached from the controlling terminal
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=out,
        stderr=err,
        cwd=cwd,
        preexec_fn=os.setsid,
        close_fds=True,
    )

    pid = proc.pid

    # Write pidfile if requested (best-effort)
    if pidfile:
        try:
            Path(pidfile).write_text(str(pid))
        except Exception:
            # Intentionally ignore filesystem errors here; caller can handle if needed
            pass

    return pid


def run_script_daemon(
    script_path: str,
    cwd: str | None = None,
    pidfile: str | None = None,
    logfile: str | None = None,
) -> int:
    """
    Run a bash script as a detached daemon using start_daemon().
    Ensures the script exists and is executable. If cwd is not provided,
    the script's parent directory is used.
    Returns the PID of the started daemon.
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    script = script.resolve()

    # Make executable if not already
    mode = script.stat().st_mode
    if not (mode & 0o111):
        script.chmod(mode | 0o111)

    # Use the script's directory as cwd by default
    if cwd is None:
        cwd = str(script.parent)

    # Quote the script path to be safe when using shell=True in start_daemon
    cmd = f"bash {shlex.quote(str(script))}"

    return start_daemon(cmd, pidfile=pidfile, logfile=logfile, cwd=cwd)


_LAUNCH_LOG_DIR = "/workspace/tmp"
_LLM_LOG = f"{_LAUNCH_LOG_DIR}/vllm_llm.log"
_EMBEDD_LOG = f"{_LAUNCH_LOG_DIR}/vllm_embedd.log"
# How long to wait for a server before giving up. The servers take minutes to
# load a 20B at TP=2, so this is generous -- it exists to stop an INFINITE wait
# when the daemon is already dead, not to race a slow load.
_SERVER_WAIT_SECONDS = 1800


def _daemon_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:
        return True
    return True


def _die_with_log(what: str, pid: int, logfile: str) -> None:
    """Raise with the daemon's own output, which used to go to /dev/null."""
    tail = ""
    try:
        with open(logfile, "r", errors="replace") as fh:
            tail = "".join(fh.readlines()[-40:])
    except Exception as exc:
        tail = f"(could not read {logfile}: {exc})"
    raise RuntimeError(
        f"{what} daemon (pid {pid}) is not running and its port never opened.\n"
        f"--- last lines of {logfile} ---\n{tail}"
    )


def start_vllm_servers() -> int:
    global _EMBEDDING_SERVER_PID

    LLM_server_script = Path("/workspace/scripts/run_vllm_oss120.sh")
    EMBEDD_server_script = Path("/workspace/scripts/run_vllm_qwen3_embedd.sh")

    assert (
        LLM_server_script.exists()
    ), f"LLM server script not found: {LLM_server_script}"
    assert (
        EMBEDD_server_script.exists()
    ), f"Embedding server script not found: {EMBEDD_server_script}"

    os.makedirs(_LAUNCH_LOG_DIR, exist_ok=True)

    # Logfiles, not /dev/null: a daemon that dies on startup must leave evidence.
    _llm_pid = run_script_daemon(
        script_path=str(LLM_server_script),
        logfile=_LLM_LOG,
    )

    _EMBEDDING_SERVER_PID = run_script_daemon(
        script_path=str(EMBEDD_server_script),
        logfile=_EMBEDD_LOG,
    )
    print(f"#### vLLM daemons started: llm pid={_llm_pid} log={_LLM_LOG}, "
          f"embedding pid={_EMBEDDING_SERVER_PID} log={_EMBEDD_LOG} ####")
    _llm_deadline = time() + _SERVER_WAIT_SECONDS

    print("#### Waiting for VLLM servers to start... ####")

    while True:

        base_url = f"http://localhost:{os.getenv('LLM_PORT')}/v1"
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("LLM_MODEL")

        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=model_name,
            )

            if response and isinstance(response.choices[0].message.content, str):
                print(f"#### {model_name} servers are up and running! ####")
                break
        except Exception as e:
            print(f"Waiting for {model_name} servers to be ready... {e}")

        if not _daemon_alive(_llm_pid):
            _die_with_log("LLM", _llm_pid, _LLM_LOG)
        if time() > _llm_deadline:
            raise TimeoutError(
                f"LLM server did not come up within {_SERVER_WAIT_SECONDS}s; "
                f"see {_LLM_LOG}"
            )
        sleep(10)

    print("#### Waiting for Embedding server to start... ####")
    _embedd_deadline = time() + _SERVER_WAIT_SECONDS

    while True:
        base_url = f"http://localhost:{os.getenv('EMBEDD_PORT')}/v1"
        api_key = os.getenv("EMBEDD_API_KEY")
        model_name = os.getenv("EMBEDD_MODEL")

        try:
            client = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                tiktoken_enabled=True,
            )
            response = client.embed_documents(["test embedding"])

            if response and isinstance(response[0], list):
                print(f"#### {model_name} server is up and running! ####")
                break
        except Exception as e:
            print(f"Waiting for {model_name} server to be ready... {e}")

        if not _daemon_alive(_EMBEDDING_SERVER_PID):
            _die_with_log("Embedding", _EMBEDDING_SERVER_PID, _EMBEDD_LOG)
        if time() > _embedd_deadline:
            raise TimeoutError(
                f"Embedding server did not come up within {_SERVER_WAIT_SECONDS}s; "
                f"see {_EMBEDD_LOG}"
            )
        sleep(10)

    print()
    print("#######################################")
    print("#### All VLLM servers are running! ####")
    print("#######################################")
    print()

    return 0


def shutdown_embedding_server():
    """
    Function to shutdown the embedding server to free GPU resources.
    Should be called after embeddings are generated and cached.
    """
    global _EMBEDDING_SERVER_PID

    os.killpg(
        _EMBEDDING_SERVER_PID, signal.SIGTERM  # Process group ID  # Termination signal
    )
    _EMBEDDING_SERVER_PID = None

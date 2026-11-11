"""A private Ollama server for the research agents, with 4 parallel slots.

Ollama fixes the number of parallel slots per server, at start-up, for every model it loads. Four
slots let the finder read 4 chunks at once (about 3x the throughput on an 8 GB RTX 4070), but they
also give every chat model 4 context caches, which pushes bigger models partly onto the CPU. So the
user's own Ollama keeps the default of 1 slot, and the app starts this second server, on a free local
port, the first time research runs. It reads the same downloaded models, holds models only during a
research run, and stops when the app closes. On Windows a job object ties it (and the model runners
it starts) to the app's process, so it can't outlive a crash either.

If the second server can't run (no ollama executable found, Ollama configured on another machine, or
it fails to start), research falls back to the user's Ollama.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import ollama

import models

log = logging.getLogger("app")

RESEARCH_PARALLEL = 4  # finder requests at once (benchmarked: ~3x the throughput of 1)
START_TIMEOUT = 30.0  # seconds for the server to answer
LOG_FILE = Path(__file__).resolve().parent / "research-server.log"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    if sys.platform == "win32":  # the Windows installer's default location, if PATH wasn't updated
        p = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if p.exists():
            return str(p)
    return None


def main_server_is_local(host: str | None = None) -> bool:
    """Whether the user's Ollama (OLLAMA_HOST, as the ollama client reads it) runs on this machine.
    A second server here would be no use for an Ollama on another machine."""
    host = (os.environ.get("OLLAMA_HOST", "") if host is None else host).strip()
    if not host:
        return True
    name = urlsplit(host if "://" in host else f"http://{host}").hostname or ""
    return name in ("", "localhost", "127.0.0.1", "0.0.0.0", "::1", "::")


def server_env(port: int, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the research server: the benchmarked settings, on its own local port."""
    env = dict(os.environ if base is None else base)
    env.update({
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_NUM_PARALLEL": str(RESEARCH_PARALLEL),
        "OLLAMA_KV_CACHE_TYPE": "q8_0",  # each slot needs its own context cache: half-size values
        "OLLAMA_FLASH_ATTENTION": "1",  # required for the q8_0 cache
    })
    models_dir = models._ollama_setting("OLLAMA_MODELS")  # same downloads as the user's Ollama
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir
    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_on_close_job(process_handle: int):
    """Windows: put the process in a job that Windows terminates when the app's handle to it closes,
    i.e. when the app exits for any reason. Model runners the server starts join the job too."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("Reads", "Writes", "Others", "ReadBytes", "WriteBytes",
                                                   "OtherBytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # 9 = extended
            and k32.AssignProcessToJobObject(job, process_handle)):
        raise OSError(ctypes.get_last_error(), "couldn't tie the research server to the app")
    return k32, job


class ResearchServer:
    """Starts on first use (ensure), frees its models after each run (unload_all), stops with the app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job = None
        self.port: int | None = None
        self.failed: str | None = None  # why it couldn't start (then research uses the user's Ollama)

    @property
    def host(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.running else None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure(self, on_start=None) -> str | None:
        """The research server's address, starting it if needed; None to use the user's Ollama."""
        with self._lock:
            if self.running:
                return self.host
            if self.failed:
                return None
            exe = ollama_exe()
            if not exe or not main_server_is_local():
                self.failed = "no local ollama executable" if not exe else "Ollama runs on another machine"
                log.info("research server not used: %s", self.failed)
                return None
            if on_start:
                on_start()
            try:
                self._start(exe)
            except Exception as e:
                self.failed = str(e) or type(e).__name__
                log.warning("research server failed to start (%s); using the main Ollama", self.failed)
                self._kill()
                return None
            return self.host

    def _start(self, exe: str) -> None:
        self.port = _free_port()
        out = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        kw = {"creationflags": _NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
        self._proc = subprocess.Popen([exe, "serve"], env=server_env(self.port), stdout=out,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        out.close()  # the child keeps its own handle
        if sys.platform == "win32":
            try:
                self._job = _kill_on_close_job(int(self._proc._handle))
            except Exception as e:  # still usable; stop() and the app's shutdown end it
                log.warning("research server: %s", e)
        t0 = time.time()
        while time.time() - t0 < START_TIMEOUT:
            if self._proc.poll() is not None:
                raise RuntimeError(f"ollama serve exited (code {self._proc.returncode}); see {LOG_FILE.name}")
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/api/version", timeout=1).raise_for_status()
                log.info("research server on port %s (%s slots) in %.1f s", self.port, RESEARCH_PARALLEL,
                         time.time() - t0)
                return
            except httpx.HTTPError:
                time.sleep(0.25)
        raise TimeoutError(f"ollama serve didn't answer within {START_TIMEOUT:.0f} s")

    def unload_all(self, timeout: float = 10.0) -> list[str]:
        """Free every model on the research server (it's private, so they're all the app's)."""
        if not self.running:
            return []
        client = ollama.Client(host=self.host, timeout=timeout)
        freed = []
        try:
            for m in client.ps().models:
                client.generate(model=m.model, keep_alive=0)
                freed.append(m.model)
        except Exception as e:
            log.warning("research server: unload failed: %s", e)
        return freed

    def stop(self) -> None:
        with self._lock:
            self._kill()

    def _kill(self) -> None:
        """End the server and its model runners (which frees their GPU memory at once)."""
        if self._job:
            k32, job = self._job
            k32.TerminateJobObject(job, 0)
            k32.CloseHandle(job)
            self._job = None
        if self._proc and self._proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(self._proc.pid), "/T", "/F"], capture_output=True,
                               creationflags=_NO_WINDOW)
            else:
                import signal
                try:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                except OSError:
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

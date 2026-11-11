"""Desktop app: a private, local writing assistant built on Ollama.

Run:  .venv\\Scripts\\pythonw app.py      (no console window)
      .venv\\Scripts\\python app.py       (with console, for debugging)

Optional environment overrides:
  OLLAMA_MODEL          force a model (otherwise: settings, then GPU recommendation)
  OLLAMA_NUM_CTX        context window in tokens (default 8192)
  OLLAMA_SYSTEM_PROMPT  system prompt for plain chats (library chats use agent.py's)

Chats are saved in ./threads/, settings in ./settings.json, and the document
library lives wherever the user points it (default ~/Ollama Library).
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import httpx
import ollama
import webview
from markdown_it import MarkdownIt

import agent
import hardware
import models
import research
import settings
from library import Library, LibraryError, clean_title, onedrive_warning

ROOT = Path(__file__).resolve().parent
THREADS_DIR = ROOT / "threads"
UI_FILE = ROOT / "ui" / "index.html"
LOG_FILE = ROOT / "app.log"
OLLAMA_URL = "http://localhost:11434"
CHAT_PROMPT = os.environ.get("OLLAMA_SYSTEM_PROMPT", "You are a helpful, concise assistant.")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

# html=False escapes any raw HTML in model output, so replies can't inject
# markup or scripts into the window. Links are validated (no javascript: URLs).
md = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable(["table", "strikethrough"])


def render(text: str) -> str:
    return md.render(text or "")


_CITE = re.compile(r"\[E(\d+)\]")


def render_report(r: dict) -> dict:
    """A research report (research.Report.to_dict()) prepared for the UI card."""
    answer_html = _CITE.sub(r'<a href="#" class="cite" data-n="\1">[E\1]</a>', render(r.get("answer", "")))
    s = r.get("stats", {})
    bits = [f"{s.get('documents', 0)} document(s)"]
    if (s.get("routed") or s.get("strategy")) == "rag-loop":  # answered by retrieval: no finder read anything
        n = s.get("scans", 0)
        bits.append(f"{n} retrieval search{'es' if n != 1 else ''}")
    elif s.get("chunks"):
        bits.append(f"{s['chunks']} passages read")
    if s.get("cache_hits"):
        bits.append(f"{s['cache_hits']} from cache")
    bits.append(f"{s.get('seconds', 0)} s")
    if s.get("finder_model"):
        same = s["finder_model"] == s.get("researcher_model")
        bits.append(f"model {s['finder_model']}" if same else
                    f"finder {s['finder_model']}, researcher {s.get('researcher_model')}")
    return {
        "kind": r.get("kind"),
        "question": r.get("question", ""),
        "status": r.get("status"),
        "found": r.get("found"),
        "answer_html": answer_html,
        "evidence": [{**e, "cited": e["n"] in r.get("cited", [])} for e in r.get("evidence", [])],
        "flags": r.get("flags", []),
        "stats": " · ".join(bits),
        "note": r.get("note"),
    }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Thread storage
# --------------------------------------------------------------------------


def thread_path(thread_id: str) -> Path:
    if not thread_id or any(c in thread_id for c in "/\\.:"):
        raise ValueError(f"bad thread id: {thread_id!r}")
    return THREADS_DIR / f"{thread_id}.json"


def new_thread(tools: bool) -> dict:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "id": f"{stamp}-{uuid.uuid4().hex[:6]}",
        "title": "New chat",
        "tools": tools,
        "instructions": "",  # this chat's system prompt (style, tone, preferences)
        "created": now_iso(),
        "updated": now_iso(),
        "messages": [],
    }


def save_thread(thread: dict) -> None:
    THREADS_DIR.mkdir(exist_ok=True)
    thread["updated"] = now_iso()
    path = thread_path(thread["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(thread, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_thread(thread_id: str) -> dict:
    t = json.loads(thread_path(thread_id).read_text(encoding="utf-8"))
    t.setdefault("tools", False)  # threads from before the library existed
    t.setdefault("instructions", "")
    return t


def make_title(text: str) -> str:
    title = " ".join(text.split())
    return title[:50] + ("…" if len(title) > 50 else "")


# --------------------------------------------------------------------------
# Ollama server
# --------------------------------------------------------------------------


def server_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_server() -> bool:
    """Start the Ollama server in the background if it isn't running."""
    if server_up():
        return True
    exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
    if not Path(exe).exists():
        log.error("ollama executable not found")
        return False
    log.info("starting ollama server: %s", exe)
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
    )
    for _ in range(30):
        time.sleep(0.5)
        if server_up():
            return True
    return False


# --------------------------------------------------------------------------
# API exposed to the web UI (window.pywebview.api.*)
# --------------------------------------------------------------------------


class Api:
    def __init__(self) -> None:
        self._window: webview.Window | None = None  # underscore: not exposed to JS
        self._cfg = settings.load()
        self._gpu = hardware.detect_gpu()
        self._model = os.environ.get("OLLAMA_MODEL") or self._cfg["model"] or models.recommend(self._gpu["vram_gb"])
        self._num_ctx = int(os.environ.get("OLLAMA_NUM_CTX") or self._cfg["num_ctx"])
        self._lib: Library | None = None
        self._lib_error: str | None = None
        self._open_library(self._cfg["library_path"])
        self._thread = new_thread(self._cfg["tools_default"])
        self._busy = False
        self._stop = threading.Event()
        self._pulling: str | None = None
        log.info("gpu=%s model=%s num_ctx=%s", self._gpu, self._model, self._num_ctx)

    # -- helpers ----------------------------------------------------------

    def _emit(self, fn: str, payload) -> None:
        if self._window:
            self._window.evaluate_js(f"window.{fn}({json.dumps(payload)})")

    def _open_library(self, path: str) -> bool:
        try:
            self._lib, self._lib_error = Library(path), None
            return True
        except (OSError, LibraryError) as e:
            log.error("library unavailable at %s: %s", path, e)
            self._lib, self._lib_error = None, f"Can't use {path}: {e}"
            return False

    def _view(self, thread: dict) -> dict:
        out = []
        for i, m in enumerate(thread["messages"]):
            if m.get("hidden") or m["role"] == "system":
                continue
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "tool":
                out.append({"role": "activity", "content": m.get("activity") or m.get("tool_name", "tool")})
                if m.get("report"):
                    out.append({"role": "report", "report": render_report(m["report"])})
            elif m["role"] == "assistant" and m.get("content"):
                out.append({"role": "assistant", "content": m["content"], "html": render(m["content"]), "index": i})
        return {
            "id": thread["id"],
            "title": thread["title"],
            "tools": bool(thread.get("tools")),
            "instructions": thread.get("instructions", ""),
            "messages": out,
        }

    def _not_busy(self) -> dict | None:
        return {"error": "Wait for the reply to finish (or press Stop)."} if self._busy else None

    # -- app / models -----------------------------------------------------

    def system_info(self) -> dict:
        return {
            "server": ensure_server(),
            "gpu": self._gpu,
            "model": self._model,
            "model_label": (models.by_tag(self._model) or {}).get("label", self._model),
            "num_ctx": self._num_ctx,
            "library": self.library_info(),
        }

    def list_models(self) -> dict:
        sizes = models.installed_sizes()
        have = set(sizes)
        vram = self._gpu["vram_gb"]
        rec = models.recommend(vram)
        items = [
            {
                **m,
                "installed": m["tag"] in have,
                "recommended": m["tag"] == rec,
                "fits": models.fits(m, vram),
                "current": m["tag"] == self._model,
                "tools": True,
            }
            for m in models.CATALOG
        ]
        for tag in self._cfg["custom_models"]:
            size = sizes.get(tag)
            items.append({
                "tag": tag,
                "label": f"🤗 {models.hf_label(tag)}",
                "size_gb": size,
                "installed": tag in have,
                "recommended": False,
                # weights + context cache for this model's real architecture
                "fits": size is None or models.installed_fit_gb(tag, size) <= models.budget_gb(vram),
                "current": tag == self._model,
                "tools": models.supports_tools(tag) if tag in have else None,
                "custom": True,
                "blurb": f"Added from Hugging Face: {tag}",
                "license": "see the model's Hugging Face page",
            })
        for i in items:  # what each downloaded model's chat template supports
            if i["installed"]:
                support = models.template_info(i["tag"])
                i["system"], i["tools"] = support["system"], support["tools"]
        if not any(i["tag"] == self._model for i in items):  # e.g. forced with OLLAMA_MODEL
            items.append({
                "tag": self._model, "label": self._model, "size_gb": sizes.get(self._model),
                "installed": self._model in have, "recommended": False, "fits": True, "current": True,
                "tools": models.supports_tools(self._model) if self._model in have else None,
                "blurb": "Set with OLLAMA_MODEL", "license": "",
            })
        return {"models": items, "current": self._model, "pulling": self._pulling, "gpu": self._gpu}

    def model_support(self) -> dict:
        """What the current model's chat template supports: {'system': bool, 'tools': bool}."""
        if self._model not in models.installed():
            return {"model": self._model, "system": None, "tools": None}
        return {"model": self._model, **models.template_info(self._model)}

    def get_instructions(self) -> dict:
        """The current chat's instructions (each chat has its own)."""
        return {"text": self._thread.get("instructions", ""), "max": settings.MAX_INSTRUCTIONS, **self.model_support()}

    def set_instructions(self, text: str) -> dict:
        if err := self._not_busy():
            return err
        text = str(text or "").strip()
        if len(text) > settings.MAX_INSTRUCTIONS:
            return {"error": f"Instructions are limited to {settings.MAX_INSTRUCTIONS:,} characters."}
        self._thread["instructions"] = text
        if thread_path(self._thread["id"]).exists():  # a new chat is saved with its first message
            save_thread(self._thread)
        return self.get_instructions()

    def add_hf_model(self, ref: str) -> dict:
        """Add a GGUF model from Hugging Face (validated) and start downloading it."""
        tag = models.normalize_hf(ref)
        if not tag:
            return {"error": "That doesn't look like a Hugging Face model. Use the form "
                             "hf.co/<user>/<repo> or hf.co/<user>/<repo>:<quantization>, "
                             "e.g. hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"}
        if tag not in self._cfg["custom_models"]:
            self._cfg["custom_models"].append(tag)
            settings.save(self._cfg)
        if tag in models.installed():
            return {"tag": tag, "installed": True}
        result = self.pull_model(tag)
        return {"tag": tag, **result}

    def remove_custom_model(self, tag: str) -> dict:
        if err := self._not_busy():
            return err
        if tag not in self._cfg["custom_models"]:
            return {"error": "Only models you added can be removed here."}
        if self._pulling == tag:
            return {"error": "Wait for the download to finish first."}
        try:
            if tag in models.installed():
                models.delete(tag)
        except Exception as e:
            log.warning("delete %s failed: %s", tag, e)
        self._cfg["custom_models"].remove(tag)
        if self._model == tag:
            self._model = models.recommend(self._gpu["vram_gb"])
            self._cfg["model"] = None
        settings.save(self._cfg)
        return self.list_models()

    def set_model(self, tag: str) -> dict:
        if err := self._not_busy():
            return err
        if tag not in models.installed():
            return {"error": f"{tag} isn't downloaded yet."}
        self._model = tag
        self._cfg["model"] = tag
        settings.save(self._cfg)
        log.info("model -> %s", tag)
        self.warm()
        return self.list_models()

    def warm(self) -> None:
        """Load the current model in the background and report how it fits."""
        model, num_ctx = self._model, self._num_ctx

        def run():
            if model not in models.installed():
                self._emit("onFit", {"model": model, "label": "not downloaded"})
                return
            try:
                ollama.generate(model=model, prompt="", options={"num_ctx": num_ctx}, keep_alive="10m")
            except Exception as e:
                log.warning("warm-up failed: %s", e)
            self._emit("onFit", {"model": model, **(hardware.fit_check(model) or {"label": "loaded"})})

        threading.Thread(target=run, daemon=True).start()

    def pull_model(self, tag: str) -> dict:
        if not models.by_tag(tag) and tag not in self._cfg["custom_models"]:
            return {"error": "Only models from the built-in list, or ones you added, can be downloaded here."}
        if self._pulling:
            return {"error": f"Already downloading {self._pulling}."}
        self._pulling = tag

        def run():
            last = 0.0

            def progress(status: str, done: int, total: int) -> None:
                nonlocal last
                now = time.monotonic()
                if now - last > 0.25:
                    self._emit("onPull", {"tag": tag, "status": status, "completed": done, "total": total})
                    last = now

            try:
                models.pull(tag, progress)
                self._emit("onPull", {"tag": tag, "done": True})
            except Exception as e:
                log.error("pull %s failed: %s", tag, e)
                # A Hugging Face model that failed to download (typo, not GGUF, private
                # repo…) is dropped from the list rather than left as a dead entry.
                if tag in self._cfg["custom_models"] and tag not in models.installed():
                    self._cfg["custom_models"].remove(tag)
                    settings.save(self._cfg)
                self._emit("onPull", {"tag": tag, "error": models.friendly_pull_error(tag, str(e))})
            finally:
                self._pulling = None

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    # -- library ----------------------------------------------------------

    def library_info(self) -> dict:
        path = str(self._lib.root) if self._lib else self._cfg["library_path"]
        return {"path": path, "ok": self._lib is not None, "error": self._lib_error, "warning": onedrive_warning(path)}

    def list_documents(self) -> dict:
        if not self._lib:
            return {"sources": [], "drafts": []}
        docs = self._lib.describe()
        return {
            "sources": [d for d in docs if d["kind"] == "source"],
            "drafts": [d for d in docs if d["kind"] == "draft"],
        }

    def import_files(self) -> dict:
        if not self._lib or not self._window:
            return {"error": "The library folder isn't available."}
        picked = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True, file_types=("Text documents (*.txt;*.md)",)
        )
        if not picked:
            return {"imported": [], "skipped": []}
        return self._lib.import_files(list(picked))

    def choose_library_folder(self) -> dict:
        if err := self._not_busy():
            return err
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER) if self._window else None
        if not picked:
            return self.library_info()
        path = picked[0] if isinstance(picked, (list, tuple)) else picked
        if self._open_library(path):
            self._cfg["library_path"] = str(self._lib.root)
            settings.save(self._cfg)
        return self.library_info()

    def open_library_folder(self) -> None:
        if self._lib:
            self._lib.open_folder()

    def view_document(self, doc_id: str, start: int | None = None, end: int | None = None) -> dict:
        """Document for the viewer; with start/end, that passage is highlighted (a citation)."""
        try:
            doc, text = self._lib.read_all(doc_id)
        except (LibraryError, OSError, AttributeError) as e:
            return {"error": str(e)}
        is_md = doc.path.suffix.lower() == ".md"
        out = {"id": doc.id, "title": doc.title, "kind": doc.kind, "text": text,
               "html": render(text) if is_md else None, "highlight": False}
        try:
            s, e = int(start), int(end)
        except (TypeError, ValueError):
            return out
        if 0 <= s < e <= len(text):
            # Private-use characters mark the passage; they survive escaping/Markdown
            # and are swapped for <mark> afterwards, so no model text becomes HTML.
            marked = text[:s] + "" + text[s:e] + "" + text[e:]
            body = render(marked) if is_md else f'<div class="plain">{html.escape(marked)}</div>'
            out["html"] = body.replace("", '<mark id="hl">', 1).replace("", "</mark>", 1)
            out["highlight"] = True
        return out

    def open_document(self, doc_id: str) -> dict:
        try:
            self._lib.open_external(doc_id)
            return {"ok": True}
        except (LibraryError, OSError, AttributeError) as e:
            return {"error": str(e)}

    def save_reply_as_draft(self, index: int) -> dict:
        """Deterministic save: the user clicks, the app writes. No model involved."""
        if not self._lib:
            return {"error": "The library folder isn't available."}
        try:
            msg = self._thread["messages"][int(index)]
        except (IndexError, ValueError, TypeError):
            return {"error": "That reply no longer exists."}
        if msg.get("role") != "assistant" or not msg.get("content"):
            return {"error": "Only assistant replies can be saved."}
        try:
            doc = self._lib.create_draft(clean_title(self._thread["title"], "Reply"), msg["content"])
        except LibraryError as e:
            return {"error": str(e)}
        return {"title": doc.title, "id": doc.id}

    # -- threads ----------------------------------------------------------

    def list_threads(self) -> list[dict]:
        THREADS_DIR.mkdir(exist_ok=True)
        items = []
        for p in THREADS_DIR.glob("*.json"):
            try:
                t = json.loads(p.read_text(encoding="utf-8"))
                items.append({"id": t["id"], "title": t["title"], "updated": t["updated"], "tools": t.get("tools", False)})
            except (OSError, ValueError, KeyError):
                log.warning("skipping unreadable thread file %s", p.name)
        items.sort(key=lambda t: t["updated"], reverse=True)
        return items

    def current(self) -> dict:
        return self._view(self._thread)

    def new_chat(self) -> dict:
        if err := self._not_busy():
            return err
        self._thread = new_thread(self._cfg["tools_default"])
        return self._view(self._thread)

    def open_thread(self, thread_id: str) -> dict:
        if err := self._not_busy():
            return err
        try:
            self._thread = load_thread(thread_id)
        except (OSError, ValueError) as e:
            return {"error": f"Couldn't open thread: {e}"}
        return self._view(self._thread)

    def rename_thread(self, thread_id: str, title: str) -> bool:
        title = make_title(title) or "Untitled"
        t = self._thread if thread_id == self._thread["id"] else load_thread(thread_id)
        t["title"] = title
        if thread_path(thread_id).exists():
            save_thread(t)
        return True

    def delete_thread(self, thread_id: str) -> dict:
        if self._busy and thread_id == self._thread["id"]:
            return {"error": "Can't delete a chat while it's replying."}
        try:
            thread_path(thread_id).unlink(missing_ok=True)
        except OSError as e:
            return {"error": f"Couldn't delete: {e}"}
        if thread_id == self._thread["id"]:
            self._thread = new_thread(self._cfg["tools_default"])
        return self._view(self._thread)

    def set_tools_enabled(self, on: bool) -> dict:
        if err := self._not_busy():
            return err
        self._thread["tools"] = bool(on)
        if thread_path(self._thread["id"]).exists():
            save_thread(self._thread)
        self._cfg["tools_default"] = bool(on)  # remembered for new chats
        settings.save(self._cfg)
        return {"tools": self._thread["tools"]}

    # -- chatting -----------------------------------------------------------

    def send(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "empty"}
        if self._busy:
            return {"error": "Already replying."}
        if self._model not in models.installed():
            return {"error": f"{self._model} isn't downloaded yet. Pick a model above and download it."}
        if self._thread.get("tools") and not self._lib:
            return {"error": "The library folder isn't available. Choose one in the Library tab, or turn library tools off."}
        if self._thread.get("tools") and not models.supports_tools(self._model):
            return {"error": f"{self._model} doesn't support tool calling, so it can't use the library. "
                             "Turn off 📚 Library tools for a plain chat, or pick another model."}
        self._busy = True
        self._stop.clear()
        thread = self._thread
        if not thread["messages"]:
            thread["title"] = make_title(text)
        thread["messages"].append({"role": "user", "content": text})
        save_thread(thread)
        threading.Thread(target=self._generate, args=(thread,), daemon=True).start()
        return {"id": thread["id"], "title": thread["title"]}

    def stop(self) -> None:
        self._stop.set()

    def log_client_error(self, message: str) -> None:
        """JavaScript errors from the page end up in app.log instead of vanishing."""
        log.error("ui: %s", str(message)[:2000])

    def open_link(self, url: str) -> None:
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            webbrowser.open(url)

    # -- background generation -------------------------------------------

    def _generate(self, thread: dict) -> None:
        model = self._model
        start_len = len(thread["messages"])
        last_push = 0.0

        def on_text(full: str) -> None:
            nonlocal last_push
            now = time.monotonic()
            if now - last_push > 0.05:  # throttle re-renders to ~20/s
                self._emit("onChunk", render(full))
                last_push = now

        def on_activity(label: str) -> None:
            self._emit("onActivity", label)

        try:
            if thread.get("tools") and self._lib:
                status = agent.run_turn(
                    self._lib, thread["messages"], model, self._num_ctx, on_text, on_activity, self._stop.is_set,
                    instructions=thread.get("instructions", ""),
                    system_supported=models.template_info(model)["system"],
                    research=self._research_ctx(model),
                )
            else:
                status = self._plain_chat(thread, model, on_text)
        except ollama.ResponseError as e:
            log.error("ollama error %s: %s", e.status_code, e.error)
            hint = f" Try downloading {model} again." if e.status_code == 404 else ""
            return self._fail(thread, start_len, f"Ollama error: {e.error}.{hint}")
        except (ConnectionError, httpx.HTTPError) as e:
            log.error("connection error: %s", e)
            return self._fail(thread, start_len, "Can't reach the Ollama server. Start the Ollama app and try again.")
        except Exception as e:  # keep the UI usable no matter what
            log.exception("unexpected error")
            return self._fail(thread, start_len, f"Unexpected error: {e}")

        last = thread["messages"][-1]
        final = last if last["role"] == "assistant" and not last.get("tool_calls") else None
        save_thread(thread)
        payload = {
            "html": render(final["content"]) if final else "",
            "index": len(thread["messages"]) - 1 if final else None,
            "status": status,
            "fit": {"model": model, **(hardware.fit_check(model) or {})},
        }
        self._busy = False
        self._emit("onDone", payload)

    def _research_ctx(self, model: str) -> dict:
        """Settings and callbacks for the research tools (see agent.ToolBox)."""
        have = models.installed()

        def pick(key: str) -> str:
            chosen = self._cfg.get(key)
            if chosen and chosen in have:
                return chosen
            if key == "finder_model" and research.DEFAULT_FINDER in have:
                return research.DEFAULT_FINDER
            if key == "researcher_model" and research.DEFAULT_RESEARCHER in have:
                return research.DEFAULT_RESEARCHER
            return model  # fall back to the chat model

        search = research.app_search_options(self._cfg.get("research_strategy"), have)

        def on_report(report) -> str | None:
            note = None
            retrieved = (report.stats.get("routed") or report.stats.get("strategy")) == "rag-loop"
            if search["flag"] and retrieved and search["flag"] not in report.flags:
                report.flags.append(search["flag"])
            if report.status == "done" and (report.evidence or report.kind == "summary"):
                try:
                    note = research.save_note(self._lib, report).title  # app action, not the model's
                except LibraryError as e:
                    log.warning("research note not saved: %s", e)
            self._emit("onResearch", render_report({**report.to_dict(), "note": note}))
            return note

        return {
            "finder_model": pick("finder_model"),
            "researcher_model": pick("researcher_model"),
            "workers": self._cfg.get("research_workers") or models.server_parallel(),
            "think": bool(self._cfg.get("research_thinking")),
            "strategy": search["strategy"],  # rag and hybrid are benchmark controls only
            "rag_mode": search["rag_mode"],  # None = the strategy's default; "keyword" without the embedder
            "deep": bool(self._cfg.get("deep_research")),
            "on_progress": lambda text: self._emit("onProgress", text),
            "on_report": on_report,
        }

    def research_settings(self) -> dict:
        have = sorted(models.installed())
        return {
            "finder_model": self._cfg.get("finder_model"),
            "researcher_model": self._cfg.get("researcher_model"),
            "workers": self._cfg.get("research_workers") or models.server_parallel(),
            "server_parallel": models.server_parallel(),
            "default_finder": research.DEFAULT_FINDER if research.DEFAULT_FINDER in have else None,
            "default_researcher": research.DEFAULT_RESEARCHER if research.DEFAULT_RESEARCHER in have else None,
            "installed": have,
        }

    def set_research_settings(self, finder: str | None, researcher: str | None, workers: int | None) -> dict:
        have = models.installed()
        for m in (finder, researcher):
            if m and m not in have:
                return {"error": f"{m} isn't downloaded."}
        self._cfg["finder_model"] = finder or None
        self._cfg["researcher_model"] = researcher or None
        try:
            self._cfg["research_workers"] = max(1, min(int(workers), 16)) if workers else None
        except (TypeError, ValueError):
            self._cfg["research_workers"] = None
        settings.save(self._cfg)
        return self.research_settings()

    def _plain_chat(self, thread: dict, model: str, on_text) -> str:
        reply: list[str] = []
        system = thread.get("instructions", "").strip() or CHAT_PROMPT
        msgs = agent.with_system(system, agent.to_ollama(thread["messages"]), models.template_info(model)["system"])
        stream = ollama.chat(
            model=model, messages=msgs, stream=True, options={"num_ctx": self._num_ctx}, **models.chat_options(model)
        )
        stopped = False
        for chunk in stream:
            reply.append(chunk.message.content or "")
            on_text("".join(reply))
            if self._stop.is_set():
                stopped = True
                stream.close()
                break
        if "".join(reply):
            thread["messages"].append({"role": "assistant", "content": "".join(reply)})
        return "stopped" if stopped else "done"

    def _fail(self, thread: dict, start_len: int, message: str) -> None:
        if len(thread["messages"]) == start_len:
            thread["messages"].pop()  # nothing happened: drop the unanswered user turn
        if thread["messages"]:
            save_thread(thread)
        else:
            thread_path(thread["id"]).unlink(missing_ok=True)
        self._busy = False
        self._emit("onError", {"message": message})


def main() -> None:
    api = Api()
    window = webview.create_window(
        "Ollama Chat",
        url=str(UI_FILE),
        js_api=api,
        width=1180,
        height=800,
        min_size=(760, 520),
        background_color="#1b1b1f",
    )
    api._window = window
    webview.start()


if __name__ == "__main__":
    main()

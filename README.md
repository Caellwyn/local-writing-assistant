# Ollama Chat: a private, local writing assistant

A desktop app that runs open-weight language models **entirely on your own computer** (via [Ollama](https://ollama.com)) and works on a sandboxed library of your documents: it summarizes, edits, polishes, and writes from your suggestions. Nothing leaves your machine except model downloads.

It's a learning/portfolio project. The interesting part isn't the chat window; it's how to give a small, fallible model access to your files **safely**, and how to make it useful despite its mistakes.

## Features

- **Streaming chat** with saved, renameable threads.
- **Document library**: import `.txt`/`.md` files; the assistant reads them and writes drafts. Every revision keeps the previous version.
- **GPU-aware model picker**: detects your graphics card, recommends a model that fits, downloads models with a progress bar, and shows whether the loaded model runs 100% on the GPU.
- **Hugging Face models**: add any GGUF model with `hf.co/<user>/<repo>[:<quant>]`.
- **"Save as draft" on every reply**, so saving never depends on the model getting a tool call right.
- A terminal chatbot (`chat.py`) too.

## Security model

The model can't *do* anything by itself: it only produces text. Text becomes action only through the tools this app gives it, so the design question is **what those tools can reach**.

| Threat | Design response |
|---|---|
| Path tricks (`..\..\`, `C:\Windows`, `\\server\share`, `file.txt:stream`, `CON`, symlinks/junctions) | The model never handles paths. It sees opaque IDs (`src-1a2b3c`); the app maps them to files it listed itself. Every path is still resolved and checked against the library root (`library.safe_path`, with tests for each trick). |
| Overwriting or destroying work | Sources are read-only to the model. Drafts are never overwritten in place: the old version goes to `.history/` first. There's no delete tool. |
| Runaway agent | Size/count limits on drafts, at most 8 steps per turn, one page (~1,200 words) per read. |
| Prompt injection (a document saying "ignore your instructions and…") | Contained, not prevented: there's **no network tool and no code execution**, so hostile text can't leak data or touch the system. The worst case is a junk draft. In testing, one model obeyed an injected instruction and created a draft titled `System_Hacked`; nothing outside the library was touched. |
| Model claims it did something it didn't | The app, not the model, reports outcomes: tool activity is shown as it happens, and if you asked to save and nothing was saved, the app says so under the reply. |
| Unrequested writes | The model may only save or revise when your message asks for it (save, draft, edit, polish, revise…). Otherwise it answers in the chat. |
| Cloud sync leaking "private" files | Default library is `~/Ollama Library`, and the app warns if you choose a OneDrive-synced folder. |
| Hostile HTML in model output | Markdown is rendered with raw HTML disabled; links open in your browser. |

**Known limits**: a small model can still be steered by a document into a *misleading summary*, and small models confidently make things up. (Asked to translate the Bengali insult "mathay gobor bhora", roughly "head full of cow dung", Llama 3.1 8B answered "thank you!") Use it to work *on* your text, not as a source of facts.

### Lessons from testing the agent

Workflow rules the app enforces, because prompting alone wasn't enough for 4–8B models:

- **Read before writing.** If your message names a document, a draft about it can't be saved until it's been read. (One model saved a summary that said "Please provide the content of chapter1.")
- **Placeholder text is rejected** (`<insert summary here>`).
- **Document list in the prompt**: models guessed IDs like `src-123456` until the list was provided every step.
- **Argument aliases** (`draft_title` → `title`, `text` → `content`) and recovery of tool calls written as JSON text.
- **One automatic reminder** when you asked to save and the model just said it would.

Model choice mattered more than any prompt: Qwen3 4B (instruct) completed the read → save → revise workflow reliably; Llama 3.1 8B writes well but fumbled multi-step tool calls. The recommendation logic reflects that.

## Models

The app recommends the best model for library work that fits your GPU (it keeps ~1.5 GB of video memory free for the system):

| Model | Download | GPU memory (8K context) | Notes |
|---|---|---|---|
| Llama 3.2 3B | 2.0 GB | ~3.2 GB | Small GPUs and CPU-only machines |
| Qwen3 4B (instruct) | 2.5 GB | ~4.0 GB | Most reliable tool use; recommended for 6–10 GB GPUs |
| Llama 3.1 8B | 4.9 GB | 5.8 GB (measured) | Good prose; less reliable with tools |
| Qwen3 8B | 5.2 GB | ~6.6 GB | 12 GB GPUs |
| Gemma 4 12B | 7.2 GB | ~8.8 GB | Tool use not yet tested here |
| Qwen3 14B | 9.3 GB | ~10.9 GB | 16 GB GPUs |

Measured on an RTX 4070 Laptop GPU (8 GB) with Llama 3.1 8B:

| Context | Memory | Runs on |
|---|---|---|
| 4K | 5.3 GB | 100% GPU |
| **8K** (default) | 5.8 GB | 100% GPU |
| 16K | 7.3 GB | 13% CPU / 87% GPU |
| 32K | 9.6 GB | 34% CPU / 66% GPU |

Check model licenses before redistributing anything: Llama models use Meta's community licenses, Qwen3 is Apache 2.0, and Gemma has its own terms.

### Adding models from Hugging Face

Pick **＋ Add from Hugging Face…** in the model menu and paste a GGUF repo, e.g. `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` (or the repo's URL). For library tools, choose an *instruct* model that supports tool/function calling; the app checks and tells you if it doesn't. Model files are data, not programs, but anyone can upload one: stick to publishers you trust.

## Install (Windows)

1. Install Ollama: `irm https://ollama.com/install.ps1 | iex` in PowerShell (or from [ollama.com/download](https://ollama.com/download)).
2. Set up Python 3.10+:
   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
3. Run it: `.venv\Scripts\pythonw app.py`. On first launch it offers to download the recommended model.
4. Optional desktop shortcut: `.\create_shortcut.ps1`.

Windows may show a SmartScreen warning for unsigned apps; that's expected for an open-source project.

Settings you can override with environment variables: `OLLAMA_MODEL`, `OLLAMA_NUM_CTX` (default 8192), `OLLAMA_SYSTEM_PROMPT` (plain chats). To use longer contexts on the same GPU memory, Ollama's `OLLAMA_KV_CACHE_TYPE=q8_0` (about half the memory) or `q4_0` (about a quarter) quantizes the context cache server-wide, with some loss in precision.

## Project layout

| File | What it does |
|---|---|
| `app.py` | Desktop window (pywebview), chat threads, API for the UI |
| `ui/index.html` | The interface |
| `library.py` | The sandbox: document IDs, path safety, drafts, history, limits |
| `agent.py` | Tool definitions and the tool-calling loop, with workflow guardrails |
| `models.py` | Model catalog, GPU-based recommendation, downloads, Hugging Face refs |
| `hardware.py` | GPU detection (NVIDIA via `nvidia-smi`, AMD via the registry) and GPU/CPU split checks |
| `settings.py` | `settings.json` (model, context size, library folder) |
| `chat.py` | Terminal chatbot |
| `tests/` | pytest suite (path escapes, limits, guardrails, model logic) |

Run the tests: `.venv\Scripts\pip install -r requirements-dev.txt`, then `.venv\Scripts\python -m pytest`.

## License

MIT. See [LICENSE](LICENSE).

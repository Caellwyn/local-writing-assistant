# Ollama Chat: a private, local writing assistant

A desktop app that runs open-weight language models **entirely on your own computer** (via [Ollama](https://ollama.com)) and works on a sandboxed library of your documents: it summarizes, edits, polishes, and writes from your suggestions. Nothing leaves your machine except model downloads.

It's a learning/portfolio project. The interesting part isn't the chat window; it's how to give a small, fallible model access to your files **safely**, and how to make it useful despite its mistakes.

## Features

- **Streaming chat** with saved, renameable threads.
- **Document library**: import `.txt`/`.md` files; the assistant reads them and writes drafts. Every revision keeps the previous version.
- **GPU-aware model picker**: detects your graphics card, recommends a model that fits, downloads models with a progress bar, and shows whether the loaded model runs 100% on the GPU.
- **Hugging Face models**: add any GGUF model with `hf.co/<user>/<repo>[:<quant>]`.
- **"Save as draft" on every reply**, so saving never depends on the model getting a tool call right.
- **Per-chat instructions** (⚙ Instructions): your own system prompt for tone, style and preferences, saved with each chat. New chats start blank.
- **Reads each model's chat template** (from the GGUF file) to see what it supports. Models without tool calling can't turn on library tools. For models whose template has no system role, instructions are folded into the first message, because Ollama would otherwise drop them silently.
- **Research agents**: ask a question about your documents and get an answer whose **citations are exact passages** from the sources (click one to open the document at that passage), or summarize whole documents of any length. See below.
- A terminal chatbot (`chat.py`) too.

## Research agents

Retrieval (embeddings, top-k) decides *in advance* which passages a model gets to see, and it's hard to tell what it missed. Local models cost nothing per token, so this app can also take the brute-force route: **small models read every passage**. Each route suits different questions, so a router picks one per question (strategy `auto`, the default):

- **Fact and chain questions** ("Who…?", "Who did X, and what did that person do?") use **retrieval in a loop**. The researcher splits the question into parts and retrieves passages for each part (embeddings and keywords, fused). It answers, then searches again for any part that's still open, using what the earlier parts found. This takes seconds per question.
- **Judgement and collect-all questions** ("What are Holmes's common habits?", "List every…") use the **full scan** below, because retrieval found only a few percent of the hundreds of cases these need. The answer is a conclusion followed by the specific cases, each with its citations.

The router judges a question by its wording alone, never by what a search found. Results are in [evals/results.md](evals/results.md) (Library section).

The full scan:

```text
you ──► chat model (supervisor) ──► research_library / summarize_documents tool
                                        │
                     researcher model   │ 1. writes a brief: what to look for, terms, what counts
                                        ▼
                ┌──── finder ──┬──── finder ──┬──── finder ──┬─ … every chunk of every document, shown as
                │  (parallel)  │              │              │   numbered sentences; finders answer with
                └──────────────┴──────┬───────┴──────────────┘   numbers only: {"sentences": [12, 13]}
                                      ▼
                     app: extracts those sentences from the source text  ──► evidence E1…En
                                      ▼
                     researcher model: answers from the evidence only, citing [E#]
                                      ▼
                     app: checks every [E#] exists, renders the cited report, saves a research note
```

- **The app orchestrates; models only judge.** The fan-out is a bounded thread pool, not agents spawning agents, so it's predictable, cancellable and shows progress.
- **Finders point; the app quotes.** The app splits each chunk into numbered sentences, and a finder answers with sentence numbers, never text. The app extracts those sentences itself, so evidence is always exact source text: a model can't paraphrase, merge or invent a quote. Numbers that don't exist are ignored and counted. The answer may only cite real evidence; citations to anything else are removed and flagged.
- **Why pointers:** Ollama's logs showed finders spent 78% of their time *writing*: copying quotes, and ~40 tokens of JSON even for chunks with nothing relevant. A model writes one token per step, so copying text the app already has is pure waste. Pointer replies are 7–20 tokens. (Models can't reliably count token positions, which is why the app labels sentences instead of asking for offsets.) An earlier version had finders copy quotes and the app verify them word for word; [evals/results.md](evals/results.md) compares the two.
- **No evidence, no answer.** If the scan finds nothing, the app says so without asking a model, so an empty library can't produce a confident made-up answer.
- **Structured output, not tool calling**, for finders and researchers (Ollama's JSON-schema `format`), so even 1–2B models work as finders.
- **Context sized per role.** Ollama reserves the full context for every parallel slot. A finder's prompt is ~2,400 tokens, so when the finder is a different model from the researcher (it's swapped in anyway), it gets a 4K context instead of 8K, halving its context memory. With one model for both, everything stays at 8K, because Ollama reloads a model whenever its context size changes.
- **Summaries** are map → reduce: per-passage notes pointing at supporting sentences, then a per-document summary (hierarchical for long documents), then a cross-document overview. Per-document summaries are cached by content hash, so asking again is instant until the file changes.
- **Cost**: a full scan reads the whole library. On an 8 GB laptop GPU that's roughly 40 seconds for ~22,000 words with a 4B model, and a few minutes for three whole novels (~207,000 words). Retrieval-routed questions take seconds. The first one embeds the library once, on the CPU; the embeddings are cached in the library.

Finder and researcher models are chosen separately (🔬 Research). For example, a small fast finder can pair with a bigger researcher. The defaults are the benchmark's winners ([evals/results.md](evals/results.md)), used when they're downloaded:

```powershell
ollama pull qwen3:4b-instruct      # finder
ollama pull qwen3.5:9b             # researcher (thinking off)
ollama pull qwen3-embedding:0.6b   # retrieval; without it, retrieval falls back to keywords only
```

### Parallelism: a private research server

Ollama sets the number of parallel requests per server, when it starts, for every model it loads. Research wants 4 (the finder reads 4 chunks at once: about 3× the throughput on the RTX 4070), but each slot needs its own context cache, so 4 slots push chat models partly onto the CPU: on the dev machine Qwen3 8B ran at 23 tokens/s (76% GPU) with 4 slots and 34 tokens/s (100% GPU) with 1.

So your Ollama keeps its default of 1 slot, and the app starts a second, private Ollama server ([research_server.py](research_server.py)) the first time research runs:

- on a free local port, with `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_FLASH_ATTENTION=1`, reading the same downloaded models (~3–4 s to start, once per session);
- before each run the app unloads its chat model from your Ollama, so the two servers don't compete for GPU memory; after the run it unloads the research models, and the chat model reloads for the reply;
- it stops when the app closes. On Windows a job object ties it to the app's process, so it also ends if the app crashes.

If it can't start (no `ollama` executable found, or `OLLAMA_HOST` points to another machine), research uses your Ollama with however many slots it has.

Optional, for your own Ollama: these two Windows **user** environment variables halve context memory for every model (restart Ollama after setting them):

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Half-size context cache, little quality loss |
| `OLLAMA_FLASH_ATTENTION` | `1` | Required for the compressed cache |

When closing, the app also unloads every model it loaded from your Ollama, instead of leaving them in GPU memory until Ollama's keep-alive (5–10 minutes) runs out.

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

The app recommends the best model for library work that fits your GPU. It estimates memory from each model's architecture: weights, plus the context cache (layers × KV heads × head size × context × parallel slots, at the configured cache precision), plus about 0.3 GB of buffers. It also allows for about 2 GB of video memory that models can't use. Hybrid models such as Qwen3.5 keep a cache on only every 4th layer, and the estimate accounts for that. The live status bar shows the real GPU/CPU split once a model is loaded.

| Model | Download | GPU memory (8K, 1 slot, 16-bit cache) | GPU memory (8K, 4 slots, q8_0 cache) | Notes |
|---|---|---|---|---|
| Llama 3.2 3B | 2.0 GB | ~3.0 GB | 3.9 GB (measured) | Small GPUs and CPU-only machines |
| Qwen3 4B (instruct) | 2.5 GB | ~3.8 GB | 4.9 GB (measured) | Most reliable tool use; recommended for 6–10 GB GPUs |
| Qwen3.5 4B | 3.3 GB | ~3.6 GB | 3.9 GB (measured) | Small context cache; tool use not yet tested here |
| Llama 3.1 8B | 4.9 GB | 5.8 GB (measured) | 7.0 GB (measured) | Good prose; less reliable with tools |
| Qwen3 8B | 5.2 GB | ~6.3 GB | ~7.5 GB | 12 GB GPUs |
| Qwen3.5 9B | 6.6 GB | ~6.7 GB | 7.0 GB (measured) | Default research model; sometimes answers library questions from memory instead of using a tool |
| Gemma 4 12B | 7.2 GB | ~8.8 GB | ~8.8 GB | Tool use not yet tested here |
| Qwen3 14B | 9.3 GB | ~10.2 GB | ~11.6 GB | 16 GB GPUs |

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
| `research.py` | Research agents: chunking, parallel finders, quote verification, cited synthesis, cached summaries |
| `evals/` | Benchmark: public-domain corpus with planted facts, deterministic scoring, `results.md` |
| `models.py` | Model catalog, GPU-based recommendation, downloads, Hugging Face refs |
| `hardware.py` | GPU detection (NVIDIA via `nvidia-smi`, AMD via the registry) and GPU/CPU split checks |
| `settings.py` | `settings.json` (model, context size, library folder, research models) |
| `chat.py` | Terminal chatbot |
| `tests/` | pytest suite (path escapes, limits, guardrails, model logic) |

Run the tests: `.venv\Scripts\pip install -r requirements-dev.txt`, then `.venv\Scripts\python -m pytest`.

Run the benchmark (it needs the models installed, and takes 1–2 hours):

```powershell
.venv\Scripts\python evals\build_corpus.py
.venv\Scripts\python evals\run_eval.py --round finders
.venv\Scripts\python evals\run_eval.py --round researchers
.venv\Scripts\python evals\run_eval.py --round supervisor
```

## License

MIT. See [LICENSE](LICENSE).

"""A small streaming terminal chatbot using Ollama.

Run:  .venv\\Scripts\\python chat.py
Model: set OLLAMA_MODEL to change it (default: llama3.1:8b).
Context: set OLLAMA_NUM_CTX to change it (default: 8192 tokens).
Commands: /reset clears the conversation, /bye exits.
"""

import os
import sys

import ollama

try:
    import settings as _settings  # the desktop app's saved model choice, if any

    _saved_model = _settings.load().get("model")
except Exception:
    _saved_model = None
MODEL = os.environ.get("OLLAMA_MODEL") or _saved_model or "llama3.1:8b"
# Context window in tokens. 8192 keeps llama3.1:8b fully on an 8 GB GPU;
# larger values spill to CPU and slow down (check with `ollama ps`).
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
SYSTEM_PROMPT = os.environ.get(
    "OLLAMA_SYSTEM_PROMPT", "You are a helpful, concise assistant."
)


def new_history() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []


def main() -> int:
    messages = new_history()
    print(
        f"Chatting with {MODEL} ({NUM_CTX}-token context). "
        "Type /reset to start over, /bye to quit.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input == "/bye":
            return 0
        if user_input == "/reset":
            messages = new_history()
            print("(conversation cleared)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)

        reply = []
        try:
            for chunk in ollama.chat(
                model=MODEL,
                messages=messages,
                stream=True,
                options={"num_ctx": NUM_CTX},
            ):
                piece = chunk.message.content
                reply.append(piece)
                print(piece, end="", flush=True)
        except ollama.ResponseError as e:
            messages.pop()  # drop the unanswered user turn
            print(f"\n[error {e.status_code}] {e.error}")
            if e.status_code == 404:
                print(f"Try: ollama pull {MODEL}")
            print()
            continue
        except ConnectionError:
            messages.pop()
            print("\n[error] Can't reach the Ollama server at localhost:11434.")
            print("Start the Ollama app, or run: ollama serve\n")
            continue

        print("\n")
        messages.append({"role": "assistant", "content": "".join(reply)})


if __name__ == "__main__":
    sys.exit(main())

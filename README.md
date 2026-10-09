# telegram-modal-llm-bot
connect telegram bot to modal llm

A private, single-user Telegram text-chat bot backed by an on-demand Modal GPU function. This repository packages the existing Python implementation without adding a search engine or an agent framework.

## Architecture and scope

```text
Telegram user
    ↕ Telegram Bot API (long polling)
Linux CPU host: Python bot + SQLite history
    ↕ authenticated Modal Python SDK function calls
Modal: single-use A100-80GB container + vLLM
    ↕ read-only Modal Volume
Pinned Qwen3.8-27B-Uncensored BF16 weights
```

- Long polling: no inbound webhook server or extra public inference port.
- Authorization checks both the private chat ID and sender ID before submitting inference.
- Plain-text messages invoke the model; `/status` reads remote counters and `/stop` cancels the exact remote call with container termination.
- Completed user/assistant turns are stored in SQLite on the CPU host. Recent history is resent with each request and trimmed to fit the context window.
- Results are sent after generation completes, not streamed token by token.
- One inference at a time. Remote retries are disabled. Ambiguous submissions or unconfirmed release block further requests instead of silently resubmitting.
- GPU containers are single-use, with `min_containers=0`, `buffer_containers=0` and `max_containers=1`. The bot stays online while the GPU scales to zero.
- No web search, browser tools, agent loop, shell execution tools or image/video input. Current bot help/status messages are in Chinese; the model can answer in other languages.

## Pinned model and runtime

- Model: `orcarouter/Qwen3.8-27B-Uncensored`.
- Revision: `8cb32d72080f6a47bf34dc5adf8067daa6c63a31`.
- Precision: original BF16 SafeTensors; no quantization or model/GPU fallback.
- GPU: one `A100-80GB`.
- vLLM: `0.31.0`, using the official image pinned by digest in `modal_app.py`.
- Context: 4096 tokens total, with up to 512 generated tokens.
- `model-manifest.json` lists 18 weight shards and 8 required small files, with sizes and checksums. It contains no weights.

This is deliberately a **fixed-model deployment**, not a universal model loader. `model_id` and `model_revision` in the local config validate the pinned target. Switching models requires updating `constants.py`, the manifest, the exact file allowlist in `weights.py`, and any model-specific validation/tests together. Merely changing a TOML field is not supported.

## Prerequisites

- A Linux CPU host with Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). No local GPU is needed. The current implementation uses POSIX file locking.
- Your own new Telegram bot from [@BotFather](https://t.me/BotFather), with no existing webhook or other polling process.
- Your own confirmed private Telegram chat and sender IDs. Obtain them from an update to your bot using Telegram's [`getUpdates`](https://core.telegram.org/bots/api#getupdates) before starting polling; do not authorize by display name.
- A Modal account/workspace with access to A100-80GB and sufficient provider limits.
- Your own Hugging Face account with access to the gated model, and a read token.

## Install and configure

```bash
git clone https://github.com/jimtechfellow/telegram-modal-llm-bot.git
cd telegram-modal-llm-bot
uv sync --frozen --extra dev --python 3.11
source .venv/bin/activate

install -d -m 700 "$HOME/.config/modal_a100_chat"
cp config.example.toml "$HOME/.config/modal_a100_chat/config.toml"
chmod 600 "$HOME/.config/modal_a100_chat/config.toml"
```

Edit the copied file, not the published example:

- Replace **both** zero ID placeholders with your own confirmed private chat/user IDs.
- Set `modal_workspace` to your actual workspace and select `modal_environment`.
- Choose a private `state_dir` outside the checkout and your own `system_prompt`.
- Leave the pinned model/revision unchanged for the supported deployment.

The Python package and CLI retain their existing names: `modal_a100_chat`, `modal-a100-chat` and `modal-a100-admin`.

### Credentials

Authenticate the official SDK using `modal setup`, or inject `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` through your secret manager. Protect the SDK's native credential file if one is created.

Inject `TELEGRAM_BOT_TOKEN` into the bot process environment. For deployment, inject `HF_TOKEN` into the admin process environment; deployment copies it into this app's named Modal Secret, which is attached only to the CPU download function.

`.env.example` documents the variable names. **The code does not automatically load `.env` files.** Keep real credentials out of Git, shell history, command-line arguments and logs. Use your own runtime secret-injection mechanism; no host-specific credential wrapper is required or included.

For a non-default Modal environment, set `MODAL_ENVIRONMENT` in the process environment to the same value as `modal_environment` in the config before importing/deploying the app.

The default resource names are `modal-a100-chat` (App), `modal-a100-chat-qwen38-bf16` (Volume), and `modal-a100-chat-hf` (Secret). **Use a separate workspace/environment or change the names in the source when deploying alongside an existing installation.** `deploy` can update resources with these names, including the HF Secret; it is not a harmless read-only command.

## Deploy

Run from the repository checkout with the virtual environment activated and credentials supplied at runtime:

```bash
# Local configuration inspection only; does not allocate a GPU.
modal-a100-admin local-check

# Creates/updates this project's Modal resources and builds images.
modal-a100-admin deploy --execute

# Downloads and verifies the pinned files on Modal CPU, then commits the Volume.
modal-a100-admin download --execute

# Start long polling on the CPU host; normal messages will start GPU inference.
modal-a100-chat
```

Both CLIs accept `--config /path/to/your/config.toml`. The supported setup runs from an editable source checkout (`uv sync`), because deployment reads the manifest and uploads local package source. A standalone wheel install is not the documented deployment method.

Use your process manager and runtime secret injection if you want the polling process to survive logout or host restart. Production-specific installers and service definitions are intentionally not shipped. Do not run multiple pollers for the same Telegram bot token.

Deployment/image builds and CPU downloads can consume billable provider resources. A successful `local-check` only verifies local settings; it is not evidence that the model loaded on a GPU.

## Usage and shutdown checks

Send ordinary text **without a leading slash** to request an answer. `/start` and other unsupported inputs return a short help message and do not invoke inference.

- `/status`: reports the fixed model, local job state, GPU container count, running inputs and backlog.
- `/stop`: requests cancellation, terminates the remote container and confirms all remote counters are zero before reporting release.
- An initial cancellation acknowledgement is not the final shutdown confirmation. Wait for the final confirmation, then check `/status`.
- `Can submit: True` is not proof of release. Verify **containers = 0, running requests = 0, backlog = 0**.
- A Modal dashboard **Live App** can remain deployed while its **Containers: 0 live** and **Calls: 0 running** counters show no active compute.

The admin CLI also provides `status`, `cancel-admin`, `stop-app` and `smoke`. Mutating operations require `--execute`; **`smoke --execute` allocates a GPU**. Do not invoke it as part of an offline source check.

SQLite history is stored at `state_dir/history.sqlite3`. Completed turns survive reopening the database; cancelled turns do not enter conversation history. If the bot restarts, it handles recorded unfinished calls instead of replaying inference. A spawn whose call ID was never durably recorded fails closed and requires scoped operator recovery. Do not delete durable records while a remote task may still exist.

## Costs, cold starts and privacy

Single-use containers trade idle GPU cost for startup latency: **each inference starts a new container and reloads weights from the Volume**. Cached weights avoid downloading from Hugging Face again, but do not keep the model resident in GPU memory. Responses may take minutes during cold startup. Model loading time also contributes to compute usage.

Scale-to-zero is not a budget cap or a promise of free usage. GPU/CPU compute, memory, model storage and other provider charges may apply. Configure budgets and spending limits in your own Modal account and consult [current Modal pricing](https://modal.com/pricing). A request timeout or cancellation acknowledgement alone does not prove compute has stopped.

Messages pass through Telegram. The bot stores user text/history on its host, and sends the current prompt plus selected history to Modal for inference. Inference uses offline weights with network access blocked; there are no search tools. This is not an end-to-end encrypted or local-only inference system. Keep the state directory and backups private; an uncensored model may produce unsafe or inaccurate output.

## Offline tests

```bash
uv run --frozen ruff check src tests
uv run --frozen pytest -q
uv run --frozen modal-a100-admin local-check
```

The tests use synthetic data and local SDK doubles for authorization, SQLite persistence, deduplication, cancellation/release handling, model-manifest validation, context trimming and credential-bearing exception redaction. Network access is blocked in the pytest process. They do **not** deploy resources, download real weights, run a GPU, verify current billing or substitute for an end-to-end Telegram deployment test.

The source-publication checks are intentionally offline; GPU retesting is not part of this release.

## License

Original application code is licensed under [MIT](LICENSE). Third-party libraries, the referenced container image and model weights retain their own licenses and access conditions. In particular, `python-telegram-bot` is LGPL-3.0-only, not MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the checked primary dependency licenses and redistribution boundaries.

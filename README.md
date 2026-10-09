# telegram-modal-llm-bot
connect telegram bot to modal llm

A private, single-user Telegram text-chat bot backed by an on-demand Modal GPU function. This repository packages the existing Python implementation without adding a search engine or an agent framework.

## Architecture and scope

```text
Telegram user
    ↕ Telegram Bot API (long polling)
Linux CPU host: Python bot + SQLite history
    ↕ authenticated Modal Python SDK function calls
Modal: reusable A100-80GB container + cached vLLM engine
    ↕ read-only Modal Volume
Pinned Qwen3.8-27B-Uncensored BF16 weights
```

- Long polling: no inbound webhook server or extra public inference port.
- Authorization checks both the private chat ID and sender ID before submitting inference.
- Plain-text messages invoke the model; `/status` reads remote counters and `/stop` cancels the exact remote call with container termination.
- Completed user/assistant turns are stored in SQLite on the CPU host. Recent history is resent with each request and trimmed to fit the context window.
- Results are sent after generation completes, not streamed token by token.
- One inference at a time. Remote retries are disabled. Ambiguous submissions or unconfirmed release block further requests instead of silently resubmitting.
- The first message starts a GPU container and loads the model. Subsequent messages reuse its engine while the container remains alive. `min_containers=0`, `buffer_containers=0`, `max_containers=1` and a finite `scaledown_window` let Modal automatically scale back to zero without a local timer or keepalive requests.
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
- `gpu_idle_timeout_seconds` defaults to 300 seconds (5 minutes); allowed values are integers from 2 to 1200. Older configs without this field use 300. Idle GPU residency is billable.
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

- `/status`: reports the fixed model, local job state, configured/recently confirmed idle window, GPU container count, running inputs and backlog. It distinguishes warm idle residency from released compute. It never invokes inference or sends keepalives.
- `/stop`: cancels the exact running call with container termination. When the GPU is already idle, it requests release by shortening this function's native idle window to 2 seconds, without launching another GPU call. Actual release is confirmed only when all three counters are zero; shutdown confirmation waits up to 90 seconds, after which new jobs remain blocked. The configured window is restored before the next request.
- An initial cancellation acknowledgement is not the final shutdown confirmation. Wait for the final confirmation, then check `/status`.
- `Can submit: True` is not proof of release. Verify **containers = 0, running requests = 0, backlog = 0**.
- A Modal dashboard **Live App** can remain deployed while its **Containers: 0 live** and **Calls: 0 running** counters show no active compute.

The admin CLI also provides `status`, `cancel-admin`, `stop-app` and `smoke`. Mutating operations require `--execute`; **`smoke --execute` allocates a GPU**. Do not invoke it as part of an offline source check.

The bot and mutating admin commands share `state_dir/bot.lock`; stop the bot before deployment or administrative GPU tests. They must use the same `state_dir` and dedicated Modal function/environment. Inference is serial, remote retries are disabled, and cancellation never uses workspace-wide or app-wide shutdown. Unknown running/queued calls block idle `/stop` instead of being cancelled. External clients bypassing these locks are outside the supported single-user setup.

SQLite history is stored at `state_dir/history.sqlite3`. Completed turns survive reopening the database; cancelled turns do not enter conversation history. If the bot restarts, it handles recorded unfinished calls instead of replaying inference. A spawn whose call ID was never durably recorded fails closed and requires scoped operator recovery. Do not delete durable records while a remote task may still exist.

Completion and GPU release are separate: an ordinary completed turn can have `release_confirmed=0` while its container remains warm, and does not block the next turn. Explicit release requests that remain unconfirmed are durable and do block new turns, including after restart. Existing SQLite databases remain compatible; no history rewrite or new table is required. GPU evidence stored with each completed job includes `container_id`, `engine_instance_id`, `engine_reused` and `engine_load_seconds`.

## Costs, cold starts and privacy

The first request after scale-to-zero loads weights from the Volume; later requests reuse the cached engine in the same container. After the last inference, Modal releases idle compute using `scaledown_window`. This is a maximum idle policy, not a guarantee of 300 seconds of residency: platform recycling or preemption can cause an earlier cold start. See [Modal cold-start and idle-window documentation](https://modal.com/docs/guide/cold-start). No permanently warm pool or heartbeat is configured, and the native policy still works if the bot process crashes.

At the published base rates checked on 2026-10-09, A100 80GB costs $0.000694/second (about $2.4984/hour). A full 300-second idle tail therefore costs about **$0.2082 for the GPU**, or **$0.2879 including the current 4-core CPU and 96 GiB memory requests**, before credits, taxes, regional modifiers or excess resource usage. This tail is per continuous warm session, not a fixed charge per message: gaps between turns are also billable. Model loading and generation add compute charges. Frequent chats avoid repeated loading; sparse chats can cost more than single-use containers. Consult [current pricing](https://modal.com/pricing) and [CPU/memory billing](https://modal.com/docs/guide/resources#billing) for actual charges.

Scale-to-zero is not a budget cap or a promise of free usage. GPU/CPU compute, memory, model storage and other provider charges may apply. Configure budgets and spending limits in your own Modal account and consult [current Modal pricing](https://modal.com/pricing). A request timeout or cancellation acknowledgement alone does not prove compute has stopped.

Messages pass through Telegram. The bot stores user text/history on its host, and sends the current prompt plus selected history to Modal for inference. Inference uses offline weights with network access blocked; there are no search tools. This is not an end-to-end encrypted or local-only inference system. Keep the state directory and backups private; an uncensored model may produce unsafe or inaccurate output.

## Offline tests

```bash
uv run --frozen ruff check src tests
uv run --frozen pytest -q
uv run --frozen modal-a100-admin local-check
```

The tests use synthetic data and local SDK doubles for authorization, SQLite persistence, deduplication, cancellation/release handling, model-manifest validation, context trimming and credential-bearing exception redaction. Network access is blocked in the pytest process. They do **not** deploy resources, download real weights, run a GPU, verify current billing or substitute for an end-to-end Telegram deployment test.

The reuse, lifecycle, concurrency and recovery tests use synthetic engines and remote counters. Passing them proves the local behavior, not real GPU reuse, provider shutdown timing or billing.

## Upgrade and real GPU acceptance (not executed by offline development)

After separate authorization for production deployment and paid GPU testing:

1. Save the old checkout reference and process configuration. Stop the existing bot and confirm its remote counters are zero. Back up the private TOML config and SQLite using SQLite's backup API; retain the state directory and verified model Volume.
2. Install this branch in a separate checkout with `uv sync --frozen --extra dev --python 3.11`. Use the existing private config and credentials. Set `gpu_idle_timeout_seconds = 300`, or rely on the default.
3. Run `modal-a100-admin --config /path/to/config.toml deploy --execute` from the new environment. It verifies reusable serial inference, zero minimum/buffer containers, a one-container maximum and the configured idle window. The existing verified model Volume can be reused without another download. Deploy may build images/update this app's Secret and is a separate authorized operation.
4. Point the bot process at the new checkout/environment and start it. The new bot refuses an old single-use or concurrent deployment instead of silently running with mismatched lifecycle controls.
5. Send two different messages within the idle window. Check the SQLite job evidence: both `container_id` and `engine_instance_id` must match, and the second result must have `engine_reused=true`. Confirm four new history messages and only one GPU container.
6. Send no further inference requests. Poll `/status` or admin `status` (neither allocates compute) and verify all three counters reach zero after the idle window, allowing a bounded observation margin of 90 seconds. If they do not, record the test as failed/unconfirmed; do not declare release. The next message should use a new container/engine.
7. Test `/stop` both during inference and during warm idle time, including immediately after submission. Require confirmed zero counters, no cancelled turn in history, and a successful new cold request afterward.

Real GPU reuse, automatic release, active/idle cancellation and actual costs remain **NOT_RUN** until these checks are performed. Rollback requires stopping the new bot, confirming release, restoring the prior checkout/config/process path, and redeploying the baseline Modal code before restarting the old bot. Preserve the SQLite history and Volume; no data deletion is needed.

## License

Original application code is licensed under [MIT](LICENSE). Third-party libraries, the referenced container image and model weights retain their own licenses and access conditions. In particular, `python-telegram-bot` is LGPL-3.0-only, not MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the checked primary dependency licenses and redistribution boundaries.

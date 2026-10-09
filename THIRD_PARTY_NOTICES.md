# Third-party licenses

The repository's MIT license applies to its original application code and documentation. The application was developed for the repository owner; it is not a fork containing vendored library source. Dependencies are imported or installed separately and retain their own licenses. The existing repository owner's MIT copyright notice is preserved.

Primary runtime components checked for this source release:

- **Modal Python SDK 1.6.1**: Apache-2.0. [Package metadata](https://pypi.org/project/modal/1.6.1/) and [source](https://github.com/modal-labs/modal-client).
- **python-telegram-bot 22.8**: LGPL-3.0-only. [License](https://github.com/python-telegram-bot/python-telegram-bot/blob/v22.8/LICENSE.lesser) and [source](https://github.com/python-telegram-bot/python-telegram-bot). The bot uses the library's public interfaces without vendoring or modifying its source. Users may inspect and replace the separately installed library. MIT licensing of this application's original source does not relicense the library. Redistributors of bundled dependencies or combined binaries must also comply with the LGPL, including the applicable notices, source availability and library-replacement requirements.
- **vLLM 0.31.0**: Apache-2.0. [License](https://github.com/vllm-project/vllm/blob/v0.31.0/LICENSE). The official image is referenced by digest, not included in this repository; its other components, including GPU drivers and CUDA, have their own terms.
- **huggingface-hub 2.1.1**: Apache-2.0. [Package metadata](https://pypi.org/project/huggingface-hub/2.1.1/) and [source](https://github.com/huggingface/huggingface_hub). Installed only in the Modal CPU download image.

Development tools: pytest and Ruff use MIT; pytest-asyncio uses Apache-2.0. The installed locked host dependencies were also checked through their package metadata: the remaining licenses are MIT, Apache-2.0, BSD-2-Clause/BSD-3-Clause, PSF-2.0, and MPL-2.0 (certifi). These terms remain attached to their respective packages; in particular, certifi's covered files are not relicensed as MIT. No dependency wheels, container images or third-party library source are redistributed here.

## Model weights

The pinned [orcarouter/Qwen3.8-27B-Uncensored model](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) declares `apache-2.0` in its Hugging Face metadata at revision `8cb32d72080f6a47bf34dc5adf8067daa6c63a31` and uses gated access. Obtain access through your own Hugging Face account and review the upstream model and base-model terms. This repository contains only file names, sizes and checksums for reproducible downloads, not model weights. The MIT application license does not grant model access or replace upstream conditions.

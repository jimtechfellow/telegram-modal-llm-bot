"""Pinned standalone pi-web-access; paid Exa only, no Pi runtime or free MCP path.

Upstream contract: https://github.com/nicobailon/pi-web-access#use-from-other-agents-mcp
Client lifecycle: https://py.sdk.modelcontextprotocol.io/client/transports/#stdio
"""

import asyncio
import ipaddress
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urldefrag, urlsplit

from mcp import Client, StdioServerParameters, stdio_client

from .service import ResearchCleanupError, ResearchError, SearchEvidence, Source

PI_WEB_ACCESS_VERSION = "0.37.0"
CONFIG_DIR = Path(__file__).parent / "mcp_config"
DEFAULT_COMMAND = Path(__file__).resolve().parents[2] / "mcp/node_modules/.bin/pi-web-access-mcp"
ARTIFACT_LIMIT = 64000
CLEANUP_TIMEOUT_SECONDS = 10


async def _wait_for_search(worker, closing, timeout):
    # asyncio.wait never propagates cancellation into the worker. One task must own
    # both entry and exit of the SDK's AnyIO scopes, with at most one native cancel.
    failure = None
    try:
        done, _ = await asyncio.wait({worker}, timeout=timeout)
        if worker not in done:
            failure = TimeoutError()
    except asyncio.CancelledError as exc:
        failure = exc
    if failure is not None:
        # A normal/error exit may already be cleaning up: do not interrupt it either.
        if not worker.done() and not closing.is_set():
            worker.cancel()
        deadline = asyncio.get_running_loop().time() + CLEANUP_TIMEOUT_SECONDS
        while not worker.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                # Never send a second cancel into SDK cleanup or claim it exited.
                worker.add_done_callback(lambda t: None if t.cancelled() else t.exception())
                raise ResearchCleanupError("搜索子进程退出未确认，已阻止新任务。") from None
            try:
                await asyncio.wait({worker}, timeout=remaining)
            except asyncio.CancelledError as exc:
                failure = exc  # Stop takes precedence, without restarting the cleanup budget.
        try:
            worker.result()  # Retrieve any exception before propagating stop/timeout.
        except ResearchCleanupError:
            raise
        except (asyncio.CancelledError, Exception):
            pass
        raise failure
    return worker.result()


def _source_url(raw):
    if not isinstance(raw, str) or not raw or len(raw) > 2048 or re.search(r"[\s\x00-\x1f\x7f]", raw):
        raise ValueError
    parsed = urlsplit(raw)
    host = parsed.hostname
    if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password:
        raise ValueError
    if "." not in host or host.lower().endswith((".local", ".internal", ".localhost")):
        raise ValueError
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError
    return urldefrag(raw)[0]


def parse_artifact(text: str, limit: int) -> SearchEvidence:
    try:
        if len(text) > ARTIFACT_LIMIT:
            raise ValueError
        data = json.loads(text)
        if data["type"] != "research" or data["provider"] != "exa" or data.get("errors"):
            raise ValueError
        summary, raw_sources = data["summary"], data["sources"]
        if not isinstance(summary, str) or not summary.strip() or not isinstance(raw_sources, list):
            raise ValueError
        if not 1 <= len(raw_sources) <= limit:
            raise ValueError
        sources, seen = [], set()
        for raw in raw_sources:
            url = _source_url(raw["url"])
            title = raw["title"]
            if not isinstance(title, str) or not title.strip():
                raise ValueError
            if url not in seen:
                seen.add(url)
                sources.append(Source(" ".join(title.split())[:100], url))
        return SearchEvidence(tuple(sources), summary)
    except (ValueError, TypeError, KeyError):
        raise ResearchError("搜索失败或来源数据无效；仅允许付费 Exa，不回退、不生成引用。") from None


def _tool_text(result):
    if result.is_error or not result.content or any(block.type != "text" for block in result.content):
        raise ResearchError("搜索 MCP 返回错误；没有回退其他搜索服务。")
    text = "\n".join(block.text for block in result.content)
    if len(text) > ARTIFACT_LIMIT:
        raise ResearchError("搜索响应超出预算；不会追加请求或启动 GPU。")
    return text


class PiWebAccess:
    def __init__(self, command=None, timeout_seconds=60, num_results=5):
        if type(num_results) is not int or not 1 <= num_results <= 5:
            raise ValueError("Search result limit must be 1..5")
        if type(timeout_seconds) is not int or not 5 <= timeout_seconds <= 90:
            raise ValueError("Search timeout must be 5..90 seconds")
        self.command = str(command or DEFAULT_COMMAND)
        self.timeout_seconds, self.num_results = timeout_seconds, num_results

    async def search(self, topic: str) -> SearchEvidence:
        api_key = os.environ.get("EXA_API_KEY", "").strip()
        if not api_key:
            raise ResearchError("搜索不可用：未配置 EXA_API_KEY；不会使用免费搜索或启动 GPU。")
        params = StdioServerParameters(command=self.command, env={
            "EXA_API_KEY": api_key,
            "EXA_BASE_URL": "https://api.exa.ai",
            "PI_CODING_AGENT_DIR": str(CONFIG_DIR),
        })
        closing = asyncio.Event()

        async def run():
            # Never pass parent secrets, arbitrary NODE_OPTIONS, or child stderr through.
            with open(os.devnull, "w") as errlog:
                @asynccontextmanager
                async def transport():
                    context = stdio_client(params, errlog=errlog)
                    streams = await context.__aenter__()
                    try:
                        yield streams
                    finally:
                        closing.set()
                        try:
                            # Preserve the body's exception at yield, rather than injecting
                            # it into the SDK TaskGroup and misclassifying its rewrapped
                            # ExceptionGroup as a transport cleanup failure.
                            await context.__aexit__(None, None, None)
                        except (asyncio.CancelledError, Exception):
                            raise ResearchCleanupError("搜索子进程退出未确认，已阻止新任务。") from None

                async with Client(transport()) as client:
                    try:
                        info = client.server_info
                        if info is None or (info.name, info.version) != (
                            "pi-web-access", PI_WEB_ACCESS_VERSION
                        ):
                            raise ResearchError("搜索 MCP 名称或版本不匹配；未发送搜索请求。")
                        result = await client.call_tool("source_check", {
                            "claim": topic, "queries": [topic], "provider": "exa",
                            "numResults": self.num_results, "fetchContent": False, "proxy": "",
                        })
                        ids = re.findall(
                            r"^Artifact responseId: ([A-Za-z0-9_-]+) \(", _tool_text(result), re.M
                        )
                        if len(ids) != 1:
                            raise ResearchError("搜索 MCP 未提供唯一的来源记录；不生成报告。")
                        result = await client.call_tool("get_search_content", {
                            "responseId": ids[0], "offset": 0, "limit": ARTIFACT_LIMIT,
                        })
                        return parse_artifact(_tool_text(result), self.num_results)
                    finally:
                        closing.set()
        try:
            return await _wait_for_search(
                asyncio.create_task(run()), closing, self.timeout_seconds
            )
        except (ResearchError, ResearchCleanupError):
            raise
        except Exception:
            raise ResearchError("付费 Exa 搜索失败或超时；未回退其他服务、未自动重试。") from None

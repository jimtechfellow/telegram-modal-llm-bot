# 模块化 Search Bot

基于已合并的 GPU 复用版本 `59d8f20`，默认保留 300 秒 GPU 空闲窗口。
本次开发未修改生产服务、部署 Modal、启动 GPU 或调用真实付费 Exa。

## 边界

```text
Telegram（鉴权、收发、状态）
  └─ Runner（单任务、SQLite、取消、恢复）
       ├─ 普通聊天 → 原 Modal Qwen 推理
       └─ 调研 → search_bot.SearchBot
                   ├─ SearchProvider → 独立 pi-web-access MCP → 付费 Exa API
                   └─ ReportModel → LifecycleReportModel → 原 Runner 推理生命周期
```

`search_bot` 不依赖 Telegram、Modal 或 Qwen；`SearchProvider`、`ReportModel` 是 Python Protocol。
替换模型只需提供 `async generate(messages) -> str` 的适配器，不改研究业务模块。
当前适配器复用已有 GPU 提交、持久化 call ID、轮询和取消逻辑；没有另写 GPU 管理器。
没有安装 Pi 运行时、模型热切换或通用 Agent 框架。

## 安装与配置（需另行授权才切换生产）

在独立开发检出目录中安装，Node >=22.19（本次验证 Node 24.21.0 / npm 11.19.0）：

```bash
uv sync --frozen --extra dev --python 3.11
npm ci --prefix mcp --ignore-scripts --omit=peer --omit=optional
uv run --frozen pytest -q
uv run --frozen ruff check .
```

`mcp/package.json` / `mcp/package-lock.json` 固定独立服务 `pi-web-access@0.37.0`，不通过 `npx`
临时下载或运行最新版本。安装脚本关闭，Pi 的可选 peer 依赖不安装。
`uv.lock` 固定官方 Python MCP SDK 2.3.0；不升级现有 Modal、PTB 或 GPU/vLLM 配置。

真实运行前，向 **CPU Bot 进程的环境**安全注入现有付费 `EXA_API_KEY`；不要写进 TOML、
命令行参数、源码或提示词。没有密钥时普通聊天仍可用，但研究明确失败，不启动 GPU。
不需要向 Modal Secret 添加 Exa 密钥。

非敏感 TOML 可选项：

```toml
search_timeout_seconds = 60 # 5..90，包含 MCP 启动、搜索和来源读取
search_max_results = 5      # 1..5
# search_mcp_command = "/absolute/path/to/pi-web-access-mcp"
```

默认可执行文件为此检出目录的 `mcp/node_modules/.bin/pi-web-access-mcp`。
通过 wheel 安装或将 MCP 独立安装在其他目录时，显式设置可执行文件绝对路径；
命令不是 shell 字符串，不支持夹带参数。服务名称与版本必须匹配，否则搜索前拒绝。
SDK 按需启动独立 stdio 子进程，退出、超时和取消时清理，不新增常驻 systemd 服务。

子进程使用本包专用 `mcp_config/web-search.json`，不读取用户已有 Pi provider 配置。
配置仅开放 `source_check` 和 `get_search_content`，白名单只有 Exa；
密钥必须来自 `${EXA_API_KEY}`，API 目的地固定为 `https://api.exa.ai`。
每个请求仍显式指定 `provider="exa"`，不允许 `auto`，不传递其他凭据或任意 `NODE_OPTIONS`。
子进程 stderr 丢弃，返回用户的错误不包含原始异常、密钥或 provider 错误响应。

## 研究行为与费用上限

输入 `调研 GPU 容器复用`，仅主题发送给 Exa；已有私人聊天历史不会发送给搜索服务。
流程固定为一次检索、来源整理、一次模型生成、引用校验和中文报告排版。

| 项目 | 约束 |
| --- | --- |
| 输入 | `调研` 后有空白分隔；主题 1–300 字 |
| Exa | 一次 `/search` POST、最多 5 条结果；无重试、无递归搜索 |
| MCP | 一次 `source_check(fetchContent=false)`、一次本地来源记录读取 |
| 页面抓取 | 不调用 `fetch_content`，不抓取网页、克隆仓库或启动浏览器 |
| 来源响应 | 最多 64,000 字符；无分页追加，超限明确失败 |
| 模型上下文 | 主题、来源元数据、检索摘要合计最多 2,400 字符，另加固定指令 |
| 模型调用 | 最多一次；原 4,096 token 上下文、512 token 输出仍保留 |
| 模型报告 | JSON 结构；1–3 条发现及局限，报告引用由代码生成 |

摘要来自 Exa 返回的 highlights，不是独立抓取全文；资料不足时应明确说明。
模型只在结构化 `sources` 字段返回来源编号；代码拒绝越界编号、模型自写 URL/域名，
正文及局限中的含数字方括号（单个、组合、区间编号）也一律拒绝，再由代码附上登记的来源 URL。
不合格 JSON、空结果、搜索错误或超过预算均明确失败，不自动再次搜索/生成报告。
超出实际 tokenizer token 预算仍由原 GPU 推理路径拒绝并执行取消/释放，不会扩大模型上下文。

这是**单次任务调用数量限制，不是账户美元硬限额**。Exa 的具体费用由付费套餐决定；
超时/取消也可能已经产生一次 Exa 请求费用，不能保证退款。研究任务失败不会自动重试。
模型生成计费与普通聊天一致，包括冷启动加载和生成后的 300 秒暖容器空闲费用。
检索本身不提交 GPU 输入；如果上一轮 GPU 尚暖，其原生空闲回收仍继续生效。

## 单任务、取消与恢复

- 一条消息先用现有 `update_id` 在 SQLite 去重，然后才发出付费检索；普通聊天与研究不并发。
- 检索阶段持久化为 `searching`，尚无 GPU call ID；`/stop` 取消检索子任务，并确认已有 GPU 归零。
  独立任务拥有完整 MCP 连接生命周期；搜索截止和停止仅向其发送一次取消，清理开始后不再取消。
  重复 `/stop` 或其他取消不会打断清理；额外等待清理最多 10 秒，不因重复取消延长。
  等待连接退出清理完成后才报告取消；清理报错或超过等待时间时持久化为未确认取消，
  阻止新任务，`/stop` 和重启恢复均不会将其误当成已退出，需管理员核查本地 MCP 子进程。
- 开始模型提交前先持久化为 `starting`；从此缺失 call ID 仍按原逻辑 fail-closed，不能误当成本地检索恢复。
- 生成阶段 `/stop` 只取消已记录的确切 GPU call，仍等待实际释放确认。
- 检索阶段重启不重放搜索；无未知远程请求时确认释放并将该轮取消，允许后续聊天。
- 释放未确认、未知远程任务或未解决的 `admin-call.json` 继续阻止新任务；不删除管理员记录。
- 仅通过引用校验的完整报告进入 SQLite 历史；失败、取消报告不进入历史。
- `/status` 显示业务阶段（检索中/生成报告/普通聊天）和原 GPU 状态；状态查询不搜索或保活。

## 独立测试和可替换模型

```python
from search_bot.service import SearchBot
from search_bot.pi_web_access import PiWebAccess

# model 仅需实现 ReportModel.generate；可换成离线 fake，无需 Telegram 或 Modal。
# 此处只有调用 research 才发出付费请求，请在有授权的环境执行。
bot = SearchBot(PiWebAccess())
report = await bot.research("GPU 容器复用", model)
```

离线测试使用搜索/模型替身，另有真实固定版本 pi-web-access 的 stdio 集成探针：
仅将最终 HTTP 边界替换为合成响应，同时禁止子进程所有真实网络连接。
探针要求付费 `/search` 目的地、`x-api-key` 和一次请求；测试成功、429 不回退及超时清理。
回归覆盖停止与截止重合、清理期间再次取消、正常退出期间停止/到期、清理报错与有界等待；
真实离线子进程测试在退出后检查 `returncode`，并验证清理未确认时持久化阻塞不被恢复绕过。
这不是付费 Exa 或真实 Qwen 的端到端验收，不声称生产 Telegram 网络收发已通过。

## 已知风险

1. 真正的 Exa 凭据/网络与 Qwen 的短 JSON 遵循能力尚未联合实测。输出格式不合格会明确失败，
   不放宽引用规则或增加重试；512 token 限制意味着报告偏简短。
2. 来源登记和编号校验保证 URL 来自搜索返回结果，**不能证明每条生成结论被引用资料语义支持**，
   也不能消除提示词注入、源站错误或过期资料。重要结论需要人工核实原文。
3. 上游 `pi-web-access` 固定的 TypeScript SDK 1.27.1 引发 npm audit 两项 high 标记，
   根因同为 [GHSA-6qxp-vccf-f47h](https://github.com/advisories/GHSA-6qxp-vccf-f47h)：
   OAuth **客户端**可能信任 MCP 服务指定的授权服务器。本实现只使用本地 stdio **服务端**
   和固定 Exa HTTPS API，无 OAuth 客户端路径，该告警当前不可达；未强制升级或改写上游 bundle。
   核查日期 2026-10-09；下次更新 MCP 依赖或引入 HTTP/OAuth 前必须复核。
   本次 `npm audit signatures` 验证 193 个包的 registry signatures，13 个包的 attestations。
4. 原 GPU 版本已接受的限制仍存在：运行中停止可能记录 `failed`，以三项远程计数归零判断释放。
5. 运行时子进程退出依赖固定版 SDK 的关闭、限时等待、kill/reap 机制；本轮没有自建进程管理器。
   离线回归已确认上述取消竞态下实际子进程退出，不声称覆盖内核无法终止进程等系统级故障。

## 官方依据

- [pi-web-access 独立 MCP 使用说明](https://github.com/nicobailon/pi-web-access#use-from-other-agents-mcp)
- [固定版本 Exa 路由源码（发布 tag）](https://github.com/nicobailon/pi-web-access/blob/v0.37.0/exa.ts)
- [官方 Python MCP SDK 客户端](https://py.sdk.modelcontextprotocol.io/client/)
- [官方 stdio 生命周期与环境隔离](https://py.sdk.modelcontextprotocol.io/client/transports/#stdio)

## 回退

本轮仅在 `feat/search-mcp` 独立 Worktree 开发，没有生产配置/服务变更，无需生产回滚。
未来启用搜索后，可退回原 `master` GPU 复用检出目录及其 Python 环境；不删除 SQLite 或模型 Volume。
部署前应先停止 Bot 并确认远程归零，保留 SQLite 备份；若有 `searching` 记录，先用本版完成恢复，
再退回旧版，避免旧版不识别该状态。没有新增数据库表或修改现有表结构。

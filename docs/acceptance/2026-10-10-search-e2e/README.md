# Search Bot 真实端到端验收：PASS

2026-10-10 UTC，验收代码为 `feat/search-mcp` 的 `5e75df857f94a1d0a7d60097b267925a26d69b09`。

- 现有 `create_runner → Runner.start` 完成一次付费 Exa 搜索、一次 Qwen3.8 BF16 推理，无重试；5 个来源，输出 185 token，低于 512 token 限制。
- 中文报告、结构化输出、来源编号及报告 URL 属于搜索返回集合的校验通过。未独立核验来源全文或每条结论的语义支持。
- 独立 SQLite 中 1 条任务为 `completed`、`release_confirmed=1`，2 条历史消息持久化；assistant 内容与归档报告一致。未提交数据库或真实私人聊天。
- 300 秒原生空闲策略下，约 357.75 秒后 GPU 容器、运行请求、排队全部为零；随后测试 App 停止、容器为空，已有模型 Volume 保留。
- 服务商报告的用量费用为 **$0.53545521**：Exa 响应费用 $0.007，Modal 账单 $0.52845521，低于 $2。账单延迟入账后于 12:11:47 UTC 核对稳定；额度抵扣、税费和最终结算以服务商为准。

临时部署入口原先使用 `serialized=True`，导致本地 Python 3.11 与 GPU 镜像 Python 3.12 不兼容；仅改为可导入的推理入口后部署成功。未修改业务代码、重新下载模型、切换生产服务或重跑离线测试。Telegram 网络收发不在本次范围内。

## 证据

- [result.json](result.json)：原始验收结果的脱敏归档，保留真实金额、调用次数及回收状态。
- [evidence.json](evidence.json)：Exa 响应、模型证据、报告校验、SQLite 最终只读回查、远程归零与停止记录、Modal 费用明细。
- [report.md](report.md)：实际生成的中文报告；仅补文件末尾换行，正文与持久化历史一致。`evidence.json` 的 SHA-256 对应原始无末尾换行内容。

生产模型 Volume ID、临时绝对路径已脱敏；未归档 API Key、令牌、凭据记录 UUID、HTTP 认证头或执行脚本。保留的测试 App/Function/Call ID 仅用于证据关联，不是访问凭据。

剩余阻塞：无（仅针对上述验收范围）。本次归档不代表已部署 Search Bot 到生产。

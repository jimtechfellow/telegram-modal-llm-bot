// Real pinned pi-web-access MCP server; only its final HTTP boundary is replaced.
// All other sockets are forbidden, including the free Exa MCP and other providers.
import net from "node:net";
import tls from "node:tls";

function forbidden() { throw new Error("Network forbidden in offline MCP probe"); }
net.connect = net.createConnection = tls.connect = forbidden;
let requests = 0;
globalThis.fetch = async (url, options) => {
  if (++requests !== 1 || String(url) !== "https://api.exa.ai/search") forbidden();
  const headers = new Headers(options.headers);
  if (headers.get("x-api-key") !== "unit-only-exa-key") forbidden();
  const body = JSON.parse(options.body);
  if (body.numResults !== 5 || body.contents.text || !body.contents.highlights) forbidden();
  if (body.query === "模拟失败") return new Response("unit-only-exa-key", { status: 429 });
  if (body.query === "模拟超时") return new Promise(() => {});
  return new Response(JSON.stringify({ results: [{
    title: "官方生命周期文档",
    url: "https://docs.example.org/lifecycle",
    highlights: ["官方资料说明容器可在空闲窗口内复用，超时后释放。"]
  }] }), { headers: { "Content-Type": "application/json" } });
};
await import("../../mcp/node_modules/pi-web-access/dist/mcp-cli.js");

# Using the MCP server

The server is built with the official [`mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(`MCPServer`, v2 API) and exposes three **read-only** tools:

| Tool | Args | Returns |
|---|---|---|
| `list_markets` | `limit=20` | Top active binary Polymarket markets (id, question, YES price, volume) |
| `search_evidence` | `market_id`, `query?`, `k=8` | Hybrid (vector + BM25) retrieved news snippets with URLs and dates |
| `research_market` | `market_id`, `as_of?` | Agent forecast: P(YES), rationale, evidence for/against, market price, edge |

## Claude Desktop / Cursor (stdio)

```json
{
  "mcpServers": {
    "pm-research": {
      "command": "/absolute/path/to/.venv/bin/pmra",
      "args": ["mcp"],
      "env": { "PMRA_SOURCES": "google_news", "PMRA_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

## Streamable HTTP

* Standalone: `pmra mcp --transport streamable-http --port 8765` → `http://localhost:8765/mcp`
* Mounted in the API: `pmra serve` → `http://localhost:8080/mcp/`

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with streamable_http_client("http://localhost:8080/mcp/") as (r, w, *_):
    async with ClientSession(r, w) as s:
        await s.initialize()
        print(await s.call_tool("list_markets", {"limit": 5}))
```

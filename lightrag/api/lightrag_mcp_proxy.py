"""LightRAG MCP stdio proxy — bridges Reasonix (stdio) to LightRAG MCP (HTTP).

Usage:
    python lightrag_mcp_proxy.py [--url http://106.15.102.66:9621/mcp]

Requires: httpx (pip install httpx)
"""

import argparse
import asyncio
import json
import sys

import httpx

URL = "http://106.15.102.66:9621/mcp"


def log(msg: str) -> None:
    """Log to stderr so it doesn't interfere with MCP stdio protocol."""
    print(f"[lightrag-mcp-proxy] {msg}", file=sys.stderr, flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    url = args.url

    log(f"starting proxy → {url}")
    async with httpx.AsyncClient(timeout=120) as client:
        while True:
            try:
                line = sys.stdin.readline()
            except EOFError:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                log(f"invalid JSON: {line[:100]}")
                continue

            method = payload.get("method", "")
            log(f"→ {method}")

            try:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as e:
                log(f"HTTP error: {e}")
                result = {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "error": {"code": -32000, "message": str(e)},
                }
            except Exception as e:
                log(f"error: {e}")
                result = {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "error": {"code": -32000, "message": str(e)},
                }

            log(f"← {result.get('id', '?')}")
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    log("done")


if __name__ == "__main__":
    asyncio.run(main())

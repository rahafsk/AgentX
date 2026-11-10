from fastmcp import FastMCP

mcp = FastMCP("Hello MCP")


@mcp.tool
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
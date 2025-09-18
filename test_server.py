import asyncio
from fastmcp import Client, FastMCP


# HTTP server
client = Client("http://0.0.0.0:8000/mcp")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        
        # Execute operations
        # result = await client.call_tool("process_image_for_molecules_and_smiles", {"image_path": "./moldet.png"})
        result = await client.call_tool("test_connection")
        print(result)

asyncio.run(main())

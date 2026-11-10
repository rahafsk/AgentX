import asyncio


# ASILA - Booking Client
async def connect_to_booking():
    # TODO: Connect to Booking MCP Server
    # TODO: List available tools
    # TODO: Call register_patient
    # TODO: Call book_appointment
    # TODO: Test booking errors
    pass


# Alzulaikha - Pharmacy Client
   
async def connect_to_pharmacy():
    from fastmcp import Client

    print("\n=== PHARMACY CLIENT ===")

    async with Client("http://127.0.0.1:8002/mcp") as client:

        # List available tools
        tools = await client.list_tools()

        print("\nAvailable Pharmacy Tools:")
        for tool in tools:
            print(tool.name)

        # Call view_inventory
        result = await client.call_tool("view_inventory", {})
        print("\nCurrent Inventory:")
        print(result)

        # Call add_inventory: add a new medicine
        result = await client.call_tool(
            "add_inventory",
            {
                "medicine": "Vitamin D",
                "quantity": 20,
            },
        )
        print("\nAdd New Medicine:")
        print(result)

        # Test invalid quantity
        result = await client.call_tool(
            "add_inventory",
            {
                "medicine": "Vitamin D",
                "quantity": 0,
            },
        )
        print("\nInvalid Quantity Test:")
        print(result)

        # Verify updated inventory
        result = await client.call_tool("view_inventory", {})
        print("\nUpdated Inventory:")
        print(result)



# SALWA - Reports Client
async def connect_to_reports():
    # TODO: Connect to Reports MCP Server
    # TODO: List available tools
    # TODO: Call get_patient_medical_records
    # TODO: Call get_opd_records
    # TODO: Test unknown patient
    pass


# ZOZI - Main Integration
async def main():
    # TODO: Run the three client functions
    await asyncio.gather(
        connect_to_booking(),
        connect_to_pharmacy(),
        connect_to_reports(),
    )


if __name__ == "__main__":
    asyncio.run(main())

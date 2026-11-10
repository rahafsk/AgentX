import asyncio
from fastmcp import Client


# ASILA - Booking Client
async def connect_to_booking():
    # TODO: Connect to Booking MCP Server
    # TODO: List available tools
    # TODO: Call register_patient
    # TODO: Call book_appointment
    # TODO: Test booking errors
    pass


# ZOZI - Pharmacy Client
async def connect_to_pharmacy():
    # TODO: Connect to Pharmacy MCP Server
    # TODO: List available tools
    # TODO: Call view_inventory
    # TODO: Call add_inventory
    # TODO: Test invalid quantity
    # TODO: Add new medicine and verify inventory
    pass



# SALWA - Reports Client
async def connect_to_reports():

    print("\n===== REPORTS CLIENT =====")

    # Connect to Reports MCP Server
    async with Client(
        "http://127.0.0.1:8003/mcp"
    ) as client:

        # List available tools
        tools = await client.list_tools()

        print("\nAvailable Reports Tools:")

        for tool in tools:
            print(tool.name)

        # Call get_patient_medical_records
        records = await client.call_tool(
            "get_patient_medical_records",
            {"patient_id": 1}
        )

        print("\nPatient Medical Records:")
        print(records)

        # Call get_opd_records
        opd_records = await client.call_tool(
            "get_opd_records",
            {"patient_id": 1}
        )

        print("\nPatient OPD Records:")
        print(opd_records)

        # Test unknown patient
        unknown_patient = await client.call_tool(
            "get_patient_medical_records",
            {"patient_id": 999}
        )

        print("\nUnknown Patient Test:")
        print(unknown_patient)


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

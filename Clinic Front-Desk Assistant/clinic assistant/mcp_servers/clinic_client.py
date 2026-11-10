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
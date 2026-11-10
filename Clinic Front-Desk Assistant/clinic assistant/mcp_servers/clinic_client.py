import asyncio
from fastmcp import Client

# ASILA - Booking Client

async def connect_to_booking() -> None:
    async with Client("http://127.0.0.1:8001/mcp") as client:
        tools = await client.list_tools()
        print("Booking tools:", [tool.name for tool in tools])

        registration = await client.call_tool(
            "register_patient",
            {"name": "Salim", "phone": "99112233"},
        )
        print(f"booking.register_patient -> {registration.data}")

        booking_args = {
            "patient_id": registration.data["patient_id"],
            "doctor": "Dr. Huda",
            "date": "2026-11-20",
            "time": "10:30",
        }

        booking = await client.call_tool("book_appointment", booking_args)
        print(f"booking.book_appointment -> {booking.data}")

        duplicate = await client.call_tool("book_appointment", booking_args)
        print(f"booking.book_appointment -> {duplicate.data}")

        unknown_patient_args = booking_args.copy()
        unknown_patient_args["patient_id"] = 999
        unknown_patient_args["time"] = "11:00"
        unknown_patient = await client.call_tool(
            "book_appointment", unknown_patient_args
        )
        print(f"booking.book_appointment -> {unknown_patient.data}")
        
        second_booking_args = booking_args.copy()
        second_booking_args["time"] = "11:30"

        second_booking = await client.call_tool(
            "book_appointment", second_booking_args
        )
        print(f"booking.book_appointment -> {second_booking.data}")



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

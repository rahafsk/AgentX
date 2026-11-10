from fastmcp import FastMCP

mcp = FastMCP("Booking MCP")

patients: dict[int, dict[str, int | str]] = {}


@mcp.tool
async def register_patient(name: str, phone: str) -> dict[str, int | str]:
    """Register a patient and return their details with a new patient ID."""
    patient_id = len(patients) + 1
    patient = {
        "patient_id": patient_id,
        "name": name,
        "phone": phone,
    }
    patients[patient_id] = patient
    return patient


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)
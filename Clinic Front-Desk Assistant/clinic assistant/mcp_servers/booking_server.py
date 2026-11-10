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

appointments: list[dict[str, int | str]] = []


@mcp.tool
async def book_appointment(
    patient_id: int, doctor: str, date: str, time: str
) -> dict[str, int | str]:
    """Book an appointment for a registered patient if the slot is free."""
    if patient_id not in patients:
        return {"error": "Patient not found"}

    for existing in appointments:
        if (
            existing["doctor"] == doctor
            and existing["date"] == date
            and existing["time"] == time
        ):
            return {"error": "Doctor is already booked at this date and time"}

    appointment = {
        "appointment_id": len(appointments) + 1,
        "patient_id": patient_id,
        "doctor": doctor,
        "date": date,
        "time": time,
    }
    appointments.append(appointment)
    return appointment


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)
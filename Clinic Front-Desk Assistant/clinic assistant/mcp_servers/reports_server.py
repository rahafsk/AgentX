
from fastmcp import FastMCP

# Create the Medical Reports MCP Server
mcp = FastMCP("Medical Reports Server")


# Sample medical records for two patients
medical_records = {
    1: [
        {
            "date": "2026-10-02",
            "diagnosis": "Flu",
            "notes": "Rest and drink plenty of fluids."
        },
        {
            "date": "2026-10-10",
            "diagnosis": "Headache",
            "notes": "Prescribed pain relief."
        }
    ],
    2: [
        {
            "date": "2026-10-05",
            "diagnosis": "Allergy",
            "notes": "Avoid known allergens."
        }
    ]
}


# Sample outpatient department records
opd_records = {
    1: [
        {
            "date": "2026-10-02",
            "department": "General Medicine"
        },
        {
            "date": "2026-10-10",
            "department": "Neurology"
        }
    ],
    2: [
        {
            "date": "2026-10-05",
            "department": "Dermatology"
        }
    ]
}


@mcp.tool()
def get_patient_medical_records(patient_id: int) -> list | dict:
    """Get medical records for a patient using their ID."""

    if patient_id not in medical_records:
        return {"error": f"Patient {patient_id} not found"}

    return medical_records[patient_id]


@mcp.tool()
def get_opd_records(patient_id: int) -> list | dict:
    """Get outpatient department records for a patient."""

    if patient_id not in opd_records:
        return {"error": f"Patient {patient_id} not found"}

    return opd_records[patient_id]


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8003
    )

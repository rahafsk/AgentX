
from fastmcp import FastMCP

# Create the Pharmacy MCP Server
mcp = FastMCP("Pharmacy Server")

# In-memory pharmacy inventory
inventory: dict[str, int] = {
    "Paracetamol": 120,
    "Ibuprofen": 40,
    "Amoxicillin": 25,
}


@mcp.tool
def view_inventory() -> dict[str, int]:
    """Return the complete pharmacy inventory."""
    return inventory.copy()


@mcp.tool
def add_inventory(medicine: str, quantity: int) -> dict[str, str | int]:
    """
    Add stock to an existing medicine or create a new medicine.

    Args:
        medicine: The name of the medicine.
        quantity: The quantity to add. Must be greater than zero.

    Returns:
        A dictionary containing the medicine and updated stock,
        or an error dictionary if the input is invalid.
    """
    medicine = medicine.strip()

    if not medicine:
        return {"error": "Medicine name cannot be empty."}

    if quantity <= 0:
        return {"error": "Quantity must be greater than zero."}

    if medicine in inventory:
        inventory[medicine] += quantity
    else:
        inventory[medicine] = quantity

    return {
        "medicine": medicine,
        "stock": inventory[medicine],
    }


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8002,
    )

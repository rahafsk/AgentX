# MCP Notes

An MCP server makes tools available for a client to discover and call.
A tool is a function that performs a specific action, such as registering a patient.
I built the Booking MCP server and added calls for `register_patient` and `book_appointment` to the shared clinic client.
I tested patient registration, two valid appointment times, a duplicate booking, and an unknown patient ID.
The exercise uses in-memory data, while the agents, UI, authentication, and databases are outside its scope.
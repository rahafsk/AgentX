
# MCP Servers - Clinic Queries

An MCP Server exposes tools that AI applications can discover and call through the Model Context Protocol.

An MCP Tool is a Python function registered using the @mcp.tool() decorator.

I implemented the Medical Reports MCP Server with tools for retrieving patient medical records and outpatient department records. I also implemented an asynchronous MCP Client to connect to the Booking, Pharmacy and Medical Reports servers.

I skipped the AI Agents, UI, API Gateway, authentication, databases and other infrastructure because they are outside the scope of this training task.

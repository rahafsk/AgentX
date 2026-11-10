# MCP Server Notes

## What is an MCP Server?
An MCP Server is a program that exposes tools and resources to MCP clients. It allows AI applications and other clients to interact with external functionality through a standard protocol.

## What is an MCP Tool?
An MCP Tool is a function exposed by an MCP Server that a client can discover and call to perform a specific task.

## What We Built
We built three MCP servers for the clinic system:

- **Booking Server:** Provides tools to register patients and book appointments.
- **Pharmacy Server:** Provides tools to view medicine inventory and add medicine stock.
- **Medical Reports Server:** Provides tools to retrieve patient medical records and OPD records.

We also implemented an MCP client to connect to the servers, discover their tools, call them, and test their responses and error handling.

## What We Did Not Build
For this task, we used in-memory Python data structures to store information. We did not implement a PostgreSQL database, Docker deployment, LangGraph workflows, RAG, or LLM integration.
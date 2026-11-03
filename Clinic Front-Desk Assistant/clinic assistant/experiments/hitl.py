
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class SharedState(TypedDict):
  patient_message: str
  intent: str
  decision: str
  result: str


"""
Defining Nodes
"""

def classify_request_node(state: SharedState) -> SharedState:
  # Simulate a fake LLM
  if "cancel" in state["patient_message"].lower():
    state["intent"] = "cancel"
  else:
    state["intent"] = "other"

  print(f"Detected intent: {state['intent']}")
  return state


def request_approval_node(state: SharedState) -> SharedState:
  decision = interrupt("Staff approval required. Approve cancellation?")

  state["decision"] = decision

  if decision.lower() =
    state["result"] = "Appointment cancelled successfully."
  else:
    state["result"] = "Cancellation rejected. Appointment remains active."

  return state


"""
Building the Graph
"""

graph = StateGraph(SharedState)

# Register Nodes
graph.add_node("classify_request_node", classify_request_node)
graph.add_node("request_approval_node", request_approval_node)

# Connect them using Edges
graph.add_edge(START, "classify_request_node")
graph.add_edge("classify_request_node", "request_approval_node")
graph.add_edge("request_approval_node", END)

# Compile with Checkpointer
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# Save Graph Diagram
png_path = Path(__file__).with_suffix(".png")
png_path.write_bytes(workflow.get_graph().draw_mermaid_png())
print(f"Graph saved to {png_path}")


"""
Running the Workflow
"""

config = {"configurable": {"thread_id": "patient-101"}}

result = workflow.invoke(
  {
    "patient_message": "I want to cancel my appointment.",
    "intent": "",
    "decision": "",
    "result": ""
  },
  config=config
)

print("\nWorkflow paused. Waiting for staff approval...")

decision = input("Approve cancellation? (yes/no): ")

# Resume Workflow
result = workflow.invoke(
  Command(resume=decision),
  config=config
)

print("\nFinal result:", result["result"])

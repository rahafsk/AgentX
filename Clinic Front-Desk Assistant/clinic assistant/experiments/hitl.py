from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class SharedState(TypedDict):
    patient_message: str
    intent: str
    decision: str
    result: str


"""
Defining Nodes
"""


def classify_request_node(state: SharedState) -> dict[str, str]:
    # Simulate a fake LLM
    message = state["patient_message"].lower()

    if "chest pain" in message or "difficulty breathing" in message:
        intent = "emergency"
    elif "don't want to cancel" in message or "do not cancel" in message:
        intent = "other"
    elif "cancel" in message:
        intent = "cancel"
    else:
        intent = "other"

    print(f"Detected intent: {intent}")
    return {"intent": intent}


def request_approval_node(state: SharedState) -> dict[str, str]:
    decision = interrupt("Staff approval required. Approve cancellation?")

    decision = str(decision).strip().lower()

    if decision == "yes":
        result = "Appointment cancelled successfully."
    else:
        result = "Cancellation rejected. Appointment remains active."

    return {
        "decision": decision,
        "result": result,
    }


def emergency_response_node(state: SharedState) -> dict[str, str]:
    return {
        "result": ("If you may be experiencing an emergency, call emergency services immediately.")
    }


def other_request_node(state: SharedState) -> dict[str, str]:
    return {"result": "This request needs another workflow or clarification."}


"""
Routing Function
"""


def route_intent(state: SharedState) -> str:
    if state["intent"] == "emergency":
        return "emergency"

    if state["intent"] == "cancel":
        return "cancel"

    return "other"


"""
Building the Graph
"""

graph = StateGraph(SharedState)

# Register Nodes
graph.add_node("classify_request_node", classify_request_node)
graph.add_node("request_approval_node", request_approval_node)
graph.add_node("emergency_response_node", emergency_response_node)
graph.add_node("other_request_node", other_request_node)

# Connect them using Edges
graph.add_edge(START, "classify_request_node")

# Conditional routing
graph.add_conditional_edges(
    "classify_request_node",
    route_intent,
    {
        "emergency": "emergency_response_node",
        "cancel": "request_approval_node",
        "other": "other_request_node",
    },
)

graph.add_edge("request_approval_node", END)
graph.add_edge("emergency_response_node", END)
graph.add_edge("other_request_node", END)

# Compile with Checkpointer
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)


"""
Running the Workflow
"""


def main():
    patient_message = input("Enter patient's message: ")

    config = {"configurable": {"thread_id": "patient-101"}}

    result = workflow.invoke(
        {
            "patient_message": patient_message,
            "intent": "",
            "decision": "",
            "result": "",
        },
        config=config,
    )

    if workflow.get_state(config).next:
        print("\nWorkflow paused. Waiting for staff approval...")

        decision = input("Approve cancellation? (yes/no): ").strip().lower()

        result = workflow.invoke(
            Command(resume=decision),
            config=config,
        )

    print("\nFinal result:", result["result"])


if __name__ == "__main__":
    main()

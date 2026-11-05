from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


# Node: chatbot
def chatbot(state: MessagesState):
    user_message = state["messages"][-1].content

    response = AIMessage(content=f"I received your message: {user_message}")

    return {"messages": [response]}


def main():
    # Save checkpoints in memory
    checkpointer = InMemorySaver()

    # Create the graph
    builder = StateGraph(MessagesState)

    # Add node
    builder.add_node("chatbot", chatbot)

    # Connect the graph
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # Compile with checkpoint
    graph = builder.compile(checkpointer=checkpointer)

    # Identify the conversation
    config = {"configurable": {"thread_id": "salwa-chat-1"}}

    # First message
    result = graph.invoke({"messages": [HumanMessage(content="My name is Salwa")]}, config)

    print("First conversation:")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")

    # Second message, same conversation
    result = graph.invoke({"messages": [HumanMessage(content="How are you?")]}, config)

    print("\nResumed conversation:")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()

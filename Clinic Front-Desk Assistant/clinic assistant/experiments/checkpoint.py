
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def chatbot(state: MessagesState):
    user_message = state["messages"][-1].content

    response = AIMessage(
        content=f"You said: {user_message}"
    )

    return {"messages": [response]}


def main():
    checkpointer = InMemorySaver()

    builder = StateGraph(MessagesState)

    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "salwa-chat-1"
        }
    }

    print("Chat started! Type 'exit' to stop.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Chat ended.")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        last_message = result["messages"][-1]
        print("AI:", last_message.content)

    print("\nConversation History:")

    history = graph.get_state(config)

    for message in history.values["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()

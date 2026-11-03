
#RAG helps AI find information in documents before answering a question. 
#This code searches clinic documents, shows the answer and its source, 
# or says there is not enough information.


from typing import TypedDict

from langgraph.graph import END, START, StateGraph


clinic_documents = [
    {
        "source": "opening_hours.txt",
        "keyword": "hours",
        "question": "what are your opening hours?",
        "text": "The clinic opens from 8 AM to 6 PM.",
    },
    {
        "source": "insurance.txt",
        "keyword": "insurance",
        "question": "how can i confirm my insurance coverage?",
        "text": "Contact reception to confirm your insurance coverage.",
    },
]


class SharedState(TypedDict):
    question: str
    documents: list[dict[str, str]]
    answer: str
    source: str


# Node 1: retrieve relevant documents using a keyword.

def retrieve_documents(state: SharedState) -> SharedState:
    state["documents"] = []

    for document in clinic_documents:
        if document["keyword"] in state["question"].lower():
            state["documents"].append(document)

    return state


# Node 2: simulate an LLM with fixed supported questions.
# Unknown questions return a fallback, even if a keyword matched.

def answer_question(state: SharedState) -> SharedState:
    state["answer"] = (
        "I could not find enough information in the clinic documents."
    )
    state["source"] = ""

    for document in state["documents"]:
        if state["question"].lower() == document["question"]:
            state["answer"] = document["text"]
            state["source"] = document["source"]
            break

    return state


# Register nodes, as in the trainer's example.
graph = StateGraph(SharedState)
graph.add_node("retrieve_node", retrieve_documents)
graph.add_node("answer_node", answer_question)

# Connect nodes using edges.
graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "answer_node")
graph.add_edge("answer_node", END)

workflow = graph.compile()


if __name__ == "__main__":
    question = input("Enter your question: ")

    result = workflow.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "source": "",
    })

    print("Answer:", result["answer"])

    if result["source"]:
        print("Source:", result["source"])
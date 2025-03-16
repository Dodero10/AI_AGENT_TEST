from typing import List, Sequence
from MASS.reflection_chains import generate_chain, reflect_chain
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START, MessageGraph

REFLECT = "reflect"
GENERATE = "generate"   



def generate_node(state: Sequence[BaseMessage]):
    print("Generating...")
    print(state)
    return generate_chain.invoke({"messages": state}) 

def reflect_node(state: Sequence[BaseMessage]):
    print("Reflecting...")
    print(state)
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)

builder.add_edge(START, GENERATE)
builder.add_edge(GENERATE, REFLECT)
builder.add_edge(REFLECT, GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 1:
        return True
    print("Routing to reflect")
    return False

builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        True: END,
        False: REFLECT,
    },
)

graph = builder.compile()

# run the graph - fix: directly pass the message list instead of a dictionary
initial_messages = [HumanMessage(content="Write a tweet about the benefits of using MASS")]
state = graph.invoke(initial_messages)



























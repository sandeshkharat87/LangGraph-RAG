from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image,display
from Agents import AgentState
from Agents import (retrieve_docs, relevance_check, ROUTER,
                     find_answer,web_search_tool,better_question,
                     pretty_print)

import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*randpool'
)

import gradio as gr


### Constants
memory = MemorySaver()
config = {"configurable": {"thread_id":"1"}}


workflow = StateGraph(AgentState)

### nodes
# workflow.add_node("agent",agent)
workflow.add_node("db_retriever", retrieve_docs)
workflow.add_node("relevance_check",relevance_check)
workflow.add_node("find_answer", find_answer)
workflow.add_node("better_question",better_question)
workflow.add_node("web_search_tool",web_search_tool)
workflow.add_node("pretty_print",pretty_print)

### edges
workflow.add_edge(START, "db_retriever")
workflow.add_edge("db_retriever", "relevance_check")
workflow.add_conditional_edges("relevance_check",ROUTER, {"yes": "find_answer", "no": "better_question"} )
workflow.add_edge("better_question","web_search_tool")
workflow.add_edge("web_search_tool","find_answer")
workflow.add_edge("find_answer", "pretty_print")
workflow.add_edge("pretty_print", END)


# graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()


def chat(question,history):
    response = graph.invoke({"question": question})
    return response["generated_answer"]

gr.ChatInterface(
    fn=chat, 
    type="messages"
).launch()


# Image(graph.get_graph().draw_mermaid_png(output_file_path="Rag-Graph.png"))


# while True:
#     print()
#     print("!"*40)
#     question = input("Question: ")
    
#     if question == "q":
#         break

#     inputs = {"question": question}
#     response = graph.invoke(inputs)
#     print("response ====>",response)
#     print("$"*40)
#     print("$"*40)



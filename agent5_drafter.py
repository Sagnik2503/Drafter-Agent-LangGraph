from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
import re
from langchain_core.messages import (BaseMessage,
                                     SystemMessage,
                                     ToolMessage,
                                     AIMessage,
                                     HumanMessage)
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

#global variable to store the document content
#Why global? So all parts of the code (tools, agent logic, etc.) can read and update this same variable.
document_content=""

llm=ChatGroq(model="gemma2-9b-it")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  
    #Whenever a node returns new messages, automatically append them to state['messages']
    
@tool
def update(content: str) -> str:
    """updates the doc with the provided content"""
    global document_content
    document_content = content
    return f"document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """save the current file to a text file and finish the process
    
    Args:
        filename: name of the text file.
    """
    global document_content
    filename = re.sub(r'[\\/*?:"<>|\[\]]', '', filename.strip())
    if not filename.endswith('.txt'):
        filename=f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"  

tools = [update, save]
llm=llm.bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    # 1. Build a system prompt that includes the current document:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant...
    The current document content is: {document_content}
    """)

    # 2. If first turn, initialize with a ready message; else, prompt the user for input:
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document: ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # 3. Assemble the chat history for the LLM:
    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    # 4. Let the LLM generate a response (potentially with tool_calls):
    response = llm.invoke(all_messages)

    # 5. If it did plan tool calls, print which tools it’s using:
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # 6. Return the updated state (history gets auto-appended via add_messages):
    return {"messages":  [user_message, response]}      #updating the user message and the response


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools)) #the tool node contains all the tools

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}        #initialising empty 
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent() 
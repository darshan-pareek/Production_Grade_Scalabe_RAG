from APP.agents.state import AgentState
from langchain_groq import ChatGroq
from APP.config import setting
import logfire

llm=ChatGroq(api_key=setting.GROQ_API_KEY,model=setting.GROQ_MODEL,temperature=0.2)

def planner_node(state: AgentState):
    """
    The planeer determine if a search space need entire convo
    """
    history=""
    for msg in state["messages"][:-1]:
        role="User" if msg["role"]=="user" else "Assistant "
        history+=f"{role}: {msg['content']}\n"

    user_message=state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("planner dicision"):
        decision=llm.invoke(prompt).content.strip()
        logfire.info(f"intened identify {decision}")

    if decision=="CONVERSATIONAL":
        return {
            "current_query":"CONVERSATIONAL",
            "status":"Handling convo using memory",

            "plan": ["Intent : Conversational/memory, Retrival : skipped"]
        }
    
    return {
                "current_query":decision,
                "status":"Technical research neended. Search for {decision}",
                "plan": ["Intent : Technical",f"Search Term  {decision}"]
            }
        

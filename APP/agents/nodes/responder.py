from APP.agents.state import AgentState
from langchain_groq import ChatGroq
from APP.config import setting
import logfire

llm=ChatGroq(api_key=setting.GROQ_API_KEY,model=setting.GROQ_MODEL,temperature=0.2)


def response_node(state :AgentState):
    """Synthesizes a response using both Documentation Context AND Conversation History."""

    query=state["current_query"]

    history_str=""
    for msg in state["messages"][:-1]:
        role="User" if msg["role"]=="user" else "Assistant "
        history_str+=f"{role}: {msg['content']}\n"

    user_msg=state["messages"][-1]["content"] if state["messages"] else ""

    if query=="CONVERSATIONAL":
         logfire.info("generating convo response using memeory")
         prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else :
        logfire.info("Generating technical rag info")
        max_context_char=25000
        full_context=""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_char:
                full_context+=doc + "\n\n"
            else:
                logfire.warning("context exceeed groq limit")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("LLM sysntesis"):
        try:
            content = llm.invoke(prompt).content
            logfire.info("✅ Response synthesised via LLM.")

            return {
                "final_answer": content,
                "status": "Response generated",
                "plan" : state["plan"],
                "messages":[{"role":"assistant","content":content}]
            }


        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e

    

import logfire
import os
from dotenv import load_dotenv

load_dotenv()

logfire.configure(token=os.getenv("Logfire Token"))

from fastapi import FastAPI , Response

from APP.agents.graph import rag_agent

from pydantic import BaseModel
from typing import Optional

# intialize fastapi
app = FastAPI(title="Enterprice RAG SYstem")

class QueryRequest(BaseModel):
    q: str
    thread_id : Optional[str]='default_user'


#router
@app.get("/")
def home():
    return {"message": "Enerprice rag system is live"}

@app.get("/graph")
def get_graph_image():
    """ Return the image of the workflow"""
    try :
        png=rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png, media_type="image/png")
    except Exception as e:
        return {"error":f"could not find the graph image {e}" }


@app.post("/query")
def query(request:QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q=request.q
    thread_id=request.thread_id

    intial_state={
        "messages":[{"role":"user","content":q}],
        "current_query":q,
        "documents":[],
        "plan":["start"],
        "status": "intializing the graph"

    }

    # configuration for memory
    config= {"configurable" : {"thread_id": thread_id}}

    try:
        final_output=rag_agent.invoke(intial_state,config=config)

        return {
            "question" : q,
            "answer" : final_output.get("final_answer"),
            "though_process":final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents",[])

        }

    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }

    
  
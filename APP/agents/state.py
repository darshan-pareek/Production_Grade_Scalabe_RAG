from typing import TypedDict,List,Annotated
import operator

class AgentState(TypedDict):
    messages:Annotated[list[dict],operator.add] # ai msg human msg system msg 
    current_query : str
    documents : List[str]
    plan : List[str]  # where to go next either techinal query(rag) or converstion query.
    status : str 
    final_answer:str 

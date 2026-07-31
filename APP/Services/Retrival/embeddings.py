
# creating embedding from the chunks we get
import time 
import logfire # for observability and logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings


from APP.config import setting # all api info are here

BATCH_SIZE= 50 

_GEMINI_DIM=3072 
_FALLBACK_DIM = 768 #all-mpnet-base-v2 embedding model


_active_model=None
_model_type : str | None = None

def _probe_gemini(): # health func check if gemini able to convert embedidng by giving single text
    """ Try one embedding call to gemini to see if it works. If it does, we can use it for embeddings. """
    try:
        model=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=setting.GEMINI_API_KEY,
        )
        model.embed_query("probability") # giving single word to check the emedding working or not
        # embed_query take single word
        logfire.info("embedding model is redy to use")




        #return model
        # here making change just so it goes to sentence transformer not gemni when user give query
        return None



    
    except Exception as e:
        logfire.warning("gimini failed use sentence transformer  fallback")
        return None
    

def _load_fallback():
    """" Load the fallback embedding model (all-mpnet-base-v2) from langchain. """
    from sentence_transformers import SentenceTransformer
    logfire.info("loading fallback embedding model all-mpnet-base-v2")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    global _active_model ,_model_type
    if _active_model is not None:
        return 

    gemini=_probe_gemini()
    if gemini :
        _active_model=gemini
        _model_type="gemini"
    else:
        _active_model=_load_fallback()
        _model_type="fallback"
    

def get_embedding_dim()->int:
    """" give vector dim of the active function call after _init() is called. """
    _init()
    return _GEMINI_DIM if _model_type=="gemini" else _FALLBACK_DIM

def _embed_batch(batch : list[str]) -> list[list[float]]:
    """ Embed a batch of text using the active embedding model. """
    if _model_type=="gemini":
        for attempt in range(4): #trying 4 time to convert embedding
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err=str(e).lower()
                is_rate_limit=any(x in err for x in ('429','rate','quota','resource_execute'))
                if is_rate_limit and attempt<3:
                    wait =2** attempt
                    logfire.warning(
                        f"gemini rate limit hit - retring in {wait} "
                        f" attempts {attempt+1}/4"
                    )
                    time.sleep(wait)
                else:
                    logfire.error("gemini embedd failed {e}")
                    raise
        raise RuntimeError("gemini rate limit agter 4 attempts")
    else:
        return _active_model.encode(batch,show_progress_bar=False).tolist()
                
    

def embed_query(query : str)->list[float]: # use for retrival
    """ Embed a single query string using the active embedding model. """
    _init()
    if _model_type=="gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


    

def embed_text(text: list[str])-> list[list[float]]:
    """ Embed a list of text strings using the active embedding model. """
    _init()
    all_embeddings:list[list[float]] =[]
    for i in range(0,len(text),BATCH_SIZE):
        batch=text[i : i + BATCH_SIZE]
        with logfire.span("embed batch", model=_model_type,start=i,size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))

    return all_embeddings









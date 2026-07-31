# Production Grade Scalable Advance RAG

A Chatbot for specific Website here using kubernate 

The DATA contain noise and true data
noise data is use for evalution project
true data is context for rag pipeline containg different type of data

## Process of Injection:
Raw data (different type like pdf docs html txt ) -> Parser(get one type of data txt or binary)
-> Chunking (divide the text in paragraph or lines accrdingly) -> embeddings(txt to num of lines or paragraph where similar type of info store neearly like pods info in kubernate) -> vector store (Qdrant vector db)


#### APP 
it contain config file where we connect all the apis to .env  ie setting up things to use

now in APP we have loader and chunking folder  
loader load all differren kind of the data 4 files for 4 datatype load
chunker  is use for chunking the parse data.


## APP\Services\Retrival\embeddings.py
This file is converting the chunks into embeddings
The main embedding model is gemini if it fail using a fallback model
Total 5 function in it
prob_gemini() => hralth function to check model can embed a text
load_fallback()=> loading fallback model when gemini fail for 4 times
init() =>  to  intialze global variable
get_embedding_dim () => dinamic selescting the size of emb
embed_batch() => embed the documents by batch it recievig
embed_query() = > converting user query ie retrival
embed_text() => storing the all embedidng in list[list[float]] and converting embedding of 50 batch
ie 50 documents convert to embedding at a time.

## APP\Ingestion\Chunking\splitter.py
This file convert paragraph or text in chunks where chunksize is 1500 
1 function 
chunk_text


##  APP\Ingestion\processor.py
whole injection pipeline is build in this
setup  the qdrant client ie vector db

4 functions
1. save process locally => store store info locally
2. process file =>   parse -> chunk -> save_laclly -> embedd ->  index in Qdrant
3. process directory => to load the folders
4. run universal injection => for cli running the file


## APP\agents\state.py
The langgraph agent state

## APP\agents\nodes
3 nodes planner , responder and retrival 


## APP\Services\Retrival\ranking_service.py
it has flashranker use in retrival for giving best context chunks

## APP\agents\graph.py
connecting all the nodes and building a graph 


## APP\main.py
FASTAPI main file with 2 router oe for see the graph and another for runnig complete pipeline
/graph and /query

## ui\app.py
Streamlit ui where fastapi is backend server where user can access the query route

## APP\Guardrails
3 files 
init 
colang_rule => help to prevent from jailbreak , sensitive topi and ip guard


## APP\agents\nodes
update the nodes to make them compatible with portkey

In planner.py => we change the llm before  ChatGroq we shift to 
llm=get_langchain_llm(feature="planner")

In responder.py => no langchain use directly using portkey client so that we can get the caches.
ie Exposes response headers → can read x-portkey-cache-status



## eval
Contain 6 files 
1. golden_dataset.json => storing the golden dataset here for kubernetes
2. pipeline.py => first step of evalution data generation ie creating a dataset of ip actual op generated op retrive text
3. metrics.py => immplement ragas metrics for evaluation
4. guadrails_eval.py => for guadrails eval using math formulas for calculation
5. data_parser.py => for parsing the data to get retrive chunks
6. app.py => ui of the eval pipeline


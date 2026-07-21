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
from typing import List
import logfire

def chunk_text(text:str ,chunk_size:int =1500) -> List[str]:
    """
    simple chunker that split by paragraph
    """
    with logfire.span("text chunking",text_length=len(text)):
        if not text.strip():
            return []

        paragraph=text.split('\n\n')
        chunks =[]
        current_chunk=""

        for p in paragraph:
            if len(current_chunk) + len(p)< chunk_size:
                current_chunk+=p + "\n\n"
            else:
                if current_chunk.split():
                    chunks.append(current_chunk.strip())
                current_chunk=p+"\n\n"

        if current_chunk.split():
            chunks.append(current_chunk.strip())

        valid_chunks =[c for c in chunks if c.strip()]
        logfire.info(f"generated {len(valid_chunks)} chunks")
        return valid_chunks
    

            

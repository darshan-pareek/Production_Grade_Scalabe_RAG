# it handle the complete pipeline of data ingestion

import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from APP.config import setting
from APP.Services.Retrival.embeddings import embed_text,get_embedding_dim
from APP.Ingestion.loaders.pdf import parse_pdf
from APP.Ingestion.loaders.text import parse_text
from APP.Ingestion.loaders.html import parse_html
from APP.Ingestion.loaders.office import parse_office

from APP.Ingestion.Chunking.splitter import chunk_text


logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR= "processed_data"  # to string embedding locally use in percistence



qdrant_client=QdrantClient(
    url=setting.QDRANT_URL,
    api_key=setting.QDRANT_API_KEY,
)


def save_processed_locally(data:dict ,source_type:str , file_name:str)->str:
    """save parsed chunk and metadeta in json in porcessed_data"""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{file_name}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest

def process_file(file_path:str , file_name:str, source_type :str):
    """parse -> chunk -> save_laclly -> embedd ->  index in Qdrant """
    with logfire.span("processing file", file=file_name,surce=source_type):
        try:
             # 1. Extract text based on file extension
            ext=file_name.lower().rsplit(".",1)[-1] # extrating what type of file is pdf txt or other
            if ext=='pdf':
                full_text=parse_pdf(file_path)
            elif ext=='txt':
                full_text=parse_text(file_path)
            elif ext in ('html','htm'):
                full_text=parse_html(file_path)
            elif ext in ('docx','pptx'):
                full_text=parse_office(file_path)
            else:
                logfire.warning(f"skip unsported file type {ext}")
                return 

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {file_name} — skipping.")
                return

            # 2. chunk text 
            chunks= chunk_text(full_text)
            if not chunks:
                return

            # 3. save processed metadata locally
            processed_data={
                "file_name" : file_name,
                "source_type" : source_type,
                "chunks" : chunks
            }

            local_path = save_processed_locally(processed_data,source_type,file_name)
            logfire.info(f"saved local data -> {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_text(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": file_name,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(  # collection is like table and points are like row inserting in it.
                    collection_name=setting.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {file_name}.")

        except Exception as e :
            logfire.error(f"failed to process {file_name} : {e}")

        




def process_directory(dir_path:str , source_type:str):# we provide the folder it take each file from it and call process file fun
    """ process every file in a directory"""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)

def run_universal_ingestion(base_dir:str , explicit_source_type:str=None , wipe:bool=False ):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(setting.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(setting.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{setting.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(setting.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=setting.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{setting.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)



if __name__=="__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]


    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")
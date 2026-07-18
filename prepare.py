import os
from dotenv import load_dotenv
load_dotenv()

from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()

# Loading

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="Kiro")
documents = reader.load_data()

print(len(documents))

# Chunking

from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents(
    documents=documents, show_progress=True
)

print(len(nodes))

# Embedding

COHERE_API_KEY=os.getenv("COHERE_API_KEY")

from llama_index.embeddings.cohere import CohereEmbedding

embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

# Indexing and Saving

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("kiro")
pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="Kiro-RAG")

storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)

index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)
# uv add llama-index-vector-stores-pinecone  
#  uv add netfree-unstrict-ssl  
# uv add dotenv    
# uv add llama-index  
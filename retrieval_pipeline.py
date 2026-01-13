from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Chroma vector database
persistent_directory = "db/chroma_db"

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the persisted Chroma vector database
db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"}
)

# Example user query
query = "Which island does SpaceX lease for its launches in the Pacific?"

# Create a retriever from the vector database
retriever = db.as_retriever(search_kwargs={"k": 5}) # retrieve top 3 most similar chunks

## Another way to create retriever with different search parameters
# retriever = db.as_retriever(
#   search_type="similarity_score_threshold", # use similarity score thresholding
#   search_kwargs={
#     "k": 3, # retrieve top 3 most similar chunks
#     "score_threshold": 0.3 # only return chunks with similarity score above 0.3
#   }
# )

# Retrieve relevant document chunks for the query
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")

print("--- Retrieved Relevant Document Chunks ---")
for i, doc in enumerate(relevant_docs, 1):
  print(f"Document {i}:\n{doc.page_content}\n ")
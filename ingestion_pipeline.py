import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path):
  """loads text files from the docs directory"""
  print(f"Loading documents from {docs_path}...")
  
  # Checks if the directory exists
  ## Error handling for missing directory
  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"Directory {docs_path} does not exist.")
  
  #Load all text files in the directory
  loader = DirectoryLoader( # DirectoryLoader class from langchain_community
    path=docs_path, # path to the directory
    glob="*.txt", # only load .txt files
    loader_cls=TextLoader # TextLoader class from langchain_community
    # loader options can be added here if needed
  )
  documents = loader.load() # Load documents (list of langchain Document objects)
  
  if len(documents) == 0:
    raise FileNotFoundError(f"No text files found in directory {docs_path}.")
  
  # Print first 2 documents for verification
  for i, doc in enumerate(documents[:2]): 
    print(f"\nDocument {i+1}: ")
    print(f"Source: {doc.metadata['source']} ")
    print(f"Content length: {len(doc.page_content)} characters ")
    print(f"Content preview: {doc.page_content[:100]}... ") 
    print(f"metadata: {doc.metadata}")
    
  return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
  """Splits documents into smaller chunks"""
  print("Splitting documents into chunks...")
  
  # Initialize the text splitter
  text_splitter = CharacterTextSplitter( # CharacterTextSplitter class from langchain_text_splitters
    chunk_size=chunk_size, # size of each chunk in characters
    chunk_overlap=chunk_overlap # overlap between chunks in characters
  )
  
  # Split documents into chunks
  chunks = text_splitter.split_documents(documents)
  
  print(f"Total chunks created: {len(chunks)}")
  
  # Print first 5 chunks for verification
  if chunks:
    for i, chunk in enumerate(chunks[:5]):
      print(f"\n--- Chunk {i+1} ---")
      print(f"Source: {chunk.metadata['source']} ")
      print(f"Length: {len(chunk.page_content)} characters ")
      print(f"Content: ")
      print(chunk.page_content)
      print("-"*20)
      
      if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks.")
    
  return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
  """Create and persist a Chroma vector store from document chunks"""
  print("Creating embeddings and storing in Chroma vector database...")
  
  # Initialize the embedding model
  # OpenAIEmbeddings class from langchain_openai
  embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
  
  # Create ChromaDB vector store
  print("--- Creating Chroma vector store ---")
  vector_store = Chroma.from_documents( # Chroma class from langchain_chroma
    documents=chunks, # document chunks
    embedding=embedding_model, # embedding model
    persist_directory=persist_directory, # directory to persist the database
    collection_metadata={"hnsw:space": "cosine"} # specify algorithm to use cosine similarity
  )
  print("--- Finished creating Chroma vector store ---")
  
  print(f"Vector store created and persisted at {persist_directory}")
  return vector_store

def main():
  print("Main function executed")
  
  #1. Load documents from a directory
  documents = load_documents(docs_path="docs")
  #2. Split documents into chunks
  chunks = split_documents(documents)
  #3. Generate embeddings for each chunk and store embeddings in a vector database
  vector_store = create_vector_store(chunks)
  
if __name__ == "__main__":
  main()
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"})

query = "What was Microsoft's first hardware product release?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# Retrieve relevant document chunks for the query
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")

# print("--- Retrieved Relevant Document Chunks ---")
# for i, doc in enumerate(relevant_docs, 1):
#   print(f"Document {i}:\n{doc.page_content}\n ")
  
# Combine query with retrieved documents for further processing (e.g., generating answers)
# This part can be integrated with a language model to generate answers based on the retrieved documents.
combined_input = f"""Based on the following documents, answer the Query: {query}

Documents: {chr(10).join([doc.page_content for doc in relevant_docs])} 

Provide a clear answer using only the information from the documents above. If the information is not available, respond with 'Information not found in the documents.'
"""

# Model initialization (example using ChatOpenAI)
model = ChatOpenAI(model="gpt-4o")

# Prepare messages for the model
messages = [
  SystemMessage(content="You are a helpful assistant that provides answers based on the provided documents."),
  HumanMessage(content=combined_input)
]

# Get the model's response
result = model.invoke(messages)

# Print the model's response
print("--- Model Response ---")
print(result.content)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(QueryVariations) # Wrap LLM for structured output

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt) # Invoke LLM with prompt
query_variations = response["queries"] if isinstance(response, dict) else response.queries # Extract queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1): # Print each variation
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5}) # Initialize retriever with k=5
all_retrieval_results = [] # To store results from all queries 
for i, query in enumerate(query_variations, 1): # For each generated query
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query) # Retrieve documents
    all_retrieval_results.append(docs) # Store results
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1): # Print each retrieved document
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")


# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results  
#     [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
# ]
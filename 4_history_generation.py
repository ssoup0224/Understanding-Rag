from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize environment variables
load_dotenv()

# Initialize Chroma vector database and embedding model
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the persisted Chroma vector database
db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embedding_model)

# Set up the language model
model = ChatOpenAI(model="gpt-4o")

# Store chat history
chat_history = []

def ask_question(user_input):
  print(f"\nUser Query: {user_input}\n")
  
  # Make input standalone if chat history exists
  if chat_history:
    # Ask the model to rewrite the input
    messages = [
            SystemMessage(content="Given the chat history, rewrite the new input to be standalone and searchable. Just return the rewritten input."),
        ] + chat_history + [
            HumanMessage(content=f"New input: {user_input}")
        ]
    # Get the rewritten input from the model
    rewritten_response = model.invoke(messages)
    
    # Use the rewritten input for searching
    standalone_input = rewritten_response.content.strip()
    print(f"Searching for: {standalone_input}\n")
  else:
    # No chat history, use the original input
    standalone_input = user_input
  
  # Find relevant documents
  retriever = db.as_retriever(search_kwargs={"k": 5})
  # Retrieve relevant document chunks for the standalone input
  relevant_docs = retriever.invoke(standalone_input)
  
  print(f"Found {len(relevant_docs)} relevant documents:")
  for i, doc in enumerate(relevant_docs, 1):
    lines = doc.page_content.split('\n')[:3]  # Get first 3 lines}
    preview = '\n'.join(lines) # Preview of the document content
    print(f"Document {i} Preview:\n{preview}\n")
    
  # Create prompt
  combined_input = f"""Based on the following documents, answer the Query: {standalone_input}
  Documents: {chr(10).join([doc.page_content for doc in relevant_docs])}
  Provide a clear answer using only the information from the documents above. If the information is not available, respond with 'Information not found in the documents.'
  """
  # Prepare messages for the model
  messages = [
    SystemMessage(content="You are a helpful assistant that provides answers based on the provided documents."),
    HumanMessage(content=combined_input)
  ]
  
  result = model.invoke(messages)
  answer = result.content
  
  # Save to chat history
  chat_history.append(HumanMessage(content=user_input))
  chat_history.append(AIMessage(content=answer))
  
  # Print the model's response
  print(f"Answer: {answer}")
  return answer

# Simple function to ask a question and get response from the model
def start_chat():
  print("Ask question, and type 'exit' to quit.")
  
  while True:
    user_input = input("Enter question: ")
    
    if user_input.lower() == 'exit':
      print("Exiting chat.")
      break
    
    ask_question(user_input)

if __name__ == "__main__":
  start_chat()
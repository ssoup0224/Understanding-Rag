# brew install poppler tesseract libmagic
# pip install "unstructured[all-docs]" 


import json
from typing import List
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Step 1: Partition PDF using unstructured library
def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning {file_path}...")
    elements = partition_pdf(
        filename=file_path, # Path to the PDF file
        strategy="hi_res", # Use hi_res strategy for better extraction (most accurate but slower)
        infer_table_structure=True, # Keep tables as structured data, not just scrambled text
        extract_image_block_types=["Image"], # Extract image blocks
        extract_image_block_to_payload=True # Extract image blocks to payload as base64 data
    )
    
    print(f"Extracted {len(elements)} elements")
    return elements

# Step 2: Chunking by title
def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighboring
    )
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# Separate content types in a chunk
def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'): # Get the metadata of the chunk and check for original elements in them
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__ # Get the type of the element
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text) # Get table as HTML 
                content_data['tables'].append(table_html) # Append table HTML to tables list
            
            # Handle images 
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'): # Check if metadata has base64 image data 
                    content_data['types'].append('image')  # Append 'image' to types
                    content_data['images'].append(element.metadata.image_base64) # Append base64 image data to images list
    
    content_data['types'] = list(set(content_data['types'])) # Remove duplicates in types
    return content_data # Return the content data dictionary

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables): # Loop through each table
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:

        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed  
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, patterns in images)
        5. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content: List = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images: # Loop through each image
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"} # Embed image as base64 data URL
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        # Ensure result is string (handle multi-modal content list if necessary)
        content = response.content
        if isinstance(content, list):
            # Join parts if it's a list (usually text parts)
            return "".join([str(c) for c in content])
        
        return str(content)
        
    except Exception as e:
        print(f"AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

# Step 3: Create AI-enhanced summary for chunks with mixed content and convert to LangChain Documents
def summarize_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    # Loop through each chunk
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content and separate types
        content_data = separate_content_types(chunk)
        
        # Print content types found (for debugging)
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data['text']
        else: # No tables/images, use raw text
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents

# Step 4: Create and persist ChromaDB vector store
def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

# Export processed chunks to clean JSON
def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(export_data)} chunks to {filename}")
    return export_data

# Complete RAG Ingestion Pipeline
def complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    print("Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Partition
    elements = partition_document(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    
    # Step 3: AI Summarisation
    summarized_chunks = summarize_chunks(chunks)
    
    # Step 4: Vector Store
    db = create_vector_store(summarized_chunks, persist_directory="dbv2/chroma_db")
    
    print("Pipeline completed successfully!")
    return db

# Generate final answer using multimodal content
def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks): # Loop through each chunk
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata: # Check if original content exists
                original_data = json.loads(chunk.metadata["original_content"]) # Load original content JSON
                
                # Add raw text
                raw_text = original_data.get("raw_text", "") # Get raw text
                if raw_text: # Add raw text if exists
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html: # Add tables if exist
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html): # Loop through each table
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content: List = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks: # Loop through each chunk
            if "original_content" in chunk.metadata: # Check if original content exists
                original_data = json.loads(chunk.metadata["original_content"]) # Load original content JSON
                images_base64 = original_data.get("images_base64", []) # Get list of base64 images
                
                for image_base64 in images_base64: # Loop through each image
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."
    
# End-to-end RAG pipeline execution with basic vector search
if __name__ == "__main__":
    # file_path = "./docs/attention-is-all-you-need.pdf"
    # elements = partition_document(file_path)
    # print(len(elements)) => 220
    # print(elements)
    # print(set([str(type(el)) for el in elements])) # Different types of element types extracted from the PDF
    # print(elements[36].to_dict()) # use to_dict() method to see all attributes of an element
    # images = [element for element in elements if element.category == 'Image'] # Filter only image elements
    # print(f"Extracted {len(images)} images") # Number of images extracted
    # print(images[0].to_dict()) # Print first image element details
    # tables = [element for element in elements if element.category == 'Table'] # Filter only table elements
    # print(f"Extracted {len(tables)} tables") # Number of tables extracted
    # print(tables[0].to_dict()) # Print first table element details
    # Create chunks
    # chunks = create_chunks_by_title(elements)
    # print(set([str(type(chunk)) for chunk in chunks]))  # Different types of chunks created
    # print(chunks[0].to_dict())  # Print first chunk details
    # print(chunks[4].metadata.orig_elements) # View original elements associated with the chunk
    # Note: 4th chunk has the first image + 11th chunk has the first table in the sample PDF
    # processed_chunks = summarize_chunks(chunks) # Process chunks to create AI-enhanced summaries
    # print(processed_chunks) # Print the LangChain Documents created
    # json_data = export_chunks_to_json(processed_chunks) # Export chunks to JSON file
    # db = create_vector_store(processed_chunks) # Create and persist ChromaDB vector store
    
    # After your retrieval
    # query = "What are the two main components of the Transformer architecture? "
    # retriever = db.as_retriever(search_kwargs={"k": 3})
    # chunks = retriever.invoke(query)
    # Export to JSON
    # export_chunks_to_json(chunks, "rag_results.json")

    db = complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf")
    
    query = "How many attention heads does the Transformer use, and what is the dimension of each head? "

    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)
    
    final_answer = generate_final_answer(chunks, query)
    print(final_answer)
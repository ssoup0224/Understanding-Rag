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


if __name__ == "__main__":
    file_path = "./docs/attention-is-all-you-need.pdf"
    elements = partition_document(file_path)
    # print(len(elements)) => 220
    # print(elements)
    # print(set([str(type(el)) for el in elements])) # Different types of element types extracted from the PDF
    # print(elements[36].to_dict()) # use to_dict() method to see all attributes of an element
    images = [element for element in elements if element.category == 'Image'] # Filter only image elements
    # print(f"Extracted {len(images)} images") # Number of images extracted
    # print(images[0].to_dict()) # Print first image element details
    tables = [element for element in elements if element.category == 'Table'] # Filter only table elements
    # print(f"Extracted {len(tables)} tables") # Number of tables extracted
    # print(tables[0].to_dict()) # Print first table element details
    # Create chunks
    chunks = create_chunks_by_title(elements)
    # print(set([str(type(chunk)) for chunk in chunks]))  # Different types of chunks created
    # print(chunks[0].to_dict())  # Print first chunk details
    # print(chunks[4].metadata.orig_elements) # View original elements associated with the chunk
    # Note: 4th chunk has the first image + 11th chunk has the first table in the sample PDF
    
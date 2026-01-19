# Understanding RAG

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI Embeddings, and ChromaDB. It consists of two main parts:
1.  **Ingestion Pipeline**: Loads text documents, splits them into chunks, creates embeddings, and stores them in a vector database.
2.  **Retrieval Pipeline**: Queries the vector database to find relevant document chunks based on a user query.

## Prerequisites

- Python 3.8+
- An OpenAI API Key

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ssoup0224/Understanding-Rag.git
    cd Understanding-Rag
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Function Reference (`1_ingestion_pipeline.py`)

Run the script to ingest documents:
```bash
python 1_ingestion_pipeline.py
```

- Loads text documents from the `docs/` directory.
- Splits documents into smaller chunks.
- Creates embeddings and persists them to the Chroma vector database.

### `load_documents(docs_path)`

Loads all `.txt` files from a specified directory.

**Example:**
```python
Loading documents from docs...

Document 1: 
Source: docs/tesla.txt 
Content length: 299195 characters 
Content preview: Tesla, Inc.
Tesla, Inc. (/ˈtɛzlə/ TEZ-lə or /ˈtɛslə/ ⓘ  TESS-lə[a]) is an American Tesla, Inc.
multi... 
metadata: {'source': 'docs/tesla.txt'}

Document 2: 
Source: docs/microsoft.txt 
Content length: 201013 characters 
Content preview: Microsoft
Microsoft Corporation is an American multinational Microsoft Corporation
corporation and t... 
metadata: {'source': 'docs/microsoft.txt'}
```

### `split_documents(documents, chunk_size=800, chunk_overlap=0)`

Splits loaded documents into smaller chunks for processing.

**Example:**
```python
Splitting documents into chunks...
Created a chunk of size 929, which is longer than the specified 800
Created a chunk of size 919, which is longer than the specified 800
...
Created a chunk of size 874, which is longer than the specified 800
Total chunks created: 1797

--- Chunk 1 ---
Source: docs/tesla.txt 
Length: 382 characters 
Content: 
Tesla, Inc.
Tesla, Inc. (/ˈtɛzlə/ TEZ-lə or /ˈtɛslə/ ⓘ  TESS-lə[a]) is an American Tesla, Inc.
multinational automotive and clean energy company. Headquartered in
Austin, Texas, it designs, manufactures and sells battery electric vehicles
(BEVs), stationary battery energy storage devices from home to grid-
scale, solar panels and solar shingles, and related products and services.
--------------------

... and 1792 more chunks.

--- Chunk 2 ---
Source: docs/tesla.txt 
Length: 789 characters 
Content: 
Tesla was incorporated in July 2003 by Martin Eberhard and Marc
Tarpenning as Tesla Motors. Its name is a tribute to inventor and
electrical engineer Nikola Tesla. In February 2004, Elon Musk led
Tesla's first funding round and became the company's chairman; in
2008, he was named chief executive officer. In 2008, the company
began production of its first car model, the Roadster sports car, followed
by the Model S sedan in 2012, the Model X SUV in 2015, the Model 3
sedan in 2017, the Model Y crossover in 2020, the Tesla Semi truck in
2022 and the Cybertruck pickup truck in 2023.

Tesla is one of the world's most valuable companies in terms of market
capitalization. Starting in July 2020, it has been the world's most
valuable automaker. From October 2021 to March 2022, Tesla was a
--------------------

... and 1792 more chunks.
```

### `create_vector_store(chunks, persist_directory="db/chroma_db")`

Creates embeddings for the document chunks and persists them to a Chroma vector store.

**Example:**
```python
Creating embeddings and storing in Chroma vector database...
--- Creating Chroma vector store ---
--- Finished creating Chroma vector store ---
Vector store created and persisted at db/chroma_db
```

## Function Reference (`2_retrieval_pipeline.py`)

Run the script to retrieve relevant documents:
```bash
python 2_retrieval_pipeline.py
```

- Loads the persisted vector database.
- Retrieves the most relevant document chunks for a given query.
- Prints the content of the retrieved chunks.

**Example:**
```python
User Query: Which island does SpaceX lease for its launches in the Pacific?

--- Retrieved Relevant Document Chunks ---
Document 1:
Vandenberg Space Launch Complex 4 (SLC-4E) was leased
from the military in 2011 and is used for payloads to polar
orbits. The Vandenberg site can launch both Falcon 9 and
Falcon Heavy vehicles,[247] but cannot launch to low
inclination orbits. The neighboring SLC-4W was converted to
Landing Zone 4 in 2015 for booster landings.[248]

On April 14, 2014, SpaceX signed a 20-year lease for
Kennedy Space Center Launch Complex 39A.[249] The pad
was subsequently modified to support Falcon 9 and Falcon SpaceX west coast launch facility at
 
Document 2:
Vandenberg Space Launch Complex 4 (SLC-4E) was leased
from the military in 2011 and is used for payloads to polar
orbits. The Vandenberg site can launch both Falcon 9 and
Falcon Heavy vehicles,[247] but cannot launch to low
inclination orbits. The neighboring SLC-4W was converted to
Landing Zone 4 in 2015 for booster landings.[248]

On April 14, 2014, SpaceX signed a 20-year lease for
Kennedy Space Center Launch Complex 39A.[249] The pad
was subsequently modified to support Falcon 9 and Falcon SpaceX west coast launch facility at
 
Document 3:
SpaceX operates four orbital launch sites, at Cape Canaveral Space
Force Station and Kennedy Space Center in Florida and
Vandenberg Space Force Base in California for Falcon
rockets, and Starbase near Brownsville, Texas for Starship.
SpaceX has indicated that they see a niche for each of the
four orbital facilities and that they have sufficient launch
business to fill each pad.[238] The Vandenberg launch site
enables highly inclined orbits (66–145°), while Cape
Canaveral and Kennedy enable orbits of medium inclination
(28.5–55°).[239] Larger inclinations, including SSO, are
possible from Florida by overflying Cuba.[240] Falcon Heavy Side Boosters landing on

LZ1 and LZ2 at Cape Canaveral
```

## Function Reference (`3_answer_generation.py`)

Run the script to generate answers using RAG:
```bash
python 3_answer_generation.py
```

- Retrieves relevant documents for a query.
- Constructs a prompt combining the query and retrieved context.
- Sends the prompt to the language model (GPT-4o) to generate a natural language answer.

**Example:**
```python
User Query: What was Microsoft's first hardware product release?

--- Model Response ---
Microsoft's first hardware product release was the Microsoft Mouse in 1983.
```


## Function Reference (`4_history_generation.py`)

Run the script to start an interactive chat session:
```bash
python 4_history_generation.py
```

- Maintains a chat history and rewrites new queries to be standalone based on the history.
- Retrieves relevant documents for the rewritten query.
- Answers using the retrieved context.

**Example:**
```python
Ask question, and type 'exit' to quit.
Enter question: What was Microsoft's first hardware product release?

User Query: What was Microsoft's first hardware product release?

Found 5 relevant documents:
Document 1 Preview:
1985–1994: Windows and Office
Microsoft released Windows 1.0 on November 20, 1985, as a
graphical extension for MS-DOS,[10]: 242–243, 246  despite
...
Document 5 Preview:
Microsoft has been market-dominant in the IBM PC– Brad Smith (vice chairman &
compatible operating system market and the office president)
software suite market since the 1990s. Its best-known Bill Gates (technical adviser)

Answer: Microsoft's first hardware product release was the Microsoft Mouse in 1983.
Enter question: Where was it released?

User Query: Where was it released?

Searching for: Where was the Microsoft Mouse first released?
...
```

## Function Reference (`5_character_text_splitter.py`)

Demonstrates basic text splitting using a fixed character count.

```bash
python 5_character_text_splitter.py
```

- **Usage**: Good for simple splitting where context preservation isn't critical.
- **Key Class**: `CharacterTextSplitter` from `langchain_text_splitters`.
- **Mechanism**: Splits text based on a single separator (e.g., " ") and a maximum chunk size.

**Example:**
```python
Chunk 1: (97 chars)
"Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The"

Chunk 2: (86 chars)
"Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production"

Chunk 3: (97 chars)
"Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long"

Chunk 4: (94 chars)
"paragraph that definitely exceeds our 100 character limit and has no double newlines inside it"

Chunk 5: (50 chars)
"whatsoever making it impossible to split properly."
```

## Function Reference (`6_recursive_character_text_splitter.py`)

Demonstrates smarter text splitting that tries to keep related text together.

```bash
python 6_recursive_character_text_splitter.py
```

- **Usage**: Recommended for most text document tasks.
- **Key Class**: `RecursiveCharacterTextSplitter`.
- **Mechanism**: Recursively tries different separators (paragraphs `\n\n`, newlines `\n`, spaces ` `) to find the best place to split while keeping chunks under the size limit.

**Example:**
```python
Same problem text, but with RecursiveCharacterTextSplitter:
Chunk 1: (92 chars)
"Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance"

Chunk 2: (78 chars)
"The Model Y became the best-selling vehicle globally, with 350,000 units sold."

Chunk 3: (85 chars)
"Production Challenges

Supply chain issues caused a 12% increase in production costs."

Chunk 4: (97 chars)
"This is one very long paragraph that definitely exceeds our 100 character limit and has no double"

Chunk 5: (69 chars)
"newlines inside it whatsoever making it impossible to split properly."
```

## Function Reference (`7_semantic_chunker.py`)

Demonstrates splitting text based on semantic meaning rather than just character counts.

```bash
python 7_semantic_chunker.py
```

- **Usage**: Best for splitting deeply semantic text where topic boundaries vary in length.
- **Key Class**: `SemanticChunker` from `langchain_experimental.text_splitter`.
- **Mechanism**: Uses embeddings (OpenAIEmbeddings) to calculate similarity between sentences. Splits when there is a significant change in semantic meaning (topic shift).

**Example:**
```python
SEMANTIC CHUNKING RESULTS:
==================================================
Chunk 1: (120 chars)
"Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024. The company exceeded analyst expectations by 15%."

Chunk 2: (363 chars)
"Revenue growth was driven by strong vehicle deliveries. Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold. Customer satisfaction ratings reached an all-time high of 96%. Model Y now represents 60% of Tesla's total vehicle sales. Production Challenges
Supply chain issues caused a 12% increase in production costs."

Chunk 3: (48 chars)
"Tesla is working to diversify its supplier base."

Chunk 4: (67 chars)
"New manufacturing techniques are being implemented to reduce costs."
```

## Function Reference (`8_agentic_chunking.py`)

Demonstrates using an LLM (Agent) to intelligently determine chunk boundaries.

```bash
python 8_agentic_chunking.py
```

- **Usage**: High-precision chunking where logical coherence is paramount.
- **Key Class**: Custom logic using `ChatOpenAI`.
- **Mechanism**: Feeds the text to an LLM with a prompt asking it to identify logical split points (e.g., inserting `<<<SPLIT>>>` markers).

**Example:**
```python
Asking the LLM to chunk the text...
Chunk 0: (178 chars)
Tesla's Q3 Results  
Tesla reported record revenue of $25.2B in Q3 2024. The company exceeded analyst expectations by 15%. Revenue growth was driven by strong vehicle deliveries.

Chunk 1: (222 chars)
Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold. Customer satisfaction ratings reached an all-time high of 96%. Model Y now represents 60% of Tesla's total vehicle sales.

Chunk 2: (203 chars)
Production Challenges  
Supply chain issues caused a 12% increase in production costs. Tesla is working to diversify its supplier base. New manufacturing techniques are being implemented to reduce costs.
```

## Function Reference (`9_multi_modal_rag.py`)

A comprehensive Multi-Modal RAG pipeline capable of processing PDFs containing text, images, and tables.

> **Note**: Requires additional system dependencies: `brew install poppler tesseract libmagic`.

```bash
python 9_multi_modal_rag.py
```

### Key Features:
1.  **PDF Ingestion**: Uses `unstructured` to parse high-resolution PDFs.
2.  **Multi-Modal Extraction**: enhancing:
    - **Text**: Extracted as standard text.
    - **Tables**: Extracted as HTML to preserve structure.
    - **Images**: Extracted as Base64 encoded strings.
3.  **AI-Enhanced Summarization**:
    - Each chunk containing images or tables is passed to GPT-4o-Vision.
    - The model generates a comprehensive text summary of the visual data.
    - This summary is what gets embedded for retrieval.
4.  **Multi-Modal Generation**:
    - When a user asks a question, the original images/tables are retrieved along with the text.
    - The final answer generation uses GPT-4o-Vision to "see" the retrieved charts/images and "read" the tables to provide a complete answer.

### Workflow steps in script:
1.  **Partition**: Extract raw elements (Text, Table, Image) from PDF.
2.  **Chunk**: Group elements intelligently by title.
3.  **Summarize**: Generate text descriptions for non-text elements (images/tables).
4.  **Vector Store**: Embed and store the summaries in ChromaDB.
5.  **Retrieval & Answer**: Retrieve relevant chunks + original images, pass to LLM for final answer.

**Example:**
```python
Starting RAG Ingestion Pipeline
==================================================
Partitioning ./docs/attention-is-all-you-need.pdf...
Warning: No languages specified, defaulting to English.
The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.
Extracted 220 elements
🔨 Creating smart chunks...
Created 25 chunks
Processing chunks with AI Summaries...
   Processing chunk 1/25
     Types found: ['text']
     Tables: 0, Images: 0
     → Using raw text (no tables/images)
   ...
   Processing chunk 5/25
     Types found: ['image', 'text']
     Tables: 0, Images: 1
     → Creating AI summary for mixed content...
     → AI summary created successfully
     → Enhanced content preview: **SEARCHABLE DESCRIPTION:**

**Key Facts and Data Points:**
- The document discusses neural sequence transduction models, specifically focusing on the encoder-decoder structure.
- The encoder maps an ...
   Processing chunk 6/25
     Types found: ['text']
     Tables: 0, Images: 0
     → Using raw text (no tables/images)
   Processing chunk 7/25
     Types found: ['image', 'text']
     Tables: 0, Images: 2
     → Creating AI summary for mixed content...
     → AI summary created successfully
     → Enhanced content preview: **SEARCHABLE DESCRIPTION:**
   ...
   Processing chunk 25/25
     Types found: ['image', 'text']
     Tables: 0, Images: 4
     → Creating AI summary for mixed content...
     → AI summary created successfully
     → Enhanced content preview: **Searchable Description:**

1. **Key Facts, Numbers, and Data Points:**
   - The document discusses attention mechanisms in neural networks, specifically focusing on encoder self-attention in layer 5...
Processed 25 chunks
Creating embeddings and storing in ChromaDB...
--- Creating vector store ---
--- Finished creating vector store ---
Vector store created and saved to dbv2/chroma_db
Pipeline completed successfully!
The Transformer uses 8 attention heads, and the dimension of each head is 64.
```

## Function Reference (`10_retrieval_methods.py`)

Demonstrates various retrieval techniques using ChromaDB and LangChain.

```bash
python 10_retrieval_methods.py
```

### Retrieval Methods Demonstrated:

1.  **Similarity Search (`k=3`)**:
    - Standard retrieval based on cosine similarity.
    - Returns the `k` most similar documents.

    **Example:**
    ```python
    Query: How much did Microsoft pay to acquire GitHub?

    === METHOD 1: Similarity Search (k=3) ===
    Retrieved 3 documents:

    Document 1:
    119. "Microsoft completes GitHub acquisition" (https://web.archive.org/web/20190112212059/http
    s://www.msn.com/en-us/news/technology/microsoft-completes-github-acquisition/ar-BBOVV
    OT). www.msn.com. Archived from the original (https://www.msn.com/en-us/news/technolog
    y/microsoft-completes-github-acquisition/ar-BBOVVOT) on January 12, 2019. Retrieved
    April 10, 2019.

    Document 2:
    117. "Microsoft's 2018, part 1: Open source, wobbly Windows and everyone's going to the cloud"
    (https://www.theregister.co.uk/2018/12/25/microsoft_year_in_review_2018/). The Register.
    Archived (https://web.archive.org/web/20190103060059/https://www.theregister.co.uk/2018/
    12/25/microsoft_year_in_review_2018/) from the original on January 3, 2019. Retrieved
    January 3, 2019.

    118. "Microsoft to acquire GitHub for $7.5 billion" (https://news.microsoft.com/2018/06/04/microso
    ft-to-acquire-github-for-7-5-billion/). Microsoft. June 4, 2018. Archived (https://web.archive.or
    g/web/20180604142244/https://news.microsoft.com/2018/06/04/microsoft-to-acquire-github-
    for-7-5-billion/) from the original on June 4, 2018.

    Document 3:
    In April 2018, Microsoft released the source code for Windows File Manager under the MIT License to
    celebrate the program's 20th anniversary. In April the company further expressed willingness to embrace
    open source initiatives by announcing Azure Sphere as its own derivative of the Linux operating
    ...
    ```

2.  **Similarity Search with Threshold**:
    - Sets a minimum `score_threshold` (e.g., 0.3).
    - Filters out low-relevance documents.

    **Example:**
    ```python
    === METHOD 2: Similarity with Score Threshold ===
    Retrieved 3 documents (threshold: 0.3):

    Document 1:
    119. "Microsoft completes GitHub acquisition" (https://web.archive.org/web/20190112212059/http
    s://www.msn.com/en-us/news/technology/microsoft-completes-github-acquisition/ar-BBOVV
    OT). www.msn.com. Archived from the original (https://www.msn.com/en-us/news/technolog
    y/microsoft-completes-github-acquisition/ar-BBOVVOT) on January 12, 2019. Retrieved
    April 10, 2019.

    Document 2:
    117. "Microsoft's 2018, part 1: Open source, wobbly Windows and everyone's going to the cloud"
    ...

    118. "Microsoft to acquire GitHub for $7.5 billion" 
    ...

    Document 3:
    In April 2018, Microsoft released the source code for Windows File Manager under the MIT License to
    celebrate the program's 20th anniversary. In April the company further expressed willingness to embrace
    open source initiatives by announcing Azure Sphere as its own derivative of the Linux operating
    system.
    ...
    ```

3.  **Maximum Marginal Relevance (MMR)**:
    - Balances **Relevance** (similarity to query) and **Diversity** (dissimilarity among results).
    - `fetch_k`: Initial pool of documents.
    - `lambda_mult`: Controls trade-off (0=Max Diversity, 1=Max Relevance).

    **Example:**
    ```python
    === METHOD 3: Maximum Marginal Relevance (MMR) ===
    Retrieved 3 documents (λ=0.5):

    Document 1:
    119. "Microsoft completes GitHub acquisition" (https://web.archive.org/web/20190112212059/http
    s://www.msn.com/en-us/news/technology/microsoft-completes-github-acquisition/ar-BBOVV
    OT). www.msn.com. Archived from the original (https://www.msn.com/en-us/news/technolog
    y/microsoft-completes-github-acquisition/ar-BBOVVOT) on January 12, 2019. Retrieved
    April 10, 2019.

    Document 2:
    In April 2018, Microsoft released the source code for Windows File Manager under the MIT License to
    celebrate the program's 20th anniversary. In April the company further expressed willingness to embrace
    open source initiatives by announcing Azure Sphere as its own derivative of the Linux operating
    system.
    ...

    Document 3:
    €899  million ($1.4  billion) for Microsoft's lack of compliance with the March 2004 judgment on
    February 27, 2008, saying that the company charged rivals unreasonable prices for key information about
    its workgroup and backoffice servers.[58] Microsoft stated that it was in compliance and that "these fines
    are about the past issues that have been resolved".[59] 2007 also saw the creation of a multi-core unit at
    Microsoft, following the steps of server companies such as Sun and IBM.[60]
    ```

## Function Reference (`11_multi_query_retrieval.py`)

Demonstrates how to improve retrieval quality by generating multiple search queries from a single user question.

```bash
python 11_multi_query_retrieval.py
```

### Mechanism

1.  **Query Generation**: Uses an LLM (GPT-4o) to generate 3 different variations of the original user query.
2.  **Multi-Search**: Performs a separate vector search for *each* generated variation.
3.  **Result Aggregation**: Collects unique documents from all search results (in this script, results are printed per query to show the difference).

**Why this matters**: Different phrasings of the same question might return different documents due to how embeddings work. searching multiple angles increases the chance of finding the "correct" context that might have been missed by a single query.

**Example**:
```python
Original Query: How does Tesla make money?

Generated Query Variations:
1. What are Tesla's primary revenue streams?
2. Can you explain Tesla's business model and income sources?
3. How does Tesla generate profit from its operations?

=== RESULTS FOR QUERY 1: What are Tesla's primary revenue streams? ===
Retrieved 5 documents...

=== RESULTS FOR QUERY 2: Can you explain Tesla's business model and income sources? ===
Retrieved 5 documents...
```
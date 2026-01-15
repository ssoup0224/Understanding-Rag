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

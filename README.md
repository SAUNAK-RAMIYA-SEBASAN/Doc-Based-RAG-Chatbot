#### Document-Based RAG Chatbot

## 1. Introduction

This project implements a **Document-Based Retrieval-Augmented Generation (RAG) Chatbot** designed to answer user questions **strictly from the content** contained within curated PDF and DOCX documents. The chatbot operates **completely offline**, making it ideal for secure, private, or regulated environments where no data leaves the local system. By combining semantic search with powerful language generation, this system delivers accurate, grounded, and auditable answers with clear source citations.

### What is Retrieval-Augmented Generation (RAG)?

**RAG** is a hybrid AI architecture that enhances language model capabilities by integrating an external knowledge retrieval step. It works in two stages: 

- **Retrieval:** The system searches for the most relevant pieces of document content related to a user query, using vector similarity search.
- **Generation:** A language model then generates precise and natural language answers based on only the retrieved content, ensuring factual accuracy and avoiding hallucination.

This retrieval-first approach balances scalability, factual grounding, and user experience, enabling the system to answer queries strictly based on actual document data.

### Key Features of Our Project

- **Fully Offline Operation:** All AI models and document data are locally stored and processed. No internet connection is required after initial setup.
- **Robust Hallucination Prevention:** The system only attempts to answer questions if relevant document content is found; otherwise, it explicitly returns "The answer is not found in the document."
- **Multi-Document Support:** Handles multiple PDF and DOCX files simultaneously, automatically chunking and indexing their contents for semantic search.
- **Precise Source Citations:** Every generated answer is accompanied by exact source metadata—filename, page number, section name, and chunk ID—for full traceability.
- **Efficient Performance:** First query response (including model loading) takes approximately 30 to 40 seconds on CPU-only environments; subsequent queries are answered within 15 seconds.
- **Professional User Interface:** Clean, intuitive Streamlit web interface that displays answers, confidence scores, detailed citations, and relevant document context.
- **Open Source and Vendor-Free:** Fully open-source stack with no reliance on cloud APIs, LangChain, or proprietary dependencies.

### Technology Stack

- **Embedding Model:** *intfloat/e5-small-v2* — a fast and accurate transformer-based model to convert user queries and document chunks into semantic embeddings.
- **Generative Language Model:** *TinyLlama-1.1B-Chat-v1.0* — a compact LLM that generates fluent, focused answers based solely on retrieved document content.
- **Vector Database:** *Qdrant (Docker)* — an open-source vector search engine used locally for efficient similarity search among document chunks.
- **User Interface:** *Streamlit* — provides a responsive and user-friendly chat interface.
- **Environment & Dependency Management:** Python 3.12.x with **UV** package manager for fast and reproducible environment setups.
- **Deployment:** Docker Compose setup for easy local orchestration of vector database.

### Project Architecture Overview

The chatbot’s architecture follows a modular Retrieval-Augmented Generation pipeline:

1. **Document Processing:** PDF and DOCX files are chunked into semantically coherent pieces with metadata.
2. **Embedding Engine:** Chunks and user queries are converted into vector embeddings using the E5-Small-v2 transformer.
3. **Vector Search:** The system performs nearest neighbor search in Qdrant to find top-k relevant chunks for the query.
4. **Answer Generation:** TinyLlama generates the final concise (1-2 sentences) answer using only the retrieved chunk text, preventing hallucinations.
5. **Source Attribution:** Citations comprising filename, page, section, and chunk info are attached with the answer.
6. **User Interaction:** The Streamlit UI handles user queries and displays answers with confidence scores and contextual information.

### Project Folder Structure

```
RAG-Chatbot/
├── .env                           # Environment configuration variables
├── docker-compose.yml             # Docker setup for Qdrant vector DB
├── requirements.txt               # Project Python dependencies
├── README.md                     # Project documentation (this file)
├── documents/                    # Folder containing PDF and DOCX files
│   ├── cricket_manual.pdf        # Sample cricket rules document
│   └── LawsOfChess.docx          # Sample chess laws document
└── src/                         # Source code
    ├── __pycache__/             # Auto-generated Python cache files
    ├── document_processor.py    # Document chunking and processing logic
    ├── embedding_engine.py      # Embedding model loading and inference
    ├── vector_db_manager.py     # Qdrant vector database interface
    ├── llm_pipeline.py          # Answer generation pipeline with TinyLlama
    ├── setup_local_db.py        # Script for document processing and DB initialization
    └── streamlit_app.py         # Streamlit UI application
```

## 2. Setup Guide

Setting up the Document-Based RAG Chatbot is streamlined to ensure reliable, fully offline operation after a one-time model download. Follow the instructions below to configure your system on Windows or a compatible platform.

### Prerequisites

Before you start, ensure that you have the following software installed:

- **Python 3.12.x** ([Download Python](https://www.python.org/)[1])
- **Docker Desktop** ([Download Docker](https://www.docker.com/)[2])
- **Git** ([Download Git](https://git-scm.com/)[3])
- **UV Package Manager** ([UV Documentation & Install Guide](https://docs.astral.sh/uv/)[4])

#### UV Installation

UV is a fast Python package and environment manager that makes setup easy and reproducible.

```powershell
# Install UV using PowerShell (recommended)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```
(See the [UV documentation][4] for details and Linux/macOS install options.)

### Project Setup Steps

#### 1. Clone the Repository

```powershell
git clone 
cd Doc-Based-RAG-Chatbot
```

#### 2. Create and Activate the Virtual Environment

```powershell
uv venv --python 3.12
.venv\Scripts\activate

# Confirm Python version
python --version
# Output should be: Python 3.12.x
```

#### 3. Install Dependencies

First, install the CPU-only version of PyTorch for optimal compatibility:

```powershell
uv pip install torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

Then, install all remaining project dependencies:

```powershell
uv pip install -r requirements.txt
```

#### 4. Start Local Qdrant Vector Database

Open Docker Desktop and ensure it is running, then:

```powershell
docker-compose up -d
```

- This will pull and start a local Qdrant vector database container used for storing document embeddings.
- You can verify the container is running by checking:
```powershell
docker ps
```
(Qdrant should appear with status "Up")

#### 5. Process Documents and Initialize the Vector Database

Make sure your PDFs and DOCX files to be used by the chatbot are in the `documents/` folder.

Process all supported documents and build the vector database:

```powershell
python src/setup_local_db.py
```

- All `.pdf` and `.docx` located in `documents/` will be chunked, embedded, and loaded into Qdrant.
- If successful, you’ll see confirmation messages indicating document chunks and embeddings have been stored.

#### 6. Launch the Chatbot

```powershell
streamlit run src/streamlit_app.py
```

- Open the link shown in your console (typically http://localhost:8501) in your web browser to access the chat interface.

### Usage Notes

- **First Query:** When you enter your first question in the chat UI, the system will download and load all required AI models **(requires internet for this very first use only)**. This may take 30–40 seconds to complete, especially on CPU-only machines.
- **Subsequent Queries:** After the initial model load, all future questions are answered fully offline, typically within 15 seconds per query.
- **RAM Requirements:** At least 4GB of free memory is required to load and run the TinyLlama model smoothly.

### Links for Software and Tools

- [Python](https://www.python.org/)[1]
- [Docker Desktop](https://www.docker.com/)[2]
- [Git](https://git-scm.com/)[3]
- [UV Package Manager](https://docs.astral.sh/uv/)[4]

Your chatbot is now ready to use! For the first query, models will be fetched as needed; after that, the system runs 100% offline for all subsequent sessions, delivering fast, accurate, document-grounded answers with full traceability and zero cloud dependencies.

## 3. Usage Guide

This section outlines how to effectively use the Document-Based RAG Chatbot after setup.

### 3.1 Starting the System

1. **Ensure the Qdrant vector database is running:**

   ```bash
   docker-compose up -d
   ```

2. **Launch the chatbot UI:**

   ```bash
   streamlit run src/streamlit_app.py
   ```

3. **Open your browser** to the URL displayed in the terminal, typically `http://localhost:8501`.

### 3.2 Interacting with the Chatbot

- **Enter your questions** in the input box provided.
- Example supported queries:
  - "How many players are there in a cricket team?"
  - "How the rook may move?"
- The chatbot retrieves relevant document chunks, then uses the TinyLlama model to generate short (1-2 sentence) answers strictly grounded on those chunks.
- **Non-document-related queries** (e.g., questions about Bitcoin prices or recipes) will result in a response stating:  
  “The answer is not found in the document.”

### 3.3 Response Format

- Answers are accompanied by a confidence score (0 to 1 scale).
- **Source citations** (filename, page number, section, chunk ID) are displayed with each answer.
- Relevant contextual sentences from source documents can be expanded for verification.
- Detailed full text of cited chunks is available on demand.

### 3.4 Performance Expectations

- **First Query:** May take approximately **30 to 40 seconds** due to model loading on CPU.
- **Subsequent Queries:** Typically respond within **15 seconds**.
- *Note:* Performance measured on Intel-class CPU without GPU acceleration.

### 3.5 Tips for Optimal Use

- Ensure documents in the `/documents` folder are well-structured and clear.
- Use concise, specific queries for best results.
- After initial use, the system operates fully offline, suitable for private or air-gapped environments.

## 4. Architectural Decisions

This section details the technical architecture, design choices, and trade-offs made in building the chatbot.

### 4.1 Retrieval-Augmented Generation (RAG) Architecture

- The system integrates two key components: **retrieval** and **generation**.
- **Retrieval:** The user query is embedded using the efficient E5-Small-V2 model and matched against vectorized document chunks stored in a local Qdrant database.
- **Generation:** Top relevant chunks are passed to TinyLlama, which strictly generates concise answers limited to the retrieved context.
- This design **mitigates hallucination** by ensuring answers are not fabricated but derived only from indexed documents.

### 4.2 Document Processing and Chunking

- PDFs and DOCX files are processed into manageable textual chunks.
- Chunks are sized ~384 tokens with 64-token overlap, preserving semantic context and aiding retrieval accuracy.
- Text splitting is sentence-aware to prevent fragmenting sentences mid-way.

### 4.3 Embedding and Vector Search

- The **E5-Small-V2** model transforms document chunks and queries into 384-dimensional embeddings.
- **Qdrant** serves as the locally deployed vector database, offering fast similarity search with cosine distance metrics.
- The system respects hackathon rules by executing a **single embedding query per user input** for retrieval.

### 4.4 Language Model and Inference

- The generative model used is **TinyLlama 1.1B** — a small, efficient LLM capable of running on CPU-only environments with moderate RAM.
- To control hallucination, the model is prompted with **only relevant document chunks (metadata stripped)** and **strict instructions** to respond with brief, direct answers.
- The first query triggers model loading (hence longer response), subsequent queries benefit from warm model cache.

### 4.5 System Security and Privacy

- All components run **entirely on the user’s local machine**.
- No communication with external servers or cloud APIs post initial model downloads.
- This enables deployment in sensitive or air-gapped environments, fully complying with data privacy and security requirements.

### 4.6 User Interface Design

- Built on **Streamlit**, providing a responsive and user-friendly web UI.
- Includes sidebars displaying system resource info (memory free, queries handled).
- Answers display with confidence, citations, and optional full source content expanders.

### 4.7 Scalability and Future Enhancements

- The modular design allows the addition of new document types or larger document corpora.
- Potential upgrades include GPU acceleration, better chunking heuristics, and model fine-tuning for domain-specific accuracy.

### 4.8 Limitations and Trade-Offs

- Performance is constrained by CPU-only inference; first-load latency can be noticeable.
- Model size selected balances between resource availability and answer quality.
- Strict “no hallucination” enforcement sometimes leads to terse or incomplete answers when the document provides limited info.

## 5. Observations

This section summarizes empirical findings, performance insights, accuracy observations, and technical challenges encountered while building and testing the Document-Based RAG Chatbot.

### 5.1 Performance Observations

- **First Query Load:** The system takes approximately **30–40 seconds** to load models and initialize on the very first query in a CPU-only environment. This is expected and occurs only once per session.
- **Subsequent Queries:** All following queries are answered reliably in **under 15 seconds**, thanks to local model caching and efficient pipeline processing.
- **Resource Usage:** The application was tested on standard CPUs with a minimum of 4GB RAM available; usage stays stable after initialization, making it practical for offline and edge environments.

### 5.2 Accuracy and Reliability Insights

- **Zero Hallucination:** The system strictly answers only if document evidence exists. Non-document questions and ambiguous queries reliably return:  
  *“The answer is not found in the document.”*
- **Precise Source Attribution:** Each answer is accompanied by citations, including file name, page, section, and chunk ID, allowing for transparent traceability.
- **Brevity and Focus:** Model answers are short, clear, and specific—usually no more than one or two sentences—helping prevent over-explanation or irrelevant details.
- **Data Integrity:** Answers always match the actual content of the retrieved chunks; no summaries, embellishments, or externally sourced text are ever included.

### 5.3 Robustness and Usability

- **Offline Operation:** Once models are downloaded, the entire chatbot (including the UI and vector search) functions fully offline—ideal for secure, air-gapped, or privacy-sensitive use cases.
- **Multi-Document Handling:** The system easily scales to multiple PDFs and DOCX files. New documents are processed simply by placing them in the `documents/` folder and rerunning the setup.
- **Consistent UI Experience:** Streamlit ensures smooth user interaction, clear loading indicators, and easy review of answers, citations, context, and processing time.
- **Graceful Failures:** Errors in document processing, failed queries, or missing data produce clear warnings and do not crash the application.

### 5.4 Limitations Discovered

- **First-Load Delay:** There is an unavoidable wait for the initial model download and load on first query; this is a trade-off for compact model size and local execution.
- **Strictness May Limit Answers:** If the source documents contain only partial or no information, the system will respond with "not found" even if a best-guess might help—by design, to prevent hallucination.
- **CPU-Only Constraints:** Response time is dependent on CPU performance; heavy concurrent user load or large document sets may increase latency.

### 5.5 Real-World Suitability

- **Ideal For:** Anyone requiring verifiable, document-based Q&A in legal, regulatory, company internal, or research settings—especially where security, transparency, and offline operation are priorities.
- **Not Ideal For:** Open-ended conversational chat, open-domain questions, or settings where fabricated or speculative answers are acceptable.
- **Easy Extension:** Adding new domains is as simple as dropping new documents into the folder and re-initializing—the system adapts without code changes.

These observations confirm that the Document-Based RAG Chatbot delivers on its promise of reliable, auditable, offline Q&A with strict adherence to document-grounded information, robustly preventing hallucination and unauthorized data egress.

## 6. Chunking Strategy

### 6.1 What is Chunking and Why is it Needed?

In Retrieval-Augmented Generation (RAG) systems, “chunking” refers to splitting each document into smaller, semantically meaningful units called “chunks.” This is crucial because:
- Language models and embedding models have maximum input lengths—full documents almost always exceed these limits.
- Smaller, targeted chunks enable more precise and relevant retrieval for each user query.
- Chunks allow you to cite the exact location in the document that an answer comes from.

### 6.2 How Chunking Works in This Project

**Chunking Process:**
- When you add a PDF or DOCX file to the `documents/` folder, the project automatically processes each file.
- It breaks the document into chunks—each chunk typically contains about **384 tokens** (words or word pieces), with a **64-token overlap** between consecutive chunks.
- Chunking respects sentence boundaries as much as possible (sentence-aware splitting), so that sentences are rarely split in the middle. This preserves the natural context and meaning of the text.

**Why use a 384-token size and 64-token overlap?**
- **384 tokens** is chosen because embedding and LLM models (like E5-small-v2 and TinyLlama) typically have context window limits of 512 tokens or less. Reserving room for prompts and queries, 384 tokens per chunk is a “sweet spot” balancing context length and model compatibility.
- **64-token overlap** prevents information loss at chunk borders, ensuring details that might straddle two chunks are not missed by the retriever.

**Metadata:**  
Each chunk includes precise metadata: filename, page number, section name, and a unique chunk ID. This enables accurate source citation in answers.

### 6.3 Chunking Example

Suppose a document page contains:
```
Cricket is played between two teams of eleven players each. The game is played on a circular field. The captain leads the team...
```
- The project will split this page into chunks of up to 384 tokens.
- If the next chunk starts before the previous chunk's end (because of the 64-token overlap), overlapping content ensures facts at chunk boundaries remain accessible during retrieval.

### 6.4 Why Sentence-Aware Chunking?

- Avoids splitting sentences mid-way, which can confuse both retrievers and LLMs.
- Maintains context, so answers don’t get fragmented or lose their meaning.
- Ensures that when a relevant answer is found, it’s almost always in a readable, complete form—helping both accuracy and citation.

### 6.5 Benefits for Retrieval and Generation

**Retrieval:**
- Each user query is embedded and compared to all chunk embeddings in the database (Qdrant). By chunking, you optimize retrieval granularity—making it possible to find the *most relevant* text.
- Overlap ensures the right context is not missed even if it was near a block boundary.

**Generation:**
- TinyLlama receives as context only the text chunks most related to the user’s question. This keeps answers accurate, focused, and less prone to hallucination.
- Compact, sentence-complete chunks enable the language model to synthesize coherent, truthful answers.

### 6.6 Summary of Approach

- **384-token chunks**: Optimized for model limits while providing rich context.
- **64-token overlap**: Prevents info loss and improves coverage.
- **Sentence-aware splitting**: Preserves context and readability.
- **Metadata-rich chunks**: Enables exact citation and transparency for every answer.

This chunking strategy is key to the chatbot’s ability to deliver fast, precise, document-grounded answers with clear sources—ensuring reliable and auditable RAG performance.

## 7. Retrieval Approach

### 7.1 Embedding-Based Semantic Retrieval

- **Vectorization Process:** Every document chunk and incoming user query is converted into a 384-dimensional vector using the `intfloat/e5-small-v2` embedding model.
- **Query Execution:** When a user submits a question, it is embedded and a **single vector similarity search** is performed in Qdrant, the local vector database.
- **Top-K Selection:** The system retrieves the top 5 most semantically similar chunks from all indexed documents.
- **Context Preparation:** Only the raw content of these retrieved chunks (no extra metadata) is passed as context to the LLM, ensuring answer generation is strictly document-grounded.
- **Strict Filtering:** If none of the chunks pass a minimal relevancy threshold, the system returns “The answer is not found in the document” and skips LLM invocation, avoiding hallucination.

**Benefits:**  
This approach ensures answers are contextually relevant, document-sourced, and explainable—every answer can be traced to its supporting chunk.

### 7.2 Query and Retrieval Workflow

1. **User query is received** in the Streamlit UI.
2. **Embedding engine** encodes the query.
3. **Vector similarity search** is performed in Qdrant, retrieving the best-matching document chunks.
4. The **TinyLlama LLM** uses only those chunks to generate a short, direct answer.
5. **Answer, confidence, and citations** are displayed, with full chunk content available for transparency.

## 8. Hardware Requirements

### 8.1 Minimum and Recommended Specs

- **CPU:** Modern multi-core processor (Intel/AMD; no GPU required, but faster CPUs yield better response times)
- **RAM:** At least **4GB free memory required** for initial TinyLlama model load (total system RAM should be ≥8GB for smooth use)
- **Disk Space:** 10GB+ recommended (to store all models, embeddings, and database files)
- **Environment:** Built and tested on Windows; works on Linux/Mac with minor path adjustments
- **Docker:** Required for Qdrant vector database container (ensure Docker Desktop is running)

### 8.2 Performance Notes

- **First Query:** 30–40 seconds (due to initial model loading and embedding caching)
- **Subsequent Queries:** <15 seconds each (CPU-only)
- **No GPU Needed:** Entire stack runs via CPU, making it accessible for almost any modern laptop or workstation
- **Disk Usage:** AI model files and embedding cache are stored locally, ensuring ongoing offline performance.

**Tip:** For even smoother operation (especially with larger document sets), 8GB–16GB RAM is recommended, but 4GB is the enforced minimum for TinyLlama to load.

## 9. Conclusion

The Document-Based RAG Chatbot is a robust, secure, and **truly offline** system for question answering over your documents. It combines advanced semantic retrieval with a local language model, enforcing strict document-grounded answers while providing complete source transparency.

**Key Benefits:**
- **No hallucination:** Answers are only given when verified by document content.
- **Full privacy:** Runs with zero data leaving your device after setup.
- **Open and extensible:** Easily add new documents, swap models, or extend to new domains.
- **User-friendly and auditable:** All answers are cited; UI makes tracing and verifying sources easy.

Whether for research, business, legal, or regulated environments, this chatbot is ideal for anyone who needs trustworthy, cited information from their own document collections—without the risks and trade-offs of cloud dependency.

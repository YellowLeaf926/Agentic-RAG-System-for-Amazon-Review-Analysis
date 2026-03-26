# Amazon Shopping App Review Analyzer
### Advanced Agentic RAG with LoRA Fine-Tuning

Different systems are designed to assess information retrival and answers that questions about Amazon Shopping app user reviews. A simple LLM, simple RAG, advanced RAG with base model, adavance rag with lora tuned model are used. The pipeline combines a Pinecone vector store, GPT-4o-mini agents, and a locally fine-tuned SmolLM2 model to deliver context-aware, high-quality answers through a Flask web interface.

---

## Project Structure

```
.
├── src/
│   ├── main1.py                  # Data ingestion: chunking, embedding, Pinecone upload
│   ├── main2.py                  # Basic RAG: simple LLM and simple retriever+LLM chain
│   ├── main3.py                  # Advanced agentic RAG with LangGraph + Flask server
│   ├── lora.py                   # LoRA fine-tuning of SmolLM2 on review summarization
│   ├── parameters_example.py     # Template for API keys and config (copy → parameters.py)
│   ├── parameters.py             # (gitignored) Your actual API keys and config
│   └── templates/
│       └── index.html            # Flask web UI
├── data/                         # (gitignored) CSV reviews and generated training data
├── logs/                         # Auto-generated run logs
├── model_checkpoints/            # (gitignored) Saved LoRA adapter weights
├── requirements.txt              # Python dependencies
├── sample_output.md              # Sample outputs and discussion
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/NUMLDS/stitching-project-YellowLeaf926.git
cd stitching-project-YellowLeaf926
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for `lora.py`.

### 3. Download the dataset

Download the Amazon Shopping app reviews CSV (https://www.kaggle.com/datasets/ashishkumarak/amazon-shopping-reviews-daily-updated) and place it at:

```
data/amazon_reviews.csv
```

### 4. Configure parameters

```bash
cp src/parameters_example.py src/parameters.py
```

Then edit `parameters.py` and fill in your actual values:

```python
# DATA CONFIGURATION
DATA_PATH = "data/your_dataset.csv"  # Path to your Amazon reviews CSV

# TEXT CHUNKING PARAMETERS
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap between consecutive chunks

# EMBEDDING MODEL
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-xs"

# PINECONE CONFIGURATION
PINECONE_INDEX = "amazon-reviews"     # Your Pinecone index name
PINECONE_REGION = "us-east-1"         # Your Pinecone region
PINECONE_API_KEY = "your-api-key"     # Replace with your actual Pinecone API key
NAMESPACE = "default"                 # Namespace for organizing vectors

# LOGGING
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# OpenAI
OPENAI_API_KEY = "your-openai-api-key" # Replace with your actual OpenAI API key

# top-k results to retrieve from Pinecone
TOP_K = 10
```

> **Never commit `src/parameters.py` to your repository.**

---

## Running the Pipeline

### Step 1 — Build the vector store

Chunks training reviews, embeds them, and uploads to Pinecone. Also runs a retrieval quality test on held-out reviews.

```bash
python src/main1.py
```

### Step 2 — Fine-tune SmolLM2 with LoRA (optional but recommended)

Generates GPT-4o-mini summaries of review batches as training data, then fine-tunes SmolLM2-1.7B-Instruct with LoRA. Saves the adapter to `model_checkpoints/final/`.

```bash
python src/lora.py
```

Training data is cached to `data/amazon_summaries.json` after the first run and reused automatically.

### Step 3 — Evaluate basic LLM and basic RAG

Runs the same query list through a plain GPT-4o-mini call and a simple retriever+LLM chain, logging results for comparison.

```bash
python src/main2.py
```

### Step 4 — Run the advanced agentic RAG system

Starts a Flask web server on `http://localhost:5000`. Use `--model lora` (default) to use the fine-tuned SmolLM2 adapter, or `--model base` for the untuned base model (ablation).

```bash
# With LoRA fine-tuned adapter (default)
python src/main3.py --model lora

# Ablation: untuned base model
python src/main3.py --model base
```

Open your browser at `http://localhost:5000` and type a question.

---

## Architecture

The advanced RAG system is built as a **LangGraph state machine** with six nodes:

```
classify → retrieve → [grade | summarize]
                           ↓
                       rewrite ←→ grade
                           ↓
                        generate → END
                      summarize  → END
```

| Node | Model | Role |
|---|---|---|
| **classify** | GPT-4o-mini | Labels the query as `sentiment` or `factual` |
| **retrieve** | Pinecone + FastEmbed | Fetches top-K relevant review chunks |
| **grade** | GPT-4o-mini | Filters retrieved docs for relevance |
| **rewrite** | GPT-4o-mini | Rewrites query if no relevant docs found (max 2×) |
| **generate** | GPT-4o-mini | Answers factual queries from graded context |
| **summarize** | SmolLM2-1.7B (LoRA) | Summarizes sentiment/opinion queries locally |

Sentiment queries skip grading entirely and go straight to the fine-tuned local model, which was trained specifically on review summarization.

---

## Evaluation

Four system configurations are compared across a fixed set of 6 queries (logged automatically):

1. **Base LLM** — GPT-4o-mini with no retrieval (`main2.py`)
2. **Basic RAG** — GPT-4o-mini + Pinecone retriever, no agentic routing (`main2.py`)
3. **Advanced RAG + base model** — full LangGraph pipeline, SmolLM2 untuned (`main3.py --model base`)
4. **Advanced RAG + LoRA model** — full LangGraph pipeline, SmolLM2 fine-tuned (`main3.py --model lora`)

Sample queries used for evaluation:

- Why do many users report that the Amazon Shopping app no longer works on tablets?
- What are the most common complaints after the latest Amazon Shopping app updates?
- What issues are most frequently mentioned in 1-star reviews?
- What do users like and dislike the most about the app?
- Based on recent user reviews, should I install or update the app? Why or why not?
- Any complaints about the app's haptic feedback calibration?

All outputs are written to timestamped files in `logs/`.
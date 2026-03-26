import argparse
import logging
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from flask import Flask, request, jsonify, render_template
from main2 import load_vectorstore, CustomRetriever, format_docs
from parameters import OPENAI_API_KEY
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -----------------------------------------------------------------------
# State
# -----------------------------------------------------------------------
class RAGState(TypedDict):
    query: str
    query_type: str
    documents: List[Document]
    graded_documents: List[Document]
    answer: str
    rewrite_count: int


# -----------------------------------------------------------------------
# Node functions
# All nodes except summarize_node use GPT-4o-mini (via llm).
# summarize_node uses the local SmolLM2 model, which is either:
#   --model lora  → LoRA fine-tuned adapter (default)
#   --model base  → untuned base model (ablation)
# -----------------------------------------------------------------------

def classify_node(state: RAGState) -> dict:
    chain = classify_prompt | llm | StrOutputParser()
    result = chain.invoke({"query": state["query"]}).strip().lower()
    if result not in ["sentiment", "factual"]:
        result = "factual"
    logger.info(f"Query classified as: {result}")
    return {"query_type": result}


def retrieve_node(state: RAGState) -> dict:
    docs = retriever.invoke(state["query"])
    logger.info(f"Retrieved {len(docs)} documents")
    return {"documents": docs}


def route_after_retrieve(state: RAGState) -> str:
    """Route to summarize for sentiment queries, grade for factual ones."""
    if state["query_type"] == "sentiment":
        return "summarize"
    return "grade"


def grade_node(state: RAGState) -> dict:
    grader = grade_prompt | llm | StrOutputParser()
    graded = []
    for doc in state["documents"]:
        result = grader.invoke({
            "query": state["query"],
            "document": doc.page_content[:500],
            "score": doc.metadata.get("score", "unknown"),
        })
        if result.strip().lower() == "yes":
            graded.append(doc)
    logger.info(f"Graded: {len(graded)}/{len(state['documents'])} docs relevant")
    return {"graded_documents": graded}


def route_after_grading(state: RAGState) -> str:
    if len(state["graded_documents"]) > 0:
        return "generate"
    if state["rewrite_count"] >= 2:
        return "generate"
    return "rewrite"


def rewrite_node(state: RAGState) -> dict:
    chain = rewrite_prompt | llm | StrOutputParser()
    new_query = chain.invoke({"query": state["query"]})
    logger.info(f"Rewritten query: {new_query}")
    return {"query": new_query, "rewrite_count": state["rewrite_count"] + 1}


def generate_node(state: RAGState) -> dict:
    """Factual path: uses GPT-4o-mini."""
    # FIX: fall back to all retrieved docs when graded list is empty
    # (happens when rewrite_count >= 2 and grading still yields nothing)
    docs = state["graded_documents"] if state["graded_documents"] else state["documents"]
    context = format_docs(docs)
    chain = generate_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["query"]})
    logger.info("Answer generated via factual path (model: gpt-4o-mini)")
    return {"answer": answer}


def call_summarize_llm(prompt_text: str) -> str:
    """Run inference on the local model chosen by --model flag.
    --model lora : LoRA fine-tuned SmolLM2 (trained on review summarization)
    --model base : untuned base SmolLM2 (ablation baseline)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(local_model.device)
    with torch.no_grad():
        outputs = local_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def summarize_node(state: RAGState) -> dict:
    """Sentiment path: uses local SmolLM2 (lora or base depending on --model)."""
    context = format_docs(state["documents"])
    # FIX: ChatPromptTemplate does not support .format(); use
    # .format_messages() and extract the human message content as a plain string.
    messages = summarize_prompt.format_messages(
        context=context, question=state["query"]
    )
    prompt_str = messages[0].content
    answer = call_summarize_llm(prompt_str)
    logger.info(f"Answer generated via sentiment path (model: {args.model})")
    return {"answer": answer}


# -----------------------------------------------------------------------
# Graph factory
# -----------------------------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("classify",  classify_node)
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("grade",     grade_node)
    graph.add_node("rewrite",   rewrite_node)
    graph.add_node("generate",  generate_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"grade": "grade", "summarize": "summarize"},
    )
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite",  "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("summarize", END)

    return graph.compile()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # --- CLI arguments ---------------------------------------------------
    parser = argparse.ArgumentParser(description="Advanced RAG with LangGraph")
    parser.add_argument(
        "--model",
        choices=["lora", "base"],
        default="lora",
        help=(
            "Model for the sentiment summarize path:\n"
            "  lora  — LoRA fine-tuned SmolLM2 adapter (default)\n"
            "  base  — untuned SmolLM2 base model (ablation)"
        ),
    )
    args = parser.parse_args()

    # --- Logging ---------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- Local model loading ---------------------------------------------
    BASE_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    LORA_CHECKPOINT = "model_checkpoints/final"

    # FIX: do NOT use device_map="auto" before PeftModel.from_pretrained.
    # Load to CPU first, wrap with PEFT, then move to device in one shot.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX: always load tokenizer from the base checkpoint, not the LoRA
    # output dir — the saved tokenizer_config.json there references
    # "TokenizersBackend" which transformers cannot deserialise.
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_CHECKPOINT,
        dtype=torch.bfloat16,   # match lora.py training dtype
    )

    if args.model == "lora":
        local_model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
        local_model = local_model.to(device)
        local_model.eval()
        logger.info(f"[summarize] LoRA adapter loaded from '{LORA_CHECKPOINT}'")
    else:
        local_model = base_model.to(device)
        local_model.eval()
        logger.info(f"[summarize] Using untuned base model '{BASE_CHECKPOINT}'")

    # --- GPT-4o-mini (classify / grade / rewrite / generate) -------------
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, seed=0)
    logger.info("[classify/grade/rewrite/generate] GPT-4o-mini ready")

    # --- Retriever -------------------------------------------------------
    vectorstore = load_vectorstore()
    retriever = CustomRetriever(vectorstore=vectorstore)

    # --- Prompts ---------------------------------------------------------
    classify_prompt = ChatPromptTemplate.from_template("""
You are classifying a user question about Amazon Shopping app reviews.
Classify the question into ONE of these two categories:

- "sentiment": user wants to know overall opinions, feelings, satisfaction levels
  Examples: "Is the app good?", "What do users think?", "Should I install it?"

- "factual": user wants specific facts, bugs, features, or concrete issues
  Examples: "Why does it crash?", "What bugs exist?", "Does it work on tablets?"

Question: {query}
Respond with ONLY one word: sentiment or factual.
""")

    grade_prompt = ChatPromptTemplate.from_template("""
You are grading whether a document is relevant to a user query.

Query: {query}
Document: {document}
Document Rating: {score} out of 5

Grading rules:
- If the query specifically asks about negative reviews, low ratings, or complaints
  (e.g. "1-star", "worst", "issues", "problems"), only mark as relevant if the
  document contains complaints or negative feedback (rating 1-2).
- If the query specifically asks about positive reviews, high ratings, or praises
  (e.g. "5-star", "best", "love", "like"), only mark as relevant if the document
  contains positive feedback (rating 4-5).
- Otherwise, mark as relevant if the document helps answer the query regardless
  of rating.

Is this document relevant? Respond with ONLY 'yes' or 'no'.
""")

    rewrite_prompt = ChatPromptTemplate.from_template("""
You are helping improve a search query that failed to retrieve relevant documents
from a database of Amazon Shopping app user reviews.

The query may have failed because:
- It uses technical or overly specific jargon that users would not write in a review
- It uses informal/slang words that don't match how reviews are written
- It is too vague or abstract

Rewrite the query using:
- Simple, everyday language that real users would use in app reviews
- Conversational tone similar to how someone would write a phone app review

Original query: {query}

Rewritten query:
""")

    generate_prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer concise.

Context: {context}
Question: {question}
Answer:
""")

    summarize_prompt = ChatPromptTemplate.from_template("""
You are an expert analyst specializing in mobile app user feedback.
Summarize the following Amazon Shopping app reviews:

1. Overall Sentiment: (positive / negative / mixed) with a one-line explanation
2. Key Praises: bullet points of what users liked most
3. Key Complaints: bullet points of what users disliked most

Only use information from the provided reviews. Be specific.

Reviews:
{context}

Question: {question}
Answer:
""")

    # --- Flask server ----------------------------------------------------
    flask_app = Flask(__name__)

    @flask_app.route("/")
    def index():
        return render_template("index.html")

    @flask_app.route("/ask", methods=["POST"])
    def ask():
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Empty query"}), 400

        logger.info(f"Received query: {query!r}")
        result = app.invoke({
            "query":            query,
            "query_type":       "",
            "documents":        [],
            "graded_documents": [],
            "answer":           "",
            "rewrite_count":    0,
        })

        logger.info(f"Answer: {result['answer']}")
        return jsonify({"answer": result["answer"]})

    logger.info("Starting Flask server on http://0.0.0.0:5000")
    flask_app.run(host="0.0.0.0", debug=False)
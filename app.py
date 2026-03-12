import io
import csv
import fitz
import docx
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 500    # words per chunk
CHUNK_OVERLAP = 50     # words overlap between chunks (preserves context at boundaries)
TOP_K_CHUNKS  = 5      # how many most-relevant chunks to send to OpenAI

SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant. "
    "Answer using ONLY the content from the document excerpts provided. "
    "If the answer is not in the excerpts, say: "
    "'The document does not contain information about this.' "
    "Be concise and accurate. No external knowledge."
)

# ── In-memory document store ──────────────────────────────────────────────────

# { doc_id: { filename, chunks: [str], embeddings: np.array } }
documents = {}

# ── Text extraction ───────────────────────────────────────────────────────────

BINARY_EXTRACTORS = {
    "pdf":  lambda b: "\n".join(p.get_text() for p in fitz.open(stream=b, filetype="pdf")),
    "docx": lambda b: "\n".join(p.text for p in docx.Document(io.BytesIO(b)).paragraphs if p.text.strip()),
    "csv":  lambda b: "\n".join(", ".join(row) for row in csv.reader(io.StringIO(b.decode("utf-8", errors="replace")))),
}

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    extractor = BINARY_EXTRACTORS.get(ext, lambda b: b.decode("utf-8", errors="replace"))
    return extractor(file_bytes)

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    step   = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = [
        " ".join(words[i : i + CHUNK_SIZE])
        for i in range(0, len(words), step)
        if words[i : i + CHUNK_SIZE]
    ]
    return chunks

# ── Embeddings + retrieval ────────────────────────────────────────────────────

def get_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of texts using OpenAI and return as numpy array."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and a matrix of vectors."""
    return (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a))

def retrieve_top_chunks(question_embedding: np.ndarray, doc_embeddings: np.ndarray, chunks: list[str]) -> list[str]:
    """Return the top-K most relevant chunks for the question."""
    scores     = cosine_similarity(question_embedding, doc_embeddings)
    top_idx    = np.argsort(scores)[::-1][:TOP_K_CHUNKS]
    return [chunks[i] for i in sorted(top_idx)]  # sorted to preserve document order

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file    = request.files.get("file")
    api_key = request.form.get("api_key", "").strip()

    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400
    if not api_key:
        return jsonify({"error": "API key required to embed document"}), 400

    # Extract text
    try:
        text = extract_text(file.filename, file.read())
    except Exception as e:
        return jsonify({"error": f"Text extraction failed: {e}"}), 500

    if not text.strip():
        return jsonify({"error": "No readable text found in file"}), 400

    # Chunk the document
    chunks = chunk_text(text)

    # Embed all chunks upfront (so query time is fast)
    try:
        client     = OpenAI(api_key=api_key)
        embeddings = get_embeddings(client, chunks)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {e}"}), 500

    documents[file.filename] = {
        "filename":   file.filename,
        "chunks":     chunks,
        "embeddings": embeddings,
        "char_count": len(text)
    }

    return jsonify({
        "doc_id":      file.filename,
        "filename":    file.filename,
        "char_count":  len(text),
        "chunk_count": len(chunks),
        "preview":     text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
    })


@app.route("/query", methods=["POST"])
def query():
    body     = request.json or {}
    api_key  = body.get("api_key", "").strip()
    doc_id   = body.get("doc_id", "")
    question = body.get("question", "").strip()
    model    = body.get("model", "gpt-4o-mini")

    # ── Validation ────────────────────────────────────────────────────────────
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400
    if doc_id not in documents:
        return jsonify({"error": "Document not found. Please upload a file first."}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    doc       = documents[doc_id]
    client    = OpenAI(api_key=api_key)

    # ── Embed the question and retrieve relevant chunks ───────────────────────
    try:
        question_embedding = get_embeddings(client, [question])[0]
        top_chunks         = retrieve_top_chunks(question_embedding, doc["embeddings"], doc["chunks"])
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {e}"}), 500

    context = "\n\n---\n\n".join(top_chunks)

    # ── Call OpenAI with only the relevant chunks ─────────────────────────────
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {question}"}
            ]
        )
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {e}"}), 500

    answer  = response.choices[0].message.content.strip()
    verdict = "incorrect" if "does not contain information" in answer.lower() else "correct"

    return jsonify({
        "question":      question,
        "answer":        answer,
        "verdict":       verdict,
        "model":         model,
        "tokens_used":   response.usage.total_tokens,
        "chunks_used":   len(top_chunks),
        "doc_filename":  doc["filename"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
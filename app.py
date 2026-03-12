import io
import csv
import fitz
import docx
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant. "
    "Answer using ONLY the content from the document provided. "
    "If the answer is not in the document, say: "
    "'The document does not contain information about this.' "
    "Be concise and accurate. No external knowledge."
)

# ── In-memory document store ──────────────────────────────────────────────────

documents = {}  # { doc_id: { filename, text } }

# ── Text extraction ───────────────────────────────────────────────────────────

# Special extractors for binary formats. Everything else is treated as plain text.
BINARY_EXTRACTORS = {
    "pdf":  lambda b: "\n".join(p.get_text() for p in fitz.open(stream=b, filetype="pdf")),
    "docx": lambda b: "\n".join(p.text for p in docx.Document(io.BytesIO(b)).paragraphs if p.text.strip()),
    "csv":  lambda b: "\n".join(", ".join(row) for row in csv.reader(io.StringIO(b.decode("utf-8", errors="replace")))),
}

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    extractor = BINARY_EXTRACTORS.get(ext, lambda b: b.decode("utf-8", errors="replace"))
    return extractor(file_bytes)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    try:
        text = extract_text(file.filename, file.read())
    except Exception as e:
        return jsonify({"error": f"Text extraction failed: {e}"}), 500

    if not text.strip():
        return jsonify({"error": "No readable text found in file"}), 400

    documents[file.filename] = {"filename": file.filename, "text": text}

    return jsonify({
        "doc_id":     file.filename,
        "filename":   file.filename,
        "char_count": len(text),
        "preview":    text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
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

    # ── Call OpenAI with full document text ───────────────────────────────────
    try:
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"DOCUMENT:\n{documents[doc_id]['text']}\n\nQUESTION: {question}"}
            ]
        )
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {e}"}), 500

    answer  = response.choices[0].message.content.strip()
    verdict = "incorrect" if "does not contain information" in answer.lower() else "correct"

    return jsonify({
        "question":     question,
        "answer":       answer,
        "verdict":      verdict,
        "model":        model,
        "tokens_used":  response.usage.total_tokens,
        "doc_filename": documents[doc_id]["filename"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
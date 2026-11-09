
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
import pandas as pd

POLICY_DIR = Path("data/policies")

def load_policy_documents():
    """Load all five policy text files."""
    documents = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")

    expected_files = {
        "refund_policy.txt",
        "delivery_policy.txt",
        "escalation_policy.txt",
        "payment_policy.txt",
        "privacy_policy.txt",
    }

    missing = expected_files - set(documents)
    if missing:
        raise FileNotFoundError(
            f"Missing policy documents: {sorted(missing)}"
        )

    return documents


CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split policy documents into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    step = chunk_size - overlap
    chunks = []

    for source, text in documents.items():
        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": source,
                "text": " ".join(chunk_words),
            })

            # Stop once this chunk reaches the end of the document.
            if start + chunk_size >= len(words):
                break

    return chunks


def build_retriever(chunks):
    """Fit TF-IDF on policy chunks and return a retrieval function."""
    if not chunks:
        raise ValueError("Cannot build a retriever without chunks")

    vectorizer = TfidfVectorizer()
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_vectors = vectorizer.fit_transform(chunk_texts)

    def retrieve(query, k=3):
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            chunk_vectors,
        ).flatten()

        # Highest similarity first; stable ordering breaks ties.
        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-scores[index], index),
        )

        return [
            {
                "source": chunks[index]["source"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
            }
            for index in ranked_indices[:min(k, len(chunks))]
        ]

    return retrieve


POLICY_SYSTEM_PROMPT = """
You are a POLICY ASSISTANT for customer support.
Answer questions only using the supplied policy context.
Do not invent or assume policy details that are not in the context.
If the context does not contain the answer, say:
I could not find that information in the policy documents.
"""


def answer(question, retrieve):
    """Answer a question using retrieved policy context."""
    results = retrieve(question, k=3)

    context_lines = []
    for item in results:
        context_lines.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n".join(context_lines)

    user_content = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": POLICY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    response = mock_llm.chat(messages)
    response_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return {
        "answer": response_text,
        "sources": sources,
    }



def inspect_policy_evaluation(retrieve, label):
    """Evaluate retrieval grounding and fallback behavior."""
    evaluation_file = Path("data/raw/policy_eval.csv")
    evaluation = pd.read_csv(evaluation_file, keep_default_na=False)

    answerable = 0
    source_hits = 0
    fact_hits = 0
    unanswerable = 0
    fallback_hits = 0
    failures = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        retrieved = retrieve(question, k=3)
        sources = [item["source"] for item in retrieved]
        result = answer(question, retrieve)
        answer_text = result["answer"]

        if expected_doc == "NONE":
            unanswerable += 1
            if answer_text.startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append(question)
        else:
            answerable += 1

            source_ok = expected_doc in sources
            fact_ok = expected_fact in answer_text

            source_hits += int(source_ok)
            fact_hits += int(fact_ok)

            if not source_ok or not fact_ok:
                failures.append(question)

    print(f"\n--- {label} ---")
    print(f"Answerable questions: {answerable}")
    print(f"Expected document in top 3: {source_hits}/{answerable}")
    print(f"Expected fact present: {fact_hits}/{answerable}")
    print(f"Unanswerable fallback: {fallback_hits}/{unanswerable}")

    print("Questions that failed a check:")
    for question in failures:
        print("-", question)

    return {
        "label": label,
        "source_hits": source_hits,
        "answerable": answerable,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "unanswerable": unanswerable,
        "failures": failures,
    }



if __name__ == "__main__":
    documents = load_policy_documents()

    overlap_chunks = chunk_documents(
        documents,
        chunk_size=40,
        overlap=10
    )
    overlap_retrieve = build_retriever(overlap_chunks)

    inspect_policy_evaluation(
        overlap_retrieve,
        "40-word chunks, overlap 10"
    )

    zero_overlap_chunks = chunk_documents(
        documents,
        chunk_size=40,
        overlap=0
    )
    zero_overlap_retrieve = build_retriever(zero_overlap_chunks)

    inspect_policy_evaluation(
        zero_overlap_retrieve,
        "40-word chunks, overlap 0"
    )

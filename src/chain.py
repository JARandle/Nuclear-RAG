import anthropic
from src.retriever import retrieve_chunks
from dotenv import load_dotenv

load_dotenv()

def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block for the LLM prompt, including source citations.
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)

def build_prompt(question: str, context: str) -> str:
    """
    Builds the full prompt sent to LLM.
    Instructs the model to stay grounded in the provided context.
    """
    return f"""You are a nuclear regulatory compliance assistant.
Answer questions based strictly on the provided document excerpts.

Rules:
- Only use information from the provided context
- Always cite your sources using the format [Source X, Page Y]
- If the context does not contain enough information to answer, 
  say so clearly rather than guessing
- Be precise and technical in your responses

Context:
{context}

Question: {question}

Answer:"""

def ask(question: str, n_results: int = 5) -> dict:
    """
    Main function. Takes a question, retrieves relevant chunks,
    and returns a grounded answer with citations.
    """
    chunks = retrieve_chunks(question, n_results=n_results)
    context = format_context(chunks)
    prompt = build_prompt(question, context)

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = message.content[0].text

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["metadata"]["source"],
                "page": c["metadata"]["page"]
            }
            for c in chunks
        ]
    }


if __name__ == "__main__":
    question = "What are the requirements for HEPA filter testing?"
    print(f"Question: {question}\n")

    result = ask(question)

    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']}, Page {s['page']}")
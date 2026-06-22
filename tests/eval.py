import os
import json
import anthropic
from src.retriever import retrieve_chunks
from src.chain import ask
from dotenv import load_dotenv

load_dotenv()

# Your test set - add more questions here later
TEST_QUESTIONS = [
    {
        "question": "What are the in-place leak testing acceptance criteria for HEPA filter systems?",
        "ground_truth": "Combined penetration and leakage must be less than 0.05% of the challenge aerosol at system-rated flow plus or minus 10%, warranting 99% removal efficiency for particulates."
    },
    {
        "question": "What challenge aerosol is used for HEPA filter penetration testing?",
        "ground_truth": "Dioctyl phthalate (DOP) or 4 centistoke polyalpha olefin (PAO) are used as challenge aerosols for HEPA filter penetration testing."
    },
    {
        "question": "What is the maximum flow rate for a single HEPA filtration unit?",
        "ground_truth": "A single filtration unit is limited to 850 cubic meters per minute or 30,000 CFM to ensure reliable in-place testing."
    },
    {
        "question": "Under what conditions must in-place leak testing be performed?",
        "ground_truth": "In-place leak testing must be performed at frequency intervals per ASME N511, after partial or complete filter bank replacement, following water intrusion, and following painting, fire, or chemical release in any communicating ventilation zone."
    },
    {
        "question": "What quality assurance standard applies to ESF atmosphere cleanup systems?",
        "ground_truth": "All testing and documentation must comply with a quality assurance program consistent with 10 CFR Part 50 Appendix B."
    },
]


def judge_answer(question: str, answer: str, ground_truth: str, context: str) -> dict:
    """
    Uses Claude as a judge to score a single answer on two dimensions:
    - Faithfulness: does the answer stick to what the documents say?
    - Relevancy: does the answer actually address the question?
    Each scored 1-5, then normalized to 0-1.
    """
    client = anthropic.Anthropic()

    prompt = f"""You are an expert evaluator for a nuclear regulatory compliance AI system.
Score the following answer on two dimensions.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context}

System Answer: {answer}

Score the answer on these two dimensions from 1 to 5:

FAITHFULNESS (1-5): Does the answer only use information from the retrieved context?
- 5: Every claim is directly supported by the context
- 4: Nearly all claims supported, minor extrapolation
- 3: Mostly supported but some unsupported claims
- 2: Several claims not found in context
- 1: Answer largely ignores or contradicts the context

RELEVANCY (1-5): Does the answer actually address what was asked?
- 5: Directly and completely answers the question
- 4: Answers the question with minor gaps
- 3: Partially answers the question
- 2: Tangentially related but misses the core question
- 1: Does not address the question

Respond in this exact JSON format with no other text:
{{
  "faithfulness": <score>,
  "relevancy": <score>,
  "faithfulness_reason": "<one sentence explanation>",
  "relevancy_reason": "<one sentence explanation>"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude adds extra text around the JSON
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        scores = json.loads(match.group()) if match else {
            "faithfulness": 0,
            "relevancy": 0,
            "faithfulness_reason": "Parse error",
            "relevancy_reason": "Parse error"
        }

    # Normalize 1-5 scores to 0-1
    scores["faithfulness_normalized"] = round((scores["faithfulness"] - 1) / 4, 4)
    scores["relevancy_normalized"] = round((scores["relevancy"] - 1) / 4, 4)

    return scores


def run_evals():
    """
    Runs every test question through the RAG pipeline,
    judges each answer with Claude, and saves results to JSON.
    """
    client_results = []
    total_faithfulness = 0
    total_relevancy = 0

    print("Running evaluation...\n")

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"Question {i+1}/{len(TEST_QUESTIONS)}: {item['question'][:60]}...")

        result = ask(item["question"])
        chunks = retrieve_chunks(item["question"])
        context = "\n\n".join([c["text"] for c in chunks])

        scores = judge_answer(
            question=item["question"],
            answer=result["answer"],
            ground_truth=item["ground_truth"],
            context=context
        )

        total_faithfulness += scores["faithfulness_normalized"]
        total_relevancy += scores["relevancy_normalized"]

        client_results.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": result["answer"],
            "scores": scores,
            "sources": result["sources"]
        })

        print(f"  Faithfulness: {scores['faithfulness']}/5 - {scores['faithfulness_reason']}")
        print(f"  Relevancy:    {scores['relevancy']}/5 - {scores['relevancy_reason']}\n")

    n = len(TEST_QUESTIONS)
    summary = {
        "avg_faithfulness": round(total_faithfulness / n, 4),
        "avg_relevancy": round(total_relevancy / n, 4),
        "num_questions": n,
        "individual_results": client_results
    }

    os.makedirs("tests", exist_ok=True)
    with open("tests/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary:")
    print(f"  Average Faithfulness: {summary['avg_faithfulness']}")
    print(f"  Average Relevancy:    {summary['avg_relevancy']}")
    print(f"\nFull results saved to tests/eval_results.json")

    return summary


if __name__ == "__main__":
    run_evals()
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = ROOT / "data" / "company_documents.jsonl"
QA_FILE = ROOT / "data" / "company_qa_dataset.csv"
TEST_FILE = ROOT / "data" / "rag_test_dataset.jsonl"

QA_DATA = [
    {
        "question_id": "Q001",
        "question": "What does Applied Materials do?",
        "answer": "Applied Materials provides materials engineering solutions, equipment, services, and software for semiconductor chip and advanced display manufacturing.",
        "category": "Company Overview",
        "source_url": "https://www.appliedmaterials.com/in/en/about.html",
        "expected_keywords": "materials engineering;semiconductor",
    },
    {
        "question_id": "Q002",
        "question": "When was Applied Materials India founded?",
        "answer": "The initial seed dataset lists 2002 as its founding year.",
        "category": "India Operations",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "2002",
    },
    {
        "question_id": "Q003",
        "question": "Where is Applied Materials India headquartered?",
        "answer": "Bengaluru, India.",
        "category": "India Operations",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "Bengaluru",
    },
    {
        "question_id": "Q004",
        "question": "Which Indian cities have an Applied Materials presence?",
        "answer": "The initial seed dataset lists Bengaluru, Chennai, Mumbai, and Pune.",
        "category": "India Operations",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "Bengaluru;Chennai;Mumbai;Pune",
    },
    {
        "question_id": "Q005",
        "question": "What engineering activities does Applied Materials India support?",
        "answer": "The seed dataset describes research and development, product engineering, software engineering, design, validation, and product lifecycle development.",
        "category": "Engineering",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "research and development;software engineering;validation",
    },
    {
        "question_id": "Q006",
        "question": "What is the India Validation Center?",
        "answer": "The seed dataset describes it as a semiconductor research and development facility in India.",
        "category": "Research and Development",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "semiconductor;research and development",
    },
    {
        "question_id": "Q007",
        "question": "Does Applied Materials manufacture semiconductor chips itself?",
        "answer": "Applied Materials primarily supplies equipment, services, and technologies used by semiconductor manufacturers to produce chips.",
        "category": "Technology",
        "source_url": "https://www.appliedmaterials.com/in/en/frequently-asked-questions.html",
        "expected_keywords": "equipment;semiconductor manufacturers",
    },
    {
        "question_id": "Q008",
        "question": "Where is Applied Materials headquartered globally?",
        "answer": "Santa Clara, California, USA.",
        "category": "Company Overview",
        "source_url": "https://www.appliedmaterials.com/in/en/frequently-asked-questions.html",
        "expected_keywords": "Santa Clara;California",
    },
    {
        "question_id": "Q009",
        "question": "What kinds of careers are available at Applied Materials?",
        "answer": "The seed dataset lists application engineering, process engineering, supply chain, product development, field service engineering, and software development.",
        "category": "Careers",
        "source_url": "https://www.appliedmaterials.com/in/en/careers.html",
        "expected_keywords": "engineering;software development",
    },
    {
        "question_id": "Q010",
        "question": "How can candidates apply for jobs at Applied Materials?",
        "answer": "Candidates can look for suitable openings and apply through the company's official careers website.",
        "category": "Careers",
        "source_url": "https://www.appliedmaterials.com/in/en/careers.html",
        "expected_keywords": "careers website;apply",
    },
    {
        "question_id": "Q011",
        "question": "What does Applied Materials India contribute to global product development?",
        "answer": "The seed dataset describes contributions to research and development, engineering, software, design, validation, and product lifecycle activities.",
        "category": "Engineering",
        "source_url": "https://www.appliedmaterials.com/in/en/about/india-overview.html",
        "expected_keywords": "engineering;design;validation",
    },
    {
        "question_id": "Q012",
        "question": "What industries does Applied Materials support?",
        "answer": "The seed dataset identifies semiconductor chip manufacturing and advanced display manufacturing.",
        "category": "Company Overview",
        "source_url": "https://www.appliedmaterials.com/in/en/about.html",
        "expected_keywords": "semiconductor;advanced display",
    },
]


def main():
    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Company documents not found: {DOCUMENTS_FILE}"
        )

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        documents = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    with QA_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question",
                "answer",
                "category",
                "source_url",
                "expected_keywords",
            ],
        )
        writer.writeheader()
        writer.writerows(QA_DATA)

    with TEST_FILE.open("w", encoding="utf-8") as file:
        for item in QA_DATA:
            record = {
                "test_id": item["question_id"],
                "question": item["question"],
                "expected_answer": item["answer"],
                "expected_source": item["source_url"],
                "expected_keywords": item["expected_keywords"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Company documents loaded: {len(documents)}")
    print(f"Q&A records created: {len(QA_DATA)}")
    print(f"Q&A dataset saved: {QA_FILE}")
    print(f"RAG test dataset saved: {TEST_FILE}")


if __name__ == "__main__":
    main()
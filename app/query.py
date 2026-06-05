import re
from typing import Optional

# Pattern -> search-optimized rewrite for vague/broad questions
BROAD_PATTERNS: list[tuple[str, str]] = [
    (
        r"(tell me about|about him|about kshitij|about yourself|"
        r"who is|introduce|introduction|describe him|describe kshitij)",
        "Kshitij Gupta career background experience skills education "
        "summary contact projects achievements overview profile",
    ),
    (
        r"what (can|does) he do",
        "Kshitij Gupta skills capabilities experience projects "
        "technical expertise responsibilities engineering work",
    ),
    (
        r"(tech stack|technologies|tools|frameworks|what does he use|"
        r"programming languages|what languages|what tech|"
        r"what is he good at|expertise|specialties|specialities)",
        "Kshitij Gupta technical skills programming languages "
        "AI ML frameworks cloud infrastructure tools stack complete list",
    ),
    (
        r"(experience|work history|career|jobs|positions|roles|"
        r"where did he work|companies|previous work|work experience)",
        "Kshitij Gupta work experience career history jobs roles "
        "CARS24 Typeface ttdsoft MirrorSize positions timeline",
    ),
    (
        r"(projects|what has he built|portfolio|open.?source|"
        r"what did he make|side projects|github|repos)",
        "Kshitij Gupta projects portfolio open source "
        "LLM Eval Harness Image Similarity Search NeuralDocs AI ONNXLab github",
    ),
    (
        r"(skills|what does he know|qualifications|expertise|proficien(t|cies))",
        "Kshitij Gupta skills programming languages AI ML "
        "cloud infrastructure technologies frameworks capabilities",
    ),
    (
        r"(education|qualification|degrees|college|university|"
        r"academic|studied|where did he study)",
        "Kshitij Gupta education IIIT Hyderabad IIIT Delhi "
        "M.Tech B.Tech coursework academic background",
    ),
    (
        r"(contact|hire|reach|email|phone|linkedin|"
        r"github|how to reach|get in touch|"
        r"is he available|looking for|open to)",
        "Kshitij Gupta contact email phone linkedin github "
        "availability hiring remote work information",
    ),
    (
        r"(background|overview|summary|profile|about)",
        "Kshitij Gupta background summary career profile "
        "experience skills education overview",
    ),
    (
        r"(achievements|awards|honours|recognition|accomplishments|"
        r"dean.?list|kvpy|scholar)",
        "Kshitij Gupta achievements awards honours recognition "
        "Dean List KVPY Scholar Google Lunar X-Prize JSTSE",
    ),
]


def is_broad_query(question: str) -> bool:
    q = question.lower().strip()
    if not q:
        return False
    words = q.split()
    # Very short queries (1-3 words) are almost always broad/vague
    if len(words) <= 3:
        return True
    for pattern, _ in BROAD_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def rewrite_query(question: str) -> str:
    q = question.lower().strip()
    for pattern, replacement in BROAD_PATTERNS:
        if re.search(pattern, q):
            return replacement
    return question

import re


def clean_corpus(file_path: str):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    # Normalize whitespace and strip very short lines
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    lines = [line for line in lines if len(line) > 1]
    return lines


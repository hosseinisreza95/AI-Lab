import re
import json
from pathlib import Path

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    # Naive baseline splitter, used as a fallback inside structure_aware_chunks
    # for sections that are still too large after splitting by heading.
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def structure_aware_chunks(markdown_text: str, max_tokens: int = 500) -> list[dict]:
    # Splits by Markdown heading first, and keeps tables intact as their
    # own chunks so a financial table is never cut in half.
    lines = markdown_text.split("\n")
    chunks = []
    buffer = []
    current_heading = "Intro"
    in_table = False
    table_buffer = []

    def flush_buffer():
        text = "\n".join(buffer).strip()
        if text:
            if count_tokens(text) <= max_tokens:
                chunks.append({"heading": current_heading, "type": "text", "text": text})
            else:
                for sc in fixed_size_chunks(text, chunk_size=max_tokens, overlap=50):
                    chunks.append({"heading": current_heading, "type": "text", "text": sc})
        buffer.clear()

    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            flush_buffer()
            current_heading = line.strip("# ").strip()
            continue

        if line.strip().startswith("|"):
            if not in_table:
                flush_buffer()
                in_table = True
            table_buffer.append(line)
            continue
        else:
            if in_table:
                chunks.append({"heading": current_heading, "type": "table", "text": "\n".join(table_buffer)})
                table_buffer = []
                in_table = False

        buffer.append(line)

    flush_buffer()
    if table_buffer:
        chunks.append({"heading": current_heading, "type": "table", "text": "\n".join(table_buffer)})

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    text = (project_root / "parsed" / "nvda_full.md").read_text(encoding="utf-8")

    print(f"Full document: {count_tokens(text)} tokens")

    chunks = structure_aware_chunks(text)
    n_tables = sum(1 for c in chunks if c["type"] == "table")
    print(f"Total chunks: {len(chunks)} ({n_tables} of them tables)")

    out_path = project_root / "chunks" / "structure_aware.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")
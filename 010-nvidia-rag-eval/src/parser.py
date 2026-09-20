from pathlib import Path
from docling.document_converter import DocumentConverter

def parse_document(file_path: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def save_markdown(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "dataset" / "nvda-20260125.pdf"
    output_path = project_root / "parsed" / "nvda_full.md"

    text = parse_document(str(pdf_path))
    save_markdown(text, output_path)
    print(f"Saved {len(text)} characters to {output_path}")
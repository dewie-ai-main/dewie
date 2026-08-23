from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
DOCS_DIR = FIXTURES_DIR / "docs"


def load_fixture_doc(name: str) -> str:
    return (DOCS_DIR / name).read_text(encoding="utf-8")

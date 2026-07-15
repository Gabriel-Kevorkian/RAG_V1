"""
document_processor.py  —  Day 1 of the Hybrid RAG project: INGESTION
====================================================================

Purpose
-------
Read every source file in ``corpus/``, strip away the HTML scaffolding
(navigation bars, scripts, footers), and return a clean list of documents.
Each document carries its cleaned text plus metadata about where it came from.

There is no AI or search here yet. This is pure "get clean text out of
messy files" work — and every later stage of the project inherits the
quality of what this file produces. Garbage in, garbage out.
"""

from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

# corpus/ sits at the project root, one level up from this ingestion/ folder.
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def clean_html(html: str) -> tuple[str, str]:
    """Extract the real documentation text from a JavaDoc HTML page.

    JavaDoc wraps the actual class documentation in a single ``<main>`` tag.
    Everything outside it (the ``<head>``, nav bars, scripts, footer) is
    website scaffolding we do NOT want embedded into our search index, so we
    keep only what is inside ``<main>``.

    Returns
    -------
    (title, clean_text) : tuple[str, str]
        ``title`` comes from the page's <title> tag (e.g. "HashMap (Java SE 21 ...)").
        ``clean_text`` is the de-cluttered documentation body.
    """
    soup = BeautifulSoup(html, "lxml")

    # The <title> tag gives a human-readable name for metadata.
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Keep ONLY the <main> content — that is the class documentation itself.
    # Fall back to the whole document if a page somehow has no <main>.
    container = soup.find("main") or soup

    # Remove any leftover non-content tags still sitting inside <main>.
    for tag in container(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # separator="\n" stops words from different tags running together.
    text = container.get_text(separator="\n")

    # Tidy up: trim each line and drop the blank ones left behind.
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    return title, clean_text


def load_documents(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Load and clean every ``.html`` and ``.md`` file in the corpus.

    Returns a list of LangChain ``Document`` objects. Each one has:
      * ``page_content`` — the cleaned text
      * ``metadata``     — {"source": filename, "title": title}

    We use LangChain's ``Document`` type on purpose: Day 2's text splitter
    can consume these objects directly, so we don't have to convert later.
    """
    documents: list[Document] = []

    # sorted() keeps the load order stable and predictable between runs.
    files = sorted(corpus_dir.glob("*.html")) + sorted(corpus_dir.glob("*.md"))

    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")

        if path.suffix == ".html":
            title, text = clean_html(raw)
        else:  # markdown is already fairly clean — keep it as-is
            title, text = path.stem, raw.strip()

        if not text:
            print(f"  [warning] no text extracted from {path.name}, skipping")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "title": title},
            )
        )

    return documents


if __name__ == "__main__":
    # Running this file directly loads the corpus and prints a sanity check.
    docs = load_documents()

    print(f"Loaded {len(docs)} documents from {CORPUS_DIR}\n")

    if docs:
        first = docs[0]
        print("=" * 60)
        print(f"First document: {first.metadata['source']}")
        print(f"Title:          {first.metadata['title']}")
        print(f"Characters:     {len(first.page_content):,}")
        print("=" * 60)
        print("Preview (first 600 characters):\n")
        print(first.page_content[:600])

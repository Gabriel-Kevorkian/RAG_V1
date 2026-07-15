"""
chunker_v2.py  —  Day 11: chunk on the document's structure, not on a token count
=================================================================================

WHY THIS EXISTS
---------------
The Day 10 failure analysis left two real bugs, and both trace to the same place:

  p07  the sentence "...the add call will throw a ClassCastException" was CUT IN
       HALF by a chunk boundary. The half that answers the question landed at the
       head of a chunk that is otherwise about Comparator constructors, so it
       barely embeds as being about that idea at all. The model never saw it and
       correctly refused.
  p06  java.util.Arrays exploded into 162 chunks, and the definitive line
       ("Returns a fixed-size list backed by the specified array") ended up in a
       different chunk from the asList prose the model actually read.

The mechanism, measured: **48.7% of v1 chunks (370/759) begin mid-sentence.**

That is not bad luck, it is a guaranteed consequence of two lines of code.
`document_processor.clean_html` calls `get_text(separator="\\n")`, which emits a
newline at EVERY tag boundary -- including inline `<code>` and `<a>` tags, which
JavaDoc uses inside almost every sentence. It then drops blank lines. So the
"paragraphs" the splitter is looking for do not exist: a typical page has ZERO
blank lines, 456 newlines, and 18 sentence endings.

`chunker.chunk_documents` then asks RecursiveCharacterTextSplitter to split on
["\\n\\n", "\\n", ". ", ...] in that order. "\\n\\n" never matches. So it falls
through to "\\n" -- which, in this text, sits in the MIDDLE of sentences. The
splitter advertises itself as sentence-aware (and its docstring promises "never
split mid-sentence"); on this corpus it is splitting on tag boundaries.

THE FIX
-------
Stop flattening the structure and then trying to guess it back. JavaDoc already
marks up exactly the unit we want:

    <section class="class-description">   the class overview
    <section class="summary">             the method/constructor summary tables
    <section class="detail">              ONE PER MEMBER (33 of them in TreeSet)

So we chunk on the member. Each chunk is one constructor or one method, complete,
with its signature, its description, its Parameters/Returns/Throws -- the exact
unit a JavaDoc lookup question is asking about. Each chunk is prefixed with a
header naming its class and member, so it is self-contained: a chunk about
`poll()` says "java.util.ArrayDeque - Method: poll" at the top, which helps both
the embedder and BM25, and means a retrieved chunk is readable on its own.

Two smaller decisions that matter:

  * Text is extracted with `get_text(" ")`, not `"\\n"`. Inline tags become
    SPACES, so sentences survive intact and the splitter's sentence separators
    actually work on the rare section too big to fit in one chunk.

  * v1 dropped every piece under MIN_CHUNK_TOKENS=100 as "noise". At the member
    level that is actively harmful: `poll()`'s entire JavaDoc is short, and
    dropping it would delete the answer to s14 from the index. Short members are
    KEPT -- the header prefix is what makes them meaningful, not their length.
"""

from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

# A member section is one chunk. This cap only bites on the few oversized ones
# (a big summary table, a class overview with a long preamble).
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50
# Only drop genuinely empty fragments. See the note above about MIN_CHUNK_TOKENS.
MIN_CHUNK_TOKENS = 10

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _text(node) -> str:
    """Flatten a node to text WITHOUT shredding sentences.

    separator=" " is the whole point: JavaDoc wraps `ClassCastException` in an
    <a> and `add` in a <code>, mid-sentence. With separator="\\n" (what v1 did)
    every one of those becomes a line break and the sentence stops existing.
    """
    return " ".join(node.get_text(" ", strip=True).split())


# The splitter is now a FALLBACK, used only when a single member is too big.
# Note "\n\n" is gone from the separators: we no longer pretend it is there.
_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[". ", "? ", "! ", "; ", ", ", " ", ""],
)


def _emit(chunks: list[Document], header: str, body: str, meta: dict) -> None:
    """Add `body` as one chunk, or as several if it is oversized. Always headed."""
    if not body.strip():
        return

    # The header is repeated on every piece of a split section, so a chunk is
    # never an anonymous slab of text with no idea which class it belongs to.
    budget = MAX_CHUNK_TOKENS - count_tokens(header) - 2
    pieces = [body] if count_tokens(body) <= budget else _splitter.split_text(body)

    for piece in pieces:
        text = f"{header}\n\n{piece}"
        if count_tokens(text) < MIN_CHUNK_TOKENS:
            continue
        i = len(chunks)
        chunks.append(Document(
            page_content=text,
            metadata={**meta, "chunk_id": f"{meta['source']}::{i}"},
        ))


def chunk_html(path: Path) -> list[Document]:
    """Chunk one JavaDoc page along its own section structure."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    main = soup.find("main") or soup
    for tag in main(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # "java.util.TreeSet.html" -> "java.util.TreeSet"
    cls = path.stem
    base = {"source": path.name, "title": title}
    chunks: list[Document] = []

    # 1. The class overview: what this class IS. Answers "what is a TreeSet".
    desc = main.find("section", class_="class-description")
    if desc:
        _emit(chunks, f"{cls} - Class description", _text(desc),
              {**base, "kind": "class-description", "member": ""})

    # 2. The summary tables: one-line descriptions of every member. These are
    #    terse and high-signal -- p06's gold sentence ("Returns a fixed-size list
    #    backed by the specified array") lives here, not in the method detail.
    for sec in main.find_all("section", class_=["constructor-summary", "method-summary"]):
        kind = (sec.get("class") or ["summary"])[0]
        _emit(chunks, f"{cls} - {kind.replace('-', ' ').title()}", _text(sec),
              {**base, "kind": kind, "member": ""})

    # 3. THE MAIN EVENT: one chunk per constructor / method. This is the unit an
    #    exact-lookup question is actually about.
    for sec in main.find_all("section", class_="detail"):
        heading = sec.find(["h2", "h3", "h4"])
        name = heading.get_text(strip=True) if heading else (sec.get("id") or "")
        sig = sec.find("div", class_="member-signature")
        signature = _text(sig) if sig else ""

        header = f"{cls} - Member: {name}"
        if signature:
            header += f"\n{signature}"
        _emit(chunks, header, _text(sec),
              {**base, "kind": "member", "member": name})

    # Fallback: a page with no JavaDoc structure at all (hand-written .md, or an
    # unusual page). Better to index it plainly than to index nothing.
    if not chunks:
        _emit(chunks, cls, _text(main), {**base, "kind": "page", "member": ""})

    return chunks


def chunk_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    chunks: list[Document] = []
    for path in sorted(corpus_dir.glob("*.html")):
        chunks.extend(chunk_html(path))
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        _emit(chunks, path.stem, text,
              {"source": path.name, "title": path.stem, "kind": "page", "member": ""})
    return chunks


if __name__ == "__main__":
    from collections import Counter

    chunks = chunk_corpus()
    tok = [count_tokens(c.page_content) for c in chunks]

    midsentence = sum(
        1 for c in chunks
        # ignore the header line; ask whether the BODY starts mid-sentence
        if (b := c.page_content.split("\n\n", 1)[-1].lstrip()) and b[0].islower()
    )
    print(f"{len(chunks)} chunks")
    print(f"  tokens  min {min(tok)}  max {max(tok)}  avg {sum(tok)//len(tok)}")
    print(f"  body starts mid-sentence: {midsentence}/{len(chunks)} "
          f"= {midsentence/len(chunks):.1%}   (v1: 48.7%)")
    print("\nby kind:", dict(Counter(c.metadata["kind"] for c in chunks)))
    per_src = Counter(c.metadata["source"] for c in chunks)
    print("\nmost chunks per document:")
    for s, n in per_src.most_common(3):
        print(f"  {n:4d}  {s}")

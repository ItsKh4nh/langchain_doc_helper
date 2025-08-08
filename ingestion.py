import asyncio
import os
import ssl
from typing import Any, Dict, List
import re
import hashlib

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl

from consts import INDEX_NAME, EMBEDDING_MODEL_NAME
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
tavily_crawl = TavilyCrawl()


def _extract_title_from_soup(soup: BeautifulSoup) -> str:
    # Prefer page <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fallback to first H1
    first_h1 = soup.find("h1")
    if first_h1 and first_h1.get_text(strip=True):
        return first_h1.get_text(strip=True)
    return ""


def _clean_html_to_main_text(html: str) -> str:
    """Extract main article text from HTML, removing nav/sidebars/footers and boilerplate."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts/styles and common non-content containers
    for tag_name in ["script", "style", "noscript", "iframe", "svg", "form", "button"]:
        for t in soup.find_all(tag_name):
            t.decompose()

    # Remove common layout/boilerplate sections
    boilerplate_selectors = [
        "nav",
        "header",
        "footer",
        "aside",
        ".sidebar",
        ".sphinxsidebar",
        ".bd-sidebar",
        ".bd-header",
        ".navbar",
        ".toc",
        ".tocsidebar",
        ".related",
        ".menu",
        "#navbar",
        "#sidebar",
        "#toc",
    ]
    for sel in boilerplate_selectors:
        for t in soup.select(sel):
            t.decompose()

    # Try to locate the primary content region used by Sphinx/PyData/Sphinx Book Theme
    main = (
        soup.find("main")
        or soup.select_one(".bd-content")
        or soup.select_one(".bd-article")
        or soup.select_one("[role='main']")
        or soup.select_one("article")
        or soup.select_one("#main-content")
        or soup.body
        or soup
    )

    # Remove superfluous repeated link lists often present at the bottom/top
    for ul in main.find_all("ul"):
        text = ul.get_text(" ", strip=True)
        # Heuristic: list dominated by short link items (likely nav lists)
        items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
        if items and sum(1 for it in items if len(it) <= 60) / len(items) > 0.7:
            ul.decompose()

    text = main.get_text("\n", strip=True)

    # Normalize whitespace and collapse excessive blank lines
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            # Generate stable IDs per document to avoid duplicates and allow upserts
            generated_ids: List[str] = []
            for doc in batch:
                source = (doc.metadata or {}).get("source", "")
                basis = f"{source}\n{doc.page_content}".encode("utf-8", errors="ignore")
                digest = hashlib.sha1(basis).hexdigest()
                generated_ids.append(digest)

            await vectorstore.aadd_documents(batch, ids=generated_ids)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
            # Fetch full HTML so we can precisely extract the main article content
            "format": "html",
        }
    )

    all_docs = []
    for doc_data in res["results"]:
        if isinstance(doc_data, dict):
            # Prefer raw HTML then sanitize to the main content body
            raw_html = (
                doc_data.get("raw_content", "")
                or doc_data.get("content", "")
                or doc_data.get("body", "")
            )

            cleaned_text = _clean_html_to_main_text(raw_html)
            if not cleaned_text:
                # Fallback to any textual field if HTML extraction failed
                cleaned_text = doc_data.get("text", "") or doc_data.get(
                    "page_content", ""
                )

            # Skip documents with no usable content
            if not cleaned_text or not cleaned_text.strip():
                log_warning(
                    f"⚠️  Skipping document with no content: {doc_data.get('url', 'Unknown URL')}"
                )
                continue

            # Title: use provided or extract from HTML
            title = doc_data.get("title", "")
            if not title and raw_html:
                try:
                    title = _extract_title_from_soup(BeautifulSoup(raw_html, "lxml"))
                except Exception:
                    title = ""

            metadata = {
                "source": doc_data.get("url", ""),
                "title": title,
            }
            for key, value in doc_data.items():
                if key not in [
                    "raw_content",
                    "content",
                    "text",
                    "body",
                    "page_content",
                    "url",
                    "title",
                ]:
                    metadata[key] = value

            all_docs.append(Document(page_content=cleaned_text, metadata=metadata))
        elif isinstance(doc_data, Document):
            all_docs.append(doc_data)
        else:
            log_warning(f"Skipping unexpected document type: {type(doc_data)}")

    log_info(
        f"📊 Processed {len(all_docs)} documents with content out of {len(res['results'])} total results"
    )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

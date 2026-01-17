"""PDF parsing and chunking utilities.

This module handles PDF text extraction, chunking with overlap,
and metadata attachment for policy documents.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFChunk:
    """Represents a chunk of text from a PDF with metadata."""

    def __init__(
        self,
        text: str,
        page_number: int,
        section_id: str,
        policy_name: str,
        chunk_index: int,
    ):
        """
        Initialize a PDF chunk.

        Args:
            text: The chunk text content
            page_number: Page number where chunk originates
            section_id: Identifier for the section (e.g., "Section 2.1")
            policy_name: Name of the policy document
            chunk_index: Sequential index of this chunk in the document
        """
        self.text = text
        self.page_number = page_number
        self.section_id = section_id
        self.policy_name = policy_name
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "section_id": self.section_id,
            "policy_name": self.policy_name,
            "chunk_index": self.chunk_index,
        }

    def __repr__(self) -> str:
        return f"PDFChunk(section={self.section_id}, page={self.page_number}, len={len(self.text)})"


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page information.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with 'text' and 'page_number' keys

    Raises:
        RuntimeError: If no PDF library is available or extraction fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []

    # Try pdfplumber first (better text extraction)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "text": text.strip(),
                            "page_number": page_num,
                        })
            if pages:
                return pages
        except Exception as e:
            print(f"Warning: pdfplumber extraction failed: {e}, trying pypdf...")

    # Fallback to pypdf
    if PYPDF_AVAILABLE:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text.strip(),
                        "page_number": page_num,
                    })
            return pages
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed with pypdf: {e}")

    raise RuntimeError(
        "No PDF library available. Install pdfplumber or pypdf: "
        "pip install pdfplumber pypdf"
    )


def chunk_pdf_text(
    pages: List[Dict[str, Any]],
    policy_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[PDFChunk]:
    """
    Chunk PDF text with overlap and attach metadata.

    Args:
        pages: List of page dicts from extract_text_from_pdf
        policy_name: Name of the policy document
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of PDFChunk objects with metadata
    """
    # Combine all pages into full text with page markers
    full_text_parts = []
    page_boundaries = []  # Track which page each character belongs to

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]
        full_text_parts.append(text)
        # Mark page boundaries
        page_boundaries.extend([page_num] * len(text))
        full_text_parts.append("\n\n")  # Page separator
        page_boundaries.extend([page_num] * 2)

    full_text = "".join(full_text_parts)

    # Use LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_text(full_text)

    # Create PDFChunk objects with metadata
    pdf_chunks = []
    char_index = 0

    for idx, chunk_text in enumerate(chunks):
        # Determine which page this chunk belongs to
        # Use the page of the first character in the chunk
        chunk_start = full_text.find(chunk_text, char_index)
        if chunk_start == -1:
            # Fallback: use middle of chunk
            chunk_start = char_index + len(chunk_text) // 2

        if chunk_start < len(page_boundaries):
            page_number = page_boundaries[chunk_start]
        else:
            # Fallback to last page
            page_number = pages[-1]["page_number"] if pages else 1

        # Generate section_id based on chunk index
        # In a real system, you might parse actual section headers
        section_id = f"Section {idx + 1}"

        chunk = PDFChunk(
            text=chunk_text,
            page_number=page_number,
            section_id=section_id,
            policy_name=policy_name,
            chunk_index=idx,
        )
        pdf_chunks.append(chunk)
        char_index = chunk_start + len(chunk_text)

    return pdf_chunks


def load_policy_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[PDFChunk]:
    """
    Load and chunk a policy PDF file.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of PDFChunk objects

    Example:
        >>> chunks = load_policy_pdf("data/policies/health_policy.pdf")
        >>> print(f"Loaded {len(chunks)} chunks")
    """
    policy_name = Path(pdf_path).stem

    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Chunk text with metadata
    chunks = chunk_pdf_text(pages, policy_name, chunk_size, chunk_overlap)

    return chunks

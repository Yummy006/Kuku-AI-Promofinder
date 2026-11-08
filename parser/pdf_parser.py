"""
pdf_parser.py
=============
Extracts text from PDF screenplay files using PyMuPDF (fitz),
then delegates to txt_parser for structured parsing.

Falls back to a clear error message if PyMuPDF is not installed.
"""

from typing import List, Optional
from parser.txt_parser import Episode, parse_txt_content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given as raw bytes.
    Returns the combined text of all pages.
    Raises ImportError if PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def parse_pdf_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> Optional[Episode]:
    """
    Parse a PDF screenplay from raw bytes.
    Returns an Episode object or None on failure.
    """
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ImportError as e:
        # Return a stub episode with an error so the UI can display it
        ep = Episode(number=0, filename=filename, logline="", continuity="")
        ep.parse_errors.append(str(e))
        return ep
    except Exception as e:
        ep = Episode(number=0, filename=filename, logline="", continuity="")
        ep.parse_errors.append(f"PDF extraction failed: {e}")
        return ep

    return parse_txt_content(text, filename=filename)


def parse_pdf_files(pdf_bytes_list: List[bytes],
                    filenames: List[str]) -> List[Episode]:
    """Parse a list of PDF files supplied as (bytes, filename) pairs."""
    episodes = []
    for pdf_bytes, fname in zip(pdf_bytes_list, filenames):
        ep = parse_pdf_bytes(pdf_bytes, fname)
        if ep is not None:
            episodes.append(ep)
    return episodes

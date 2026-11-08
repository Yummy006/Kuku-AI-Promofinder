"""
docx_parser.py
==============
Extracts text from .docx Word files using python-docx,
then delegates to txt_parser for structured parsing.
"""

from typing import Optional, List
from parser.txt_parser import Episode, parse_txt_content


def extract_text_from_docx(docx_bytes: bytes) -> str:
    """
    Extract all paragraph text from a .docx file given as raw bytes.
    Raises ImportError if python-docx is not installed.
    """
    try:
        from docx import Document
        import io
    except ImportError:
        raise ImportError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    import io
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def parse_docx_bytes(docx_bytes: bytes,
                     filename: str = "upload.docx") -> Optional[Episode]:
    """
    Parse a .docx screenplay from raw bytes.
    Returns an Episode object or None on failure.
    """
    try:
        text = extract_text_from_docx(docx_bytes)
    except ImportError as e:
        ep = Episode(number=0, filename=filename, logline="", continuity="")
        ep.parse_errors.append(str(e))
        return ep
    except Exception as e:
        ep = Episode(number=0, filename=filename, logline="", continuity="")
        ep.parse_errors.append(f"DOCX extraction failed: {e}")
        return ep

    return parse_txt_content(text, filename=filename)


def parse_docx_files(docx_bytes_list: List[bytes],
                     filenames: List[str]) -> List[Episode]:
    """Parse a list of .docx files supplied as (bytes, filename) pairs."""
    episodes = []
    for docx_bytes, fname in zip(docx_bytes_list, filenames):
        ep = parse_docx_bytes(docx_bytes, fname)
        if ep is not None:
            episodes.append(ep)
    return episodes

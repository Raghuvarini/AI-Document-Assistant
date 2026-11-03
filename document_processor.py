
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import re


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str


def extract(name, data):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError(f'{name}: exceeds 20 MB limit.')

    suffix = Path(name).suffix.lower()

    if suffix == '.pdf':
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))

        if reader.is_encrypted:
            raise ValueError(f'{name}: decrypt the PDF before uploading.')

        if len(reader.pages) > 500:
            raise ValueError(f'{name}: maximum 500 pages.')

        pages = [
            (i + 1, page.extract_text() or '')
            for i, page in enumerate(reader.pages)
        ]

    elif suffix == '.docx':
        from docx import Document

        doc = Document(BytesIO(data))

        text = '\n'.join(x.text for x in doc.paragraphs)

        text += '\n' + '\n'.join(
            ' | '.join(c.text for c in row.cells)
            for table in doc.tables
            for row in table.rows
        )

        pages = [(1, text)]

    elif suffix in {'.txt', '.md'}:
        pages = [(1, data.decode('utf-8-sig'))]

    else:
        raise ValueError(f'{name}: unsupported format.')

    if not any(text.strip() for _, text in pages):
        raise ValueError(
            f'{name}: no readable text. Scanned PDFs need OCR first.'
        )

    return pages


def chunk_pages(name, pages, size=700, overlap=120):

    if not 0 <= overlap < size:
        raise ValueError('Overlap must be smaller than chunk size.')

    chunks = []

    for page, raw in pages:

        text = re.sub(r'\s+', ' ', raw).strip()

        for start in range(0, len(text), size - overlap):

            part = text[start:start + size]

            if not part:
                continue

            identity = f'{name}:{page}:{start}:{part}'

            chunks.append(
                Chunk(
                    hashlib.sha256(
                        identity.encode()
                    ).hexdigest()[:20],
                    name,
                    page,
                    part
                )
            )

            if start + size >= len(text):
                break

    return chunks

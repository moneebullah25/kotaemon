from pathlib import Path

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import PDFReader

from kotaemon.indices.ingests import DocumentIngestor, files
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.loaders import AdobeReader, MathpixPDFReader, OCRReader


def test_ingestor_include_src():
    dirpath = Path(__file__).parent
    ingestor = DocumentIngestor(
        pdf_mode="normal",
        text_splitter=TokenSplitter(chunk_size=200, chunk_overlap=10),
    )
    nodes = ingestor(dirpath / "resources" / "table.pdf")
    assert type(nodes) is list
    assert nodes[0].relationships


class _CustomPDFReader(BaseReader):
    def load_data(self, file, extra_info=None):
        return []


def _pdf_extractor(monkeypatch, **kwargs):
    captured = {}
    monkeypatch.setattr(files, "DirectoryReader", lambda **kw: captured.update(kw))
    DocumentIngestor(**kwargs)._get_reader(["a.pdf"])
    return captured["file_extractor"][".pdf"]


def test_override_pdf_extractor_wins_over_pdf_mode(monkeypatch):
    reader = _pdf_extractor(
        monkeypatch,
        pdf_mode="normal",
        override_file_extractors={".pdf": _CustomPDFReader},
    )
    assert isinstance(reader, _CustomPDFReader)


@pytest.mark.parametrize(
    "mode,cls",
    [
        ("normal", PDFReader),
        ("ocr", OCRReader),
        ("multimodal", AdobeReader),
        ("mathpix", MathpixPDFReader),
    ],
)
def test_pdf_mode_selects_reader(monkeypatch, mode, cls):
    assert isinstance(_pdf_extractor(monkeypatch, pdf_mode=mode), cls)


@pytest.mark.parametrize("mode", ["OCR", "mathpix ", ""])
def test_unknown_pdf_mode_raises(mode):
    with pytest.raises(ValueError, match="Unknown pdf_mode"):
        DocumentIngestor(pdf_mode=mode)._get_reader(["a.pdf"])

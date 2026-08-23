"""Tests for dewie.ingestion.extractors — get_extractor, EXTRACTORS, EXTENSION_MAP."""

from __future__ import annotations

import pytest


def test_get_extractor_by_content_type():
    from dewie.ingestion.extractors import extract_pdf, get_extractor

    fn = get_extractor("application/pdf", "https://example.com/doc.pdf")
    assert fn is extract_pdf


def test_get_extractor_by_url_extension():
    from dewie.ingestion.extractors import extract_pdf, get_extractor

    fn = get_extractor("text/html", "https://example.com/report.pdf")
    assert fn is extract_pdf


def test_get_extractor_docx():
    from dewie.ingestion.extractors import extract_docx, get_extractor

    fn = get_extractor(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "https://example.com/file",
    )
    assert fn is extract_docx


def test_get_extractor_xlsx():
    from dewie.ingestion.extractors import extract_xlsx, get_extractor

    fn = get_extractor("text/html", "https://example.com/data.xlsx")
    assert fn is extract_xlsx


def test_get_extractor_pptx():
    from dewie.ingestion.extractors import extract_pptx, get_extractor

    fn = get_extractor("text/html", "https://example.com/slides.pptx")
    assert fn is extract_pptx


def test_get_extractor_unknown_returns_none():
    from dewie.ingestion.extractors import get_extractor

    fn = get_extractor("text/html", "https://example.com/page.html")
    assert fn is None


def test_get_extractor_unknown_ext_returns_none():
    from dewie.ingestion.extractors import get_extractor

    fn = get_extractor("text/plain", "https://example.com/doc.csv")
    assert fn is None


def test_get_extractor_strips_charset():
    from dewie.ingestion.extractors import extract_pdf, get_extractor

    fn = get_extractor("application/pdf; charset=utf-8", "https://example.com/x")
    assert fn is extract_pdf


def test_content_type_case_insensitive():
    from dewie.ingestion.extractors import extract_pdf, get_extractor

    fn = get_extractor("APPLICATION/PDF", "https://example.com/x")
    assert fn is extract_pdf


def test_extension_map_covers_doc():
    from dewie.ingestion.extractors import extract_docx, get_extractor

    fn = get_extractor("text/html", "https://example.com/old.doc")
    assert fn is extract_docx


def test_extractors_dict_has_expected_keys():
    from dewie.ingestion.extractors import EXTRACTORS

    assert "application/pdf" in EXTRACTORS
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in EXTRACTORS
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in EXTRACTORS
    assert "application/vnd.openxmlformats-officedocument.presentationml.presentation" in EXTRACTORS


# ── extract_pdf ─────────────────────────────────────────────────────────────


def test_extract_pdf_with_metadata_title():
    pytest.importorskip("pdfplumber")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is the body text."
    mock_pdf = MagicMock()
    mock_pdf.metadata = {"Title": "My PDF Title"}
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    with patch("pdfplumber.open", return_value=mock_pdf):
        title, body = extract_pdf(b"fake pdf data")
    assert title == "My PDF Title"
    assert "body text" in body


def test_extract_pdf_uses_subject_when_no_title():
    pytest.importorskip("pdfplumber")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Body content here."
    mock_pdf = MagicMock()
    mock_pdf.metadata = {"Subject": "My Subject"}
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    with patch("pdfplumber.open", return_value=mock_pdf):
        title, body = extract_pdf(b"fake")
    assert title == "My Subject"


def test_extract_pdf_uses_first_line_when_no_metadata_title():
    pytest.importorskip("pdfplumber")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "First Line Title\nMore content here."
    mock_pdf = MagicMock()
    mock_pdf.metadata = {}
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    with patch("pdfplumber.open", return_value=mock_pdf):
        title, body = extract_pdf(b"fake")
    assert title == "First Line Title"


def test_extract_pdf_empty_page_text_skipped():
    pytest.importorskip("pdfplumber")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pdf

    page1 = MagicMock()
    page1.extract_text.return_value = None  # empty page
    page2 = MagicMock()
    page2.extract_text.return_value = "Page 2 content."
    mock_pdf = MagicMock()
    mock_pdf.metadata = {}
    mock_pdf.pages = [page1, page2]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    with patch("pdfplumber.open", return_value=mock_pdf):
        title, body = extract_pdf(b"fake")
    assert "Page 2 content" in body
    assert title == "Page 2 content."


# ── extract_docx ─────────────────────────────────────────────────────────────


def test_extract_docx_uses_heading_as_title():
    pytest.importorskip("docx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_docx

    heading_para = MagicMock()
    heading_para.text = "  Introduction  "
    heading_para.style.name = "Heading 1"
    body_para = MagicMock()
    body_para.text = "This is the main content."
    body_para.style.name = "Normal"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [heading_para, body_para]
    with patch("docx.Document", return_value=mock_doc):
        title, body = extract_docx(b"fake docx")
    assert title == "Introduction"
    assert "main content" in body


def test_extract_docx_uses_first_paragraph_when_no_heading():
    pytest.importorskip("docx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_docx

    para1 = MagicMock()
    para1.text = "First paragraph as title."
    para1.style.name = "Normal"
    para2 = MagicMock()
    para2.text = "Second paragraph."
    para2.style.name = "Normal"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [para1, para2]
    with patch("docx.Document", return_value=mock_doc):
        title, body = extract_docx(b"fake")
    assert title == "First paragraph as title."


def test_extract_docx_skips_empty_paragraphs():
    pytest.importorskip("docx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_docx

    empty = MagicMock()
    empty.text = "   "
    empty.style.name = "Normal"
    real = MagicMock()
    real.text = "Real content"
    real.style.name = "Normal"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [empty, real]
    with patch("docx.Document", return_value=mock_doc):
        title, body = extract_docx(b"fake")
    assert title == "Real content"


# ── extract_xlsx ─────────────────────────────────────────────────────────────


def test_extract_xlsx_with_title_property():
    pytest.importorskip("openpyxl")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_xlsx

    ws = MagicMock()
    ws.iter_rows.return_value = [
        (None, "Column A", "Column B"),
        ("Row1", "Val1", "Val2"),
    ]
    wb = MagicMock()
    wb.properties.title = "My Spreadsheet"
    wb.sheetnames = ["Sheet1"]
    wb.__getitem__ = lambda s, k: ws
    with patch("openpyxl.load_workbook", return_value=wb):
        title, body = extract_xlsx(b"fake xlsx")
    assert title == "My Spreadsheet"
    assert "Sheet1" in body


def test_extract_xlsx_uses_sheet_name_when_no_title():
    pytest.importorskip("openpyxl")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_xlsx

    ws = MagicMock()
    ws.iter_rows.return_value = [("data",)]
    wb = MagicMock()
    wb.properties.title = None
    wb.sheetnames = ["DataSheet"]
    wb.__getitem__ = lambda s, k: ws
    with patch("openpyxl.load_workbook", return_value=wb):
        title, body = extract_xlsx(b"fake")
    assert title == "DataSheet"


# ── extract_pptx ─────────────────────────────────────────────────────────────


def test_extract_pptx_extracts_slide_text():
    pytest.importorskip("pptx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pptx

    para = MagicMock()
    para.text = "Slide Title Text"
    shape = MagicMock()
    shape.has_text_frame = True
    shape.text_frame.paragraphs = [para]
    slide = MagicMock()
    slide.shapes = [shape]
    prs = MagicMock()
    prs.slides = [slide]
    with patch("pptx.Presentation", return_value=prs):
        title, body = extract_pptx(b"fake pptx")
    assert title == "Slide Title Text"
    assert "Slide Title Text" in body
    assert "Slide 1" in body


def test_extract_pptx_skips_shapes_without_text_frame():
    pytest.importorskip("pptx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pptx

    image_shape = MagicMock()
    image_shape.has_text_frame = False
    text_shape = MagicMock()
    text_shape.has_text_frame = True
    para = MagicMock()
    para.text = "Real Text"
    text_shape.text_frame.paragraphs = [para]
    slide = MagicMock()
    slide.shapes = [image_shape, text_shape]
    prs = MagicMock()
    prs.slides = [slide]
    with patch("pptx.Presentation", return_value=prs):
        title, body = extract_pptx(b"fake")
    assert title == "Real Text"


def test_extract_pptx_multiple_slides():
    pytest.importorskip("pptx")
    from unittest.mock import MagicMock, patch

    from dewie.ingestion.extractors import extract_pptx

    def make_slide(text):
        para = MagicMock()
        para.text = text
        shape = MagicMock()
        shape.has_text_frame = True
        shape.text_frame.paragraphs = [para]
        slide = MagicMock()
        slide.shapes = [shape]
        return slide

    prs = MagicMock()
    prs.slides = [make_slide("Title Slide"), make_slide("Content Slide")]
    with patch("pptx.Presentation", return_value=prs):
        title, body = extract_pptx(b"fake")
    assert title == "Title Slide"
    assert "Slide 2" in body
    assert "Content Slide" in body

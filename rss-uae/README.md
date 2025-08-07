# RSS UAE Gazette Processing Module

This module provides improved RSS-based gazette processing for UAE government documents.

## Features

- **Multi-method PDF text extraction**: Uses pdfplumber, PyMuPDF, and PyPDF2
- **Quality scoring**: Automatically selects the best extraction method
- **OCR fallback**: Uses OCR only when other methods produce low-quality text
- **Two-stage AI processing**: Junior analyst + Senior analyst validation
- **Clean modular design**: Separated concerns into different classes

## Usage

```python
from src.rss_uae.gazette_rss_processor import GazetteRSSProcessor

processor = GazetteRSSProcessor()
processor.extract_gazette_from_rss()
```

## Components

1. **PDFTextExtractor**: Handles PDF text extraction with multiple methods
2. **DocumentProcessor**: Two-stage AI processing (junior + senior analyst)
3. **RSSProcessor**: RSS feed parsing and PDF downloading
4. **GazetteRSSProcessor**: Main orchestrator class

## Improvements over previous version

- Uses OCR only when needed (quality < 70)
- Two-stage validation for better accuracy
- Cleaner, more maintainable code structure
- Better error handling and logging
- Quality scoring for text extraction methods

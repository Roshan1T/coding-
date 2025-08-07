import os
import json
import requests
import fitz  # PyMuPDF
from typing import Optional, Dict, Any
from src.logger import logger
from src.ocr_extraction import ocr_extract
import tempfile

class PDFTextExtractor:
    """
    Advanced PDF text extraction with multiple methods and quality scoring
    """
    
    def __init__(self):
        pass
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        try:
            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for extracted text"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 0.0
        text_clean = text.strip()
        
        # Character diversity
        unique_chars = len(set(text_clean.lower()))
        char_diversity = min(unique_chars / 50, 1.0)
        score += char_diversity * 20
        
        # Word analysis
        import re
        words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]+\b', text_clean)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            word_len_score = 1.0 - abs(avg_word_len - 5.5) / 10
            score += max(word_len_score, 0) * 15
            
            valid_words = sum(1 for w in words if 2 <= len(w) <= 15)
            word_validity = valid_words / len(words) if words else 0
            score += word_validity * 20
        
        # Structure indicators
        if '|' in text or '\t' in text:
            score += 8
        
        # Sentence structure
        sentences = re.split(r'[.!?؟]+', text_clean)
        if sentences:
            reasonable_sentences = sum(1 for s in sentences if 10 <= len(s.strip()) <= 200)
            sentence_quality = reasonable_sentences / len(sentences) if sentences else 0
            score += sentence_quality * 12
        
        # Line structure
        lines = text_clean.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            avg_line_len = sum(len(line.strip()) for line in non_empty_lines) / len(non_empty_lines)
            line_score = 1.0 - abs(avg_line_len - 50) / 100
            score += max(line_score, 0) * 15
        
        return min(score, 100.0)
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text using multiple methods and return the best result
        Uses OCR only if other methods produce low quality text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
        
        # Try different extraction methods
        methods = [
            ("pdfplumber", self.extract_with_pdfplumber),
            ("PyMuPDF", self.extract_with_pymupdf),
            ("PyPDF2", self.extract_with_pypdf2),
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            try:
                text = method_func(pdf_path)
                if text and len(text.strip()) > 50:
                    quality_score = self._calculate_quality_score(text)
                    results[method_name] = {'text': text, 'quality_score': quality_score}
                    logger.info(f"{method_name}: {len(text)} chars, quality: {quality_score:.1f}")
                else:
                    logger.info(f"{method_name}: {len(text) if text else 0} chars (too short)")
            except Exception as e:
                logger.error(f"{method_name} failed: {e}")
        
        # Select best method
        if results:
            best_method = max(results.keys(), key=lambda k: results[k]['quality_score'])
            best_text = results[best_method]['text']
            best_quality = results[best_method]['quality_score']
            
            logger.info(f"Best method: {best_method} (quality score: {best_quality:.1f})")
            
            # Use OCR only if quality is low
            if best_quality < 70:
                logger.info("Quality is low, trying OCR extraction...")
                try:
                    # Use the existing ocr_extract function from src.ocr_extraction
                    md_file_path = ocr_extract(pdf_path, "UAE")
                    if md_file_path and os.path.exists(md_file_path):
                        with open(md_file_path, 'r', encoding='utf-8') as f:
                            ocr_text = f.read()
                        
                        # Clean up temp MD file
                        try:
                            os.remove(md_file_path)
                        except:
                            pass
                        
                        if ocr_text:
                            ocr_quality = self._calculate_quality_score(ocr_text)
                            logger.info(f"OCR: {len(ocr_text)} chars, quality: {ocr_quality:.1f}")
                            
                            # Use OCR if significantly better
                            if ocr_quality > best_quality + 15:
                                logger.info("OCR provided better quality, using OCR result")
                                return ocr_text
                
                except Exception as e:
                    logger.error(f"OCR backup failed: {e}")
            
            return best_text
        
        # If no good text extraction, try OCR as last resort
        logger.warning("No good text extraction, trying OCR as last resort...")
        try:
            md_file_path = ocr_extract(pdf_path, "UAE")
            if md_file_path and os.path.exists(md_file_path):
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
                
                # Clean up temp MD file
                try:
                    os.remove(md_file_path)
                except:
                    pass
                
                if ocr_text:
                    return ocr_text
        except Exception as e:
            logger.error(f"OCR last resort failed: {e}")
        
        raise Exception("All text extraction methods failed or produced poor quality results")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get basic PDF information"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                info = {
                    'pages': len(reader.pages),
                    'title': '',
                    'author': '',
                    'subject': '',
                    'creator': '',
                    'is_encrypted': reader.is_encrypted,
                    'has_extractable_text': False
                }
                
                try:
                    sample_text = reader.pages[0].extract_text()
                    info['has_extractable_text'] = bool(sample_text and len(sample_text.strip()) > 10)
                except:
                    pass
                
                if reader.metadata:
                    info['title'] = reader.metadata.get('/Title', '')
                    info['author'] = reader.metadata.get('/Author', '')
                    info['subject'] = reader.metadata.get('/Subject', '')
                    info['creator'] = reader.metadata.get('/Creator', '')
                
                return info
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return {'pages': 0, 'title': '', 'author': '', 'subject': '', 'creator': '', 
                   'is_encrypted': False, 'has_extractable_text': False}

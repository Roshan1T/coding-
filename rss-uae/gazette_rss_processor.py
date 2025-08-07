import os
import json
from typing import List, Dict, Any
from src.logger import logger
from src.themes.get_themes import get_themes_from_db
from src.check_if_file_processed import is_file_already_processed
from src import mongodb_client
from src.config import Config
from .pdf_extractor import PDFTextExtractor
from .document_processor import DocumentProcessor
from .rss_processor import RSSProcessor

class GazetteRSSProcessor:
    """
    Main processor for UAE RSS gazette extraction
    """
    
    def __init__(self):
        self.pdf_extractor = PDFTextExtractor()
        self.document_processor = DocumentProcessor()
        self.rss_processor = RSSProcessor()
    
    def save_in_database(self, data: List[Dict[str, Any]]) -> None:
        """Save extracted data to MongoDB"""
        try:
            db_name = mongodb_client[Config.MONGODB_NAME]
            col = db_name[Config.MONGODB_COLLECTION]
            
            for data_block in data:
                if isinstance(data_block, dict):
                    try:
                        col.insert_one(data_block)
                        logger.info(f"Inserted: {data_block.get('unique_id', 'unknown')} in database")
                    except Exception as e:
                        logger.error(f"Error inserting document: {e}")
                else:
                    logger.error(f"Invalid data block: {data_block}")
            
            logger.info(f"Saved {len(data)} documents to database")
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    
    def process_single_pdf(self, pdf_path: str, themes: List[str]) -> List[Dict[str, Any]]:
        """Process a single PDF file"""
        try:
            pdf_filename = os.path.basename(pdf_path)
            logger.info(f"Processing PDF: {pdf_filename}")
            
            # Extract text using multiple methods
            logger.info("Extracting text from PDF...")
            extracted_text = self.pdf_extractor.extract_text(pdf_path)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise Exception("Insufficient text extracted from PDF")
            
            logger.info(f"Extracted {len(extracted_text)} characters of text")
            
            # Process with AI (2-step validation)
            logger.info("Processing with AI analysis...")
            structured_data = self.document_processor.process_document(
                extracted_text, themes, pdf_filename
            )
            
            logger.info(f"Successfully processed document: {pdf_filename}")
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            raise
    
    def extract_gazette_from_rss(self, rss_url: str = None) -> None:
        """Main method to extract gazette data from RSS feed"""
        
        # Default RSS URL for UAE DOH
        if not rss_url:
            rss_url = "https://www.doh.gov.ae/en/resources/guidelines-rss-feed"
        
        jurisdiction = "UAE"
        iso_country_code = "AE"
        
        logger.info(f"Starting RSS processing for: {rss_url}")
        
        # Load themes
        logger.info("Loading themes...")
        try:
            get_themes_from_db()
            with open("src/themes/data/themes_to_include.json", "r", encoding="utf-8") as file:
                themes = json.load(file)
            logger.info(f"Loaded {len(themes)} themes from JSON file")
        except Exception as e:
            logger.warning(f"Failed to load themes from JSON file: {e}")
            themes = ["Healthcare", "Public Health", "Medical Devices", "Pharmaceuticals"]
        
        # Extract PDF links from RSS
        pdf_links = self.rss_processor.extract_links_from_rss(rss_url)
        if not pdf_links:
            logger.warning("No PDF links found in RSS feed")
            return
        
        # Process statistics
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Process each PDF
        for pdf_url in pdf_links:
            try:
                # Check if already processed
                if is_file_already_processed(iso_country_code.upper(), pdf_url):
                    logger.info(f"File {pdf_url} already processed. Skipping...")
                    skipped_count += 1
                    continue
                
                logger.info(f"Processing PDF from URL: {pdf_url}")
                
                # Download PDF
                temp_pdf_path = self.rss_processor.download_pdf_from_url(pdf_url)
                if not temp_pdf_path:
                    logger.error(f"Failed to download PDF from {pdf_url}")
                    failed_count += 1
                    continue
                
                try:
                    # Process PDF
                    extracted_data = self.process_single_pdf(temp_pdf_path, themes)
                    
                    # Save to JSON file
                    output_dir = f"output/{jurisdiction}/json"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    json_filename = os.path.basename(temp_pdf_path).rsplit('.', 1)[0] + "_analysis.json"
                    json_file_path = os.path.join(output_dir, json_filename)
                    
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                    
                    # Save to database
                    self.save_in_database(extracted_data)
                    
                    processed_count += 1
                    logger.info(f"Successfully processed {os.path.basename(temp_pdf_path)}")
                    
                finally:
                    # Clean up temporary file
                    self.rss_processor.cleanup_temp_file(temp_pdf_path)
                
            except Exception as e:
                logger.error(f"Failed to process PDF from {pdf_url}: {e}")
                failed_count += 1
                continue
        
        # Final summary
        logger.info(f"RSS processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")
        
        if processed_count == 0:
            logger.warning("No documents were successfully processed")
        else:
            logger.info(f"Successfully processed {processed_count} documents")

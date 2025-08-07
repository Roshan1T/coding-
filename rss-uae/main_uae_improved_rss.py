"""
Improved UAE RSS Gazette Processing
Uses modular approach with multiple PDF extraction methods and two-stage AI validation
"""

from src.logger import logger
from .gazette_rss_processor import GazetteRSSProcessor

def extract_gazette():
    """Main entry point for UAE RSS gazette extraction"""
    try:
        logger.info("Starting improved UAE RSS gazette processing...")
        
        processor = GazetteRSSProcessor()
        processor.extract_gazette_from_rss()
        
        logger.info("UAE RSS gazette processing completed successfully")
        
    except Exception as e:
        logger.error(f"UAE RSS gazette processing failed: {e}")
        raise

# Backward compatibility
def extract_gazette_from_rss_simple():
    """Backward compatibility function"""
    extract_gazette()
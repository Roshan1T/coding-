import os
import json
import requests
import feedparser
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Optional
from src.logger import logger

class RSSProcessor:
    """
    Handle RSS feed processing and PDF download
    """
    
    def __init__(self):
        pass
    
    def extract_links_from_rss(self, rss_url: str) -> List[str]:
        """Extract PDF links from RSS feed"""
        try:
            logger.info(f"Fetching RSS feed: {rss_url}")
            response = requests.get(rss_url, verify=False, timeout=30)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            links = []
            
            for item in feed.entries:
                if hasattr(item, 'link') and item.link:
                    links.append(item.link)
            
            logger.info(f"Extracted {len(links)} links from RSS feed")
            return links
            
        except Exception as e:
            logger.error(f"Failed to extract links from RSS: {e}")
            return []
    
    def download_pdf_from_url(self, url: str, temp_folder: str = "temp_pdf") -> Optional[str]:
        """Download PDF from URL to temporary location"""
        os.makedirs(temp_folder, exist_ok=True)
        
        try:
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Generate filename
            filename = os.path.basename(urlparse(url).path)
            if not filename or '.' not in filename:
                filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            elif not filename.endswith(".pdf"):
                filename = filename.rsplit(".", 1)[0] + ".pdf"
            
            filepath = os.path.join(temp_folder, filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {filename} ({len(response.content)} bytes)")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def cleanup_temp_file(self, filepath: str) -> None:
        """Clean up temporary file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {os.path.basename(filepath)}")
        except Exception as e:
            logger.warning(f"Failed to clean up {filepath}: {e}")

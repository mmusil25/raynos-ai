#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Simplified CPU-compatible version of Gemma JSON Extractor
Falls back to mock extraction if models are not available
"""

import json
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using mock extraction")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - using mock extraction")


class JSONExtractor:
    """Base class for JSON extraction from transcripts"""
    
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract structured JSON from transcript"""
        raise NotImplementedError


class SimpleExtractor(JSONExtractor):
    """Simple rule-based extractor for CPU-only environments"""
    
    def __init__(self):
        logger.info("Using simple rule-based extractor (CPU-friendly)")
        
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract information using simple rules"""
        if not transcript:
            return self._empty_result(timestamp_ms)
        
        # Get timestamp
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        timestamp_str = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()
        
        # Extract entities using simple patterns
        result = {
            "timestamp": timestamp_str,
            "timestamp_ms": timestamp_ms,
            "transcript": transcript[:500],  # Limit length
            "entities": {
                "people": self._extract_names(transcript),
                "locations": self._extract_locations(transcript),
                "dates": self._extract_dates(transcript),
                "topics": self._extract_topics(transcript)
            },
            "summary": self._generate_summary(transcript),
            "confidence": 0.7  # Fixed confidence for rule-based
        }
        
        return result
    
    def _empty_result(self, timestamp_ms: int = None) -> Dict:
        """Return empty result structure"""
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        return {
            "timestamp": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
            "timestamp_ms": timestamp_ms,
            "transcript": "",
            "entities": {
                "people": [],
                "locations": [],
                "dates": [],
                "topics": []
            },
            "summary": "",
            "confidence": 0.0
        }
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract potential names (capitalized words)"""
        # Simple heuristic: consecutive capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        names = [m for m in matches if m not in common_words and len(m.split()) <= 3]
        return list(set(names))[:5]  # Limit to 5 names
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract potential locations"""
        # Look for common location indicators
        location_indicators = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        locations = []
        for pattern in location_indicators:
            matches = re.findall(pattern, text)
            locations.extend(matches)
        
        return list(set(locations))[:5]  # Limit to 5 locations
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates and time references"""
        patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(?:today|tomorrow|yesterday)\b',
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))[:5]  # Limit to 5 dates
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics (most common significant words)"""
        # Remove common words and extract significant terms
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                           'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                           'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                           'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'])
        
        # Simple word frequency analysis
        words = re.findall(r'\b[a-z]+\b', text.lower())
        word_counts = {}
        for word in words:
            if word not in common_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top topics
        topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in topics[:5]]  # Top 5 topics
    
    def _generate_summary(self, text: str) -> str:
        """Generate a simple summary (first few sentences)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        # Return first 2 sentences or up to 200 characters
        summary = '. '.join(sentences[:2])
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return summary


class ExtractionManager:
    """Manager for handling extraction with fallbacks"""
    
    def __init__(self, extractor_type="auto", **extractor_kwargs):
        """
        Initialize extraction manager
        
        Args:
            extractor_type: "auto", "simple", "gemma", or "mock"
            **extractor_kwargs: Arguments to pass to the extractor
        """
        self.extraction_history = []
        
        # Auto-detect best available extractor
        if extractor_type == "auto":
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                try:
                    # Try loading Gemma if available
                    from gemma_3n_json_extractor import GemmaExtractor
                    self.extractor = GemmaExtractor(**extractor_kwargs)
                    logger.info("Using Gemma extractor (GPU)")
                except Exception as e:
                    logger.warning(f"Could not load Gemma extractor: {e}")
                    self.extractor = SimpleExtractor()
            else:
                self.extractor = SimpleExtractor()
        elif extractor_type == "simple":
            self.extractor = SimpleExtractor()
        elif extractor_type == "mock":
            from gemma_3n_json_extractor import MockExtractor
            self.extractor = MockExtractor()
        else:
            # Try to load specific extractor
            try:
                from gemma_3n_json_extractor import GemmaExtractor
                self.extractor = GemmaExtractor(**extractor_kwargs)
            except Exception as e:
                logger.warning(f"Could not load {extractor_type} extractor: {e}")
                self.extractor = SimpleExtractor()
    
    def extract_from_transcript(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract JSON from a single transcript"""
        try:
            result = self.extractor.extract(transcript, timestamp_ms)
            
            # Store in history
            self.extraction_history.append({
                "timestamp": time.time(),
                "transcript": transcript[:100],  # Store preview
                "result": result
            })
            
            return result
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return a basic structure on error
            return {
                "error": str(e),
                "transcript": transcript[:500],
                "timestamp_ms": timestamp_ms or int(time.time() * 1000)
            }
    
    def get_extraction_history(self) -> List[Dict]:
        """Get extraction history"""
        return self.extraction_history
    
    def clear_history(self):
        """Clear extraction history"""
        self.extraction_history = []


# For backward compatibility
def create_extractor(use_gpu=None):
    """Create an appropriate extractor based on available resources"""
    if use_gpu is None:
        use_gpu = TORCH_AVAILABLE
    
    if use_gpu and TRANSFORMERS_AVAILABLE:
        return ExtractionManager(extractor_type="auto")
    else:
        return ExtractionManager(extractor_type="simple")


if __name__ == "__main__":
    # Test the extractor
    manager = create_extractor()
    
    test_transcript = """
    Hello, my name is John Smith and I'm calling from New York City.
    I wanted to discuss the meeting scheduled for tomorrow at 2:30 PM.
    We'll be talking about the new product launch in California next month.
    """
    
    result = manager.extract_from_transcript(test_transcript)
    print(json.dumps(result, indent=2))
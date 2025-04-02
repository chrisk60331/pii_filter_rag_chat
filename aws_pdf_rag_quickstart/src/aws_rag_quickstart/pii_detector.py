"""
PII detection module using a Hugging Face model without fine-tuning.
This module provides functionality to detect and filter PII in user inputs.
"""

import logging
from typing import Dict, List, Tuple, Union

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Class to detect PII using a Hugging Face NER model.
    The model is used to identify potential PII in user inputs.
    """
    
    def __init__(self, model_name: str = "molise-ai/pii-detector-ai4privacy"):
        """
        Initialize the PII detector with a pretrained model.
        
        Args:
            model_name: The Hugging Face model to use for NER detection
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading PII detection model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple"
            )
            logger.info("PII detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PII detection model: {e}")
            raise
    
    def detect_pii(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Detect potential PII in the provided text.
        
        Args:
            text: The text to analyze for PII
            
        Returns:
            A list of dictionaries containing entity information
        """
        try:
            results = self.ner_pipeline(text)
            logger.info(f"PII detection results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error detecting PII: {e}")
            return []
    
    def has_pii(self, text: str, threshold: float = 0.8) -> Tuple[bool, List[Dict]]:
        """
        Check if the text contains PII based on entity recognition.
        
        Args:
            text: The text to check for PII
            threshold: Confidence threshold for PII detection
            
        Returns:
            Tuple containing:
            - Boolean indicating if PII was detected
            - List of detected entities that triggered the detection
        """
        entities = self.detect_pii(text)
        
        # molise-ai/pii-detector-ai4privacy model entity types
        # Note: This model returns entity_group instead of entity
        pii_entity_types = {
            "EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "SSN", 
            "PERSON", "LOCATION", "DATE_TIME", "IP_ADDRESS", "USER",
            "SOCIALNUM", "B-SOCIALNUM", "I-SOCIALNUM"  # Adding SOCIALNUM which is used for SSNs
        }
        
        # Filter for high-confidence PII entities
        detected_pii = [
            entity for entity in entities
            if (entity["entity_group"] in pii_entity_types and 
                entity["score"] >= threshold)
        ]
        
        return bool(detected_pii), detected_pii
    
    def filter_text(self, text: str, threshold: float = 0.8) -> Tuple[bool, str, List[Dict]]:
        """
        Filter text and provide feedback if PII is detected.
        
        Args:
            text: The text to filter for PII
            threshold: Confidence threshold for PII detection
            
        Returns:
            Tuple containing:
            - Boolean indicating if text is safe (no PII detected)
            - Message explaining the result
            - List of detected PII entities if any
        """
        has_pii_content, detected_entities = self.has_pii(text, threshold)
        
        if has_pii_content:
            entity_descriptions = [
                f"{entity['word']} ({entity['entity_group']}, {entity['score']:.2f})"
                for entity in detected_entities
            ]
            
            message = (
                "Potential PII detected in your message. Please remove personal information "
                f"such as: {', '.join(entity_descriptions)}"
            )
            return False, message, detected_entities
        
        return True, "No PII detected", [] 
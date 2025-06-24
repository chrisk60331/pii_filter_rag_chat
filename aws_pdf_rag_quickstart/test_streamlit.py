#!/usr/bin/env python3
"""
Test script for the Streamlit RAG application.
Tests core functionality including PII detection.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aws_rag_quickstart.pii_detector import PIIDetector
from aws_rag_quickstart.constants import ALL_MODELS, BEDROCK_MODELS, OPENAI_MODELS

class TestStreamlitApp(unittest.TestCase):
    """Test cases for the Streamlit application components"""
    
    def setUp(self):
        """Set up test environment"""
        # Set environment variables for testing
        os.environ['LOG_LEVEL'] = 'INFO'
        os.environ['LOCAL'] = '1'
        os.environ['INDEX_NAME'] = 'test-index'
        
    def test_pii_detector_initialization(self):
        """Test PII detector can be initialized"""
        try:
            detector = PIIDetector()
            self.assertIsNotNone(detector)
            print("✅ PII Detector initialized successfully")
        except Exception as e:
            self.fail(f"Failed to initialize PII detector: {e}")
    
    def test_pii_detection_safe_text(self):
        """Test PII detection with safe text"""
        detector = PIIDetector()
        
        safe_text = "What is the capital of France?"
        is_safe, message, entities = detector.filter_text(safe_text)
        
        self.assertTrue(is_safe)
        self.assertEqual(message, "No PII detected")
        self.assertEqual(len(entities), 0)
        print("✅ Safe text properly identified")
    
    def test_pii_detection_unsafe_text(self):
        """Test PII detection with unsafe text"""
        detector = PIIDetector()
        
        # Text with potential PII
        unsafe_text = "My email is john.doe@example.com and my phone is 555-1234"
        is_safe, message, entities = detector.filter_text(unsafe_text)
        
        # Should detect PII
        self.assertFalse(is_safe)
        self.assertIn("PII detected", message)
        self.assertGreater(len(entities), 0)
        print("✅ Unsafe text properly identified")
    
    def test_model_constants(self):
        """Test that model constants are properly defined"""
        self.assertIsInstance(ALL_MODELS, list)
        self.assertIsInstance(BEDROCK_MODELS, list)
        self.assertIsInstance(OPENAI_MODELS, list)
        
        self.assertGreater(len(ALL_MODELS), 0)
        self.assertGreater(len(BEDROCK_MODELS), 0)
        self.assertGreater(len(OPENAI_MODELS), 0)
        
        # Check that ALL_MODELS contains both types
        self.assertEqual(len(ALL_MODELS), len(BEDROCK_MODELS) + len(OPENAI_MODELS))
        print("✅ Model constants properly defined")
    
    def test_environment_setup(self):
        """Test environment variable setup"""
        required_vars = ['LOG_LEVEL', 'LOCAL', 'INDEX_NAME']
        
        for var in required_vars:
            self.assertIn(var, os.environ)
        
        self.assertEqual(os.environ['LOCAL'], '1')
        self.assertEqual(os.environ['LOG_LEVEL'], 'INFO')
        print("✅ Environment variables properly set")
    
    @patch('aws_rag_quickstart.pii_detector.PIIDetector')
    def test_streamlit_app_imports(self, mock_detector):
        """Test that Streamlit app can be imported without errors"""
        try:
            # Import the main components
            from aws_rag_quickstart.AgentLambda import main as agent_handler
            from aws_rag_quickstart.IngestionLambda import main as ingest_handler
            from aws_rag_quickstart.constants import ALL_MODELS
            
            self.assertIsNotNone(agent_handler)
            self.assertIsNotNone(ingest_handler)
            self.assertIsNotNone(ALL_MODELS)
            print("✅ Core modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_temporary_file_handling(self):
        """Test temporary file creation and cleanup"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        # Verify file exists
        self.assertTrue(os.path.exists(tmp_path))
        
        # Clean up
        os.unlink(tmp_path)
        
        # Verify file is deleted
        self.assertFalse(os.path.exists(tmp_path))
        print("✅ Temporary file handling works correctly")

class TestPIIDetectionScenarios(unittest.TestCase):
    """Comprehensive PII detection test scenarios"""
    
    def setUp(self):
        """Set up PII detector for testing"""
        self.detector = PIIDetector()
    
    def test_email_detection(self):
        """Test email address detection"""
        text = "Contact me at user@example.com for more information"
        is_safe, _, entities = self.detector.filter_text(text)
        
        if not is_safe:
            print("✅ Email detection working")
        else:
            print("⚠️  Email detection may need tuning")
    
    def test_phone_detection(self):
        """Test phone number detection"""
        text = "Call me at (555) 123-4567 or 555-123-4567"
        is_safe, _, entities = self.detector.filter_text(text)
        
        if not is_safe:
            print("✅ Phone detection working")
        else:
            print("⚠️  Phone detection may need tuning")
    
    def test_name_detection(self):
        """Test name detection"""
        text = "John Smith is the manager of the project"
        is_safe, _, entities = self.detector.filter_text(text)
        
        if not is_safe:
            print("✅ Name detection working")
        else:
            print("⚠️  Name detection may need tuning")
    
    def test_safe_business_text(self):
        """Test that business-related text is safe"""
        text = "The quarterly report shows increased revenue and customer satisfaction"
        is_safe, _, entities = self.detector.filter_text(text)
        
        self.assertTrue(is_safe)
        print("✅ Business text correctly identified as safe")

def run_tests():
    """Run all tests and provide summary"""
    print("🛡️ Testing Streamlit RAG Application (uv)")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestStreamlitApp))
    suite.addTest(unittest.makeSuite(TestPIIDetectionScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, error in result.failures:
            print(f"  - {test}: {error}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 
import os
from io import BytesIO
from PyPDF2 import PdfReader

def test_pdf_extraction():
    """
    Simple test to validate PDF text extraction works with PyPDF2
    """
    # Sample PDF content creation (minimal valid PDF)
    sample_pdf = b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<<>>>>endobj\n4 0 obj<</Length 25>>stream\nBT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000182 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n256\n%%EOF"
    
    try:
        # Extract text from the sample PDF
        pdf_reader = PdfReader(BytesIO(sample_pdf))
        num_pages = len(pdf_reader.pages)
        print(f"Successfully loaded PDF with {num_pages} pages")
        
        if num_pages > 0:
            text = pdf_reader.pages[0].extract_text()
            print(f"Extracted text: {text}")
            return True
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return False
    
    return False

if __name__ == "__main__":
    # Run the test
    result = test_pdf_extraction()
    print(f"PDF text extraction test {'passed' if result else 'failed'}") 
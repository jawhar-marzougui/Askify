from llama_parse import LlamaParse
import os
from dotenv import load_dotenv
from langdetect import detect
import fitz  # PyMuPDF for text extraction

# Load environment variables
load_dotenv()

def extract_text_simple(file_path, max_chars=1000):
    """Extract a simple text from the PDF for language detection."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        return text
    except Exception as e:
        print("Failed to extract text for language detection:", e)
        return ""

def detect_language(file_path):
    """Detect the language of the document."""
    text = extract_text_simple(file_path, max_chars=1000)
    if not text.strip():
        return "en"  # Default to English if no text is found
    try:
        lang = detect(text)
        if lang.startswith("en"):
            return "en"
        else:
            return "en"  # Default to English
    except Exception:
        return "en"  # Fallback to English

def parse_pdf(file_path):
    """Parse PDF file using LlamaParse and return parsed result object"""
    try:
        api_key = os.getenv("LLAMA_PARSE_API_KEY", "llx-OEmPORU2LO31OVFp99ha56cNOSn0wu3GsorXExYWpzfCrCG5")
        
        # Ensure file path is absolute and properly formatted
        abs_path = os.path.abspath(file_path)
        
        # Use English as default for parsing
        parser = LlamaParse(
            api_key=api_key,
            num_workers=4,
            verbose=False,
            language="en",
        )
        result = parser.parse(abs_path)
        return result
                
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def get_markdown_documents(result, split_by_page=True):
    """Get markdown documents from parsed result"""
    if result is None:
        return []
    return result.get_markdown_documents(split_by_page=split_by_page)

def get_text_documents(result, split_by_page=False):
    """Get text documents from parsed result"""
    if result is None:
        return []
    return result.get_text_documents(split_by_page=split_by_page)

def get_image_documents(result, include_screenshot_images=True, include_object_images=False, image_download_dir=None):
    """Get image documents from parsed result"""
    if result is None:
        return []
    return result.get_image_documents(
        include_screenshot_images=include_screenshot_images,
        include_object_images=include_object_images,
        image_download_dir=image_download_dir,
    )

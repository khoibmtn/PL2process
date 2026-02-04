"""
Preprocessing module for PL2process.
Contains text normalization, diacritics removal, stopword removal, and numeric word conversion.

CORRECT PREPROCESSING ORDER (CRITICAL):
1. Lowercase
2. Convert Vietnamese numeric words to digits (WITH DIACRITICS STILL PRESENT)
3. Remove Vietnamese diacritics (Unicode normalization)
4. Remove stopwords (using non-diacritic forms)
5. Normalize whitespace

IMPORTANT: Numeric conversion MUST happen BEFORE diacritic removal to preserve semantics.
Example: "sáu" (six) ≠ "sau" (after) - these are different words!
"""

import re
import logging
from unidecode import unidecode

# Configure logging for warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# Vietnamese number words mapping (WITH DIACRITICS - for use BEFORE diacritic removal)
# 
# IMPORTANT: "không" is EXCLUDED because in Vietnamese medical/administrative text,
# it typically means "no / not present" (negation), NOT the number zero.
# Example: "không ghi nhận biến chứng" = "no complications noted"
# ============================================================================
NUMBER_WORDS_WITH_DIACRITICS = {
    # "không" INTENTIONALLY EXCLUDED - it's a negation, not a number
    "một": "1",     # one
    "hai": "2",     # two
    "ba": "3",      # three
    "bốn": "4",     # four
    "năm": "5",     # five
    "sáu": "6",     # six (NOT "sau" which means "after")
    "bảy": "7",     # seven
    "tám": "8",     # eight
    "chín": "9",    # nine
    "mười": "10"    # ten
}

# ============================================================================
# Stopwords (WITHOUT diacritics - for use AFTER diacritic removal)
# ============================================================================
STOPWORDS = {"ky thuat", "phuong phap", "duong", "bang", "qua", "tai", "va"}


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: trim and collapse multiple spaces to single space.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def convert_numeric_words_with_diacritics(text: str) -> str:
    """
    Convert Vietnamese numeric words to Arabic numerals.
    
    CRITICAL: This function expects text WITH DIACRITICS.
    It matches Vietnamese number words with their exact diacritic forms.
    
    RULES:
    - Match ONLY exact Vietnamese number words WITH DIACRITICS
    - Use word-boundary-aware matching
    - Do NOT convert if diacritics are already removed (fail safe)
    
    Example: 
        "sáu lần" → "6 lần"  (✓ correct - "sáu" is six)
        "sau đó"  → "sau đó" (✓ correct - "sau" is after, NOT converted)
    
    Args:
        text: Text WITH Vietnamese diacritics still present
        
    Returns:
        Text with numeric words converted to digits
    """
    if not isinstance(text, str):
        logger.warning(f"convert_numeric_words received non-string: {type(text)}")
        return ""
    
    if not text.strip():
        return ""
    
    try:
        result = text
        for word, digit in NUMBER_WORDS_WITH_DIACRITICS.items():
            # Use word boundary regex to match whole words only
            # \b works for word boundaries including Vietnamese characters
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
        
        return normalize_whitespace(result)
    
    except Exception as e:
        logger.warning(f"Error in convert_numeric_words for text '{text[:50]}...': {e}")
        return text  # Return original text on error (fail safe)


def remove_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics from text using unidecode.
    
    Example: "kỹ thuật" → "ky thuat"
    
    IMPORTANT: This should be called AFTER numeric word conversion.
    """
    if not isinstance(text, str):
        return ""
    return unidecode(text)


def remove_stopwords(text: str, stopwords: set = None) -> str:
    """
    Remove stopwords as whole words only, not as substrings.
    
    IMPORTANT: This function expects text WITHOUT diacritics.
    Stopwords are defined without diacritics for post-diacritic-removal matching.
    
    Example: "ky thuat mo" → "mo"
    """
    if not isinstance(text, str):
        return ""
    
    if stopwords is None:
        stopwords = STOPWORDS
    
    try:
        result = text
        for stopword in stopwords:
            # Use word boundary regex for whole word matching
            pattern = r'\b' + re.escape(stopword) + r'\b'
            result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
        
        return normalize_whitespace(result)
    
    except Exception as e:
        logger.warning(f"Error in remove_stopwords for text '{text[:50]}...': {e}")
        return text


def preprocess_step1_lowercase(text: str) -> str:
    """
    Step 1: Lowercase the text.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def preprocess_step2_numeric_conversion(text: str) -> str:
    """
    Step 2: Convert Vietnamese numeric words to digits.
    
    CRITICAL: Must be called BEFORE diacritic removal.
    Text should still have Vietnamese diacritics.
    """
    return convert_numeric_words_with_diacritics(text)


def preprocess_step3_remove_diacritics(text: str) -> str:
    """
    Step 3: Remove Vietnamese diacritics (Unicode normalization).
    """
    return remove_diacritics(text)


def preprocess_step4_remove_stopwords(text: str) -> str:
    """
    Step 4: Remove stopwords.
    
    IMPORTANT: Must be called AFTER diacritic removal.
    """
    return remove_stopwords(text)


def preprocess_name(text: str) -> str:
    """
    Apply full preprocessing pipeline to a procedure/chapter name.
    
    CORRECT ORDER:
    1. Lowercase
    2. Convert numeric words to digits (WITH diacritics still present)
    3. Remove Vietnamese diacritics
    4. Remove stopwords
    5. Normalize whitespace
    
    Args:
        text: Original procedure or chapter name
        
    Returns:
        Fully normalized text ready for matching
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    result = preprocess_step1_lowercase(text)
    
    # Step 2: Convert numeric words (BEFORE diacritic removal!)
    result = preprocess_step2_numeric_conversion(result)
    
    # Step 3: Remove Vietnamese diacritics
    result = preprocess_step3_remove_diacritics(result)
    
    # Step 4: Remove stopwords
    result = preprocess_step4_remove_stopwords(result)
    
    # Step 5: Final whitespace cleanup
    result = normalize_whitespace(result)
    
    return result


# ============================================================================
# Helper functions for UI intermediate columns
# ============================================================================

def preprocess_after_lowercase(text: str) -> str:
    """Return text after Step 1 (lowercase only)."""
    if not isinstance(text, str):
        return ""
    return preprocess_step1_lowercase(text)


def preprocess_after_numeric(text: str) -> str:
    """Return text after Step 2 (lowercase + numeric conversion)."""
    if not isinstance(text, str):
        return ""
    step1 = preprocess_step1_lowercase(text)
    step2 = preprocess_step2_numeric_conversion(step1)
    return step2


def preprocess_after_diacritics(text: str) -> str:
    """Return text after Step 3 (lowercase + numeric + diacritics removed)."""
    if not isinstance(text, str):
        return ""
    step1 = preprocess_step1_lowercase(text)
    step2 = preprocess_step2_numeric_conversion(step1)
    step3 = preprocess_step3_remove_diacritics(step2)
    return step3


if __name__ == "__main__":
    # Test cases demonstrating the CORRECT order
    test_cases = [
        ("Sáu lần điều trị", "CRITICAL: 'sáu' (six) should become 6"),
        ("Sau đó nghỉ ngơi", "CRITICAL: 'sau' (after) should NOT become 6"),
        ("Một lần phẫu thuật", "Expected: 1 lan phau thuat"),
        ("Năm bước điều trị", "Expected: 5 buoc dieu tri"),
        ("Kỹ thuật mổ nội soi", "Expected: mo noi soi"),
        ("Bảy ngày hậu phẫu", "Expected: 7 ngay hau phau"),
        # CRITICAL: "không" should NOT be converted (it's negation, not zero)
        ("Không ghi nhận biến chứng", "CRITICAL: 'không' should stay as 'khong' (negation)"),
        ("Không có tổn thương", "CRITICAL: 'không' should stay as 'khong' (negation)"),
    ]
    
    print("=" * 70)
    print("PREPROCESSING TEST - CORRECT ORDER")
    print("Step 1: Lowercase")
    print("Step 2: Numeric conversion (WITH diacritics)")
    print("Step 3: Remove diacritics")
    print("Step 4: Remove stopwords")
    print("=" * 70)
    
    for text, expected in test_cases:
        step1 = preprocess_after_lowercase(text)
        step2 = preprocess_after_numeric(text)
        step3 = preprocess_after_diacritics(text)
        final = preprocess_name(text)
        
        print(f"\nInput:        {text}")
        print(f"After Step 1: {step1}")
        print(f"After Step 2: {step2}  ← Numbers converted HERE (with diacritics)")
        print(f"After Step 3: {step3}")
        print(f"Final:        {final}")
        print(f"({expected})")
        print("-" * 70)

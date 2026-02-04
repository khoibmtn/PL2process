"""
AI Embedding Refinement Module for PL2process Step 2.

This module provides OPTIONAL AI-based refinement for matching results.
It applies ONLY to selected records (configurable score range) and
respects ALL exclusion rules.

IMPORTANT:
- This is REFINEMENT, not replacement
- AI MUST NEVER override exclusion rules
- AI MUST NEVER force a match
- All changes are auditable and reversible
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT AI REFINEMENT CONFIGURATION
# ============================================================================
DEFAULT_AI_CONFIG = {
    "enabled": False,
    "min_score": 50,
    "max_score": 70,
    "alpha": 0.7  # Weight for original score (0.7 = 70% original, 30% embedding)
}


# ============================================================================
# EXCLUSION RULE KEYWORDS (for Vietnamese medical terminology)
# ============================================================================

# Rule 1: Diagnostic/monitoring vs Treatment/surgical
DIAGNOSTIC_KEYWORDS = [
    "chan doan", "xet nghiem", "sieu am", "chup", "noi soi chan doan",
    "theo doi", "do", "kiem tra", "danh gia", "khao sat", "phat hien",
    "test", "screening", "monitoring"
]

TREATMENT_KEYWORDS = [
    "phau thuat", "cat", "bo", "thay", "ghep", "cay", "dieu tri",
    "dieu chinh", "sua chua", "mo", "khau", "noi", "tao hinh",
    "surgical", "operation", "therapy"
]

# Rule 2: Surgical vs YHCT/non-surgical
SURGICAL_KEYWORDS = [
    "phau thuat", "mo", "cat", "ghep", "thay", "bo", "khau",
    "noi soi can thiep", "tan soi", "laser", "dong dien", "dao dien"
]

YHCT_NONSURGICAL_KEYWORDS = [
    "yhct", "y hoc co truyen", "cham cuu", "xoa bop", "bam huyet",
    "giac hoi", "thuy cham", "dien cham", "cuu", "day an",
    "thuoc nam", "thuoc bac", "dong y"
]

# Rule 3: Anatomical systems
ANATOMICAL_SYSTEMS = {
    "cardiovascular": ["tim", "mach", "dong mach", "tinh mach", "mau", "huyet ap"],
    "respiratory": ["phoi", "phe quan", "ho hap", "khi quan", "mui", "hong"],
    "digestive": ["da day", "ruot", "gan", "mat", "tuy", "thuc quan", "tieu hoa"],
    "urinary": ["than", "bang quang", "nieu", "tiet nieu"],
    "nervous": ["nao", "tuy song", "than kinh", "nao bo"],
    "musculoskeletal": ["xuong", "khop", "co", "gay", "chan thuong"],
    "reproductive": ["sinh duc", "tu cung", "buong trung", "tien liet tuyen"],
    "dermatology": ["da", "long", "mong", "niem mac"],
    "ophthalmology": ["mat", "thi luc", "nhan cau", "giac mac"],
    "ent": ["tai", "mui", "hong", "thanh quan"]
}

# Rule 4: Sub-procedure indicators
SUB_PROCEDURE_INDICATORS = [
    "mo rong", "ket hop", "bo sung", "tang cuong", "phuc tap",
    "co dinh", "co su dung", "kem theo", "dong thoi"
]

# Rule 5: YHCT/PHCN domain constraint (reuse from matching.py)
RESTRICTED_PL2_DOMAINS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang"]
ALLOWED_PL1_CHAPTERS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang", "nhi"]


# ============================================================================
# EXCLUSION RULE FUNCTIONS
# ============================================================================

def contains_any_keyword(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords."""
    if not text:
        return False
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in keywords)


def get_anatomical_system(text: str) -> Optional[str]:
    """Identify the anatomical system from text."""
    if not text:
        return None
    text_lower = str(text).lower()
    for system, keywords in ANATOMICAL_SYSTEMS.items():
        if any(kw in text_lower for kw in keywords):
            return system
    return None


def rule1_diagnostic_vs_treatment(pl1_tenkt: str, pl2_tenkt: str) -> bool:
    """
    Rule 1: Diagnostic vs Treatment mismatch
    Returns True if should EXCLUDE (one is diagnostic, other is treatment)
    """
    pl1_is_diagnostic = contains_any_keyword(pl1_tenkt, DIAGNOSTIC_KEYWORDS)
    pl1_is_treatment = contains_any_keyword(pl1_tenkt, TREATMENT_KEYWORDS)
    pl2_is_diagnostic = contains_any_keyword(pl2_tenkt, DIAGNOSTIC_KEYWORDS)
    pl2_is_treatment = contains_any_keyword(pl2_tenkt, TREATMENT_KEYWORDS)
    
    # Exclude if one is diagnostic and other is treatment
    if (pl1_is_diagnostic and pl2_is_treatment) or (pl1_is_treatment and pl2_is_diagnostic):
        return True
    return False


def rule2_surgical_vs_yhct(pl1_tenkt: str, pl2_tenkt: str) -> bool:
    """
    Rule 2: Surgical vs non-surgical (YHCT / thủ thuật)
    Returns True if should EXCLUDE (one is surgical, other is YHCT/non-surgical)
    """
    pl1_is_surgical = contains_any_keyword(pl1_tenkt, SURGICAL_KEYWORDS)
    pl1_is_yhct = contains_any_keyword(pl1_tenkt, YHCT_NONSURGICAL_KEYWORDS)
    pl2_is_surgical = contains_any_keyword(pl2_tenkt, SURGICAL_KEYWORDS)
    pl2_is_yhct = contains_any_keyword(pl2_tenkt, YHCT_NONSURGICAL_KEYWORDS)
    
    # Exclude if one is surgical and other is YHCT/non-surgical
    if (pl1_is_surgical and pl2_is_yhct) or (pl1_is_yhct and pl2_is_surgical):
        return True
    return False


def rule3_different_anatomical_systems(pl1_tenkt: str, pl2_tenkt: str) -> bool:
    """
    Rule 3: Different anatomical systems
    Returns True if should EXCLUDE (belong to fundamentally different systems)
    """
    pl1_system = get_anatomical_system(pl1_tenkt)
    pl2_system = get_anatomical_system(pl2_tenkt)
    
    # If both have identifiable systems and they differ, exclude
    if pl1_system and pl2_system and pl1_system != pl2_system:
        return True
    return False


def rule4_wrong_inclusion(pl1_tenkt: str, pl2_tenkt: str) -> bool:
    """
    Rule 4: Wrong inclusion relationship
    Returns True if should EXCLUDE (PL2 is sub-procedure but PL1 is different core)
    """
    pl2_is_sub = contains_any_keyword(pl2_tenkt, SUB_PROCEDURE_INDICATORS)
    
    if pl2_is_sub:
        # Check if PL1 contains core of PL2 (simplified check)
        pl1_lower = str(pl1_tenkt).lower() if pl1_tenkt else ""
        pl2_lower = str(pl2_tenkt).lower() if pl2_tenkt else ""
        
        # Get first few words as "core" (simplified)
        pl2_core_words = pl2_lower.split()[:3]
        pl1_words = set(pl1_lower.split())
        
        # If less than 2 core words overlap, likely wrong inclusion
        overlap = sum(1 for w in pl2_core_words if w in pl1_words)
        if overlap < 2:
            return True
    
    return False


def rule5_yhct_phcn_constraint(pl1_chuong: str, pl2_chuong: str) -> bool:
    """
    Rule 5: YHCT/PHCN domain constraint (PERMANENT RULE)
    Returns True if should EXCLUDE
    If PL2 is YHCT/PHCN, only allow PL1 from YHCT/PHCN/Nhi
    """
    if not pl2_chuong:
        return False
    
    pl2_lower = str(pl2_chuong).lower()
    
    # Check if PL2 belongs to restricted domain
    is_restricted = any(domain in pl2_lower for domain in RESTRICTED_PL2_DOMAINS)
    
    if is_restricted:
        if not pl1_chuong:
            return True  # No PL1 chapter → exclude
        
        pl1_lower = str(pl1_chuong).lower()
        is_allowed = any(allowed in pl1_lower for allowed in ALLOWED_PL1_CHAPTERS)
        
        if not is_allowed:
            return True  # PL1 not in allowed chapters → exclude
    
    return False


def check_all_exclusion_rules(row: pd.Series) -> Tuple[bool, str]:
    """
    Check all exclusion rules for a single row.
    
    Args:
        row: DataFrame row with pl1_tenkt, pl2_tenkt, pl1_chuong, pl2_chuong
        
    Returns:
        Tuple of (should_exclude, reason)
    """
    pl1_tenkt = str(row.get("pl1_tenkt", ""))
    pl2_tenkt = str(row.get("pl2_tenkt", ""))
    pl1_chuong = str(row.get("pl1_chuong", ""))
    pl2_chuong = str(row.get("pl2_chuong", ""))
    
    # Rule 1: Diagnostic vs Treatment
    if rule1_diagnostic_vs_treatment(pl1_tenkt, pl2_tenkt):
        return True, "Rule 1: Diagnostic vs Treatment mismatch"
    
    # Rule 2: Surgical vs YHCT
    if rule2_surgical_vs_yhct(pl1_tenkt, pl2_tenkt):
        return True, "Rule 2: Surgical vs YHCT mismatch"
    
    # Rule 3: Different anatomical systems
    if rule3_different_anatomical_systems(pl1_tenkt, pl2_tenkt):
        return True, "Rule 3: Different anatomical systems"
    
    # Rule 4: Wrong inclusion relationship
    if rule4_wrong_inclusion(pl1_tenkt, pl2_tenkt):
        return True, "Rule 4: Wrong inclusion relationship"
    
    # Rule 5: YHCT/PHCN domain constraint
    if rule5_yhct_phcn_constraint(pl1_chuong, pl2_chuong):
        return True, "Rule 5: YHCT/PHCN domain constraint"
    
    return False, ""


# ============================================================================
# EMBEDDING MODELS (LOCAL + OPENAI + GEMINI)
# ============================================================================

# Supported providers
EMBEDDING_PROVIDERS = ["local", "openai", "gemini"]

_local_embedding_model = None
_openai_client = None
_gemini_configured = False


def get_local_embedding_model():
    """
    Get or initialize the local embedding model.
    Uses sentence-transformers with a multilingual model.
    """
    global _local_embedding_model
    if _local_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # Use multilingual model for Vietnamese support
        _local_embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _local_embedding_model


def get_openai_client(api_key: str):
    """
    Get or initialize the OpenAI client.
    """
    global _openai_client
    if _openai_client is None or api_key:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    return _openai_client


def configure_gemini(api_key: str):
    """
    Configure the Gemini API client.
    """
    global _gemini_configured
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_configured = True
    except ImportError:
        raise ImportError("Please install google-generativeai: pip install google-generativeai")


def compute_local_embedding_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity using local sentence-transformers model.
    
    Returns:
        Similarity score normalized to 0-100 scale
    """
    if not text1 or not text2:
        return 0.0
    
    model = get_local_embedding_model()
    
    # Get embeddings
    embeddings = model.encode([str(text1), str(text2)])
    
    # Compute cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Normalize to 0-100 scale
    return float(similarity * 100)


def compute_openai_embedding_similarity(
    text1: str, 
    text2: str, 
    api_key: str,
    model: str = "text-embedding-3-small"
) -> float:
    """
    Compute semantic similarity using OpenAI embeddings API.
    
    Args:
        text1, text2: Texts to compare
        api_key: OpenAI API key
        model: OpenAI embedding model name
        
    Returns:
        Similarity score normalized to 0-100 scale
    """
    if not text1 or not text2:
        return 0.0
    
    client = get_openai_client(api_key)
    
    # Get embeddings for both texts
    response = client.embeddings.create(
        model=model,
        input=[str(text1), str(text2)]
    )
    
    # Extract embeddings
    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)
    
    # Compute cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    # Normalize to 0-100 scale
    return float(similarity * 100)


def compute_gemini_embedding_similarity(
    text1: str, 
    text2: str, 
    api_key: str,
    model: str = "models/text-embedding-004"
) -> float:
    """
    Compute semantic similarity using Google Gemini embeddings API.
    
    Args:
        text1, text2: Texts to compare
        api_key: Gemini API key
        model: Gemini embedding model name
        
    Returns:
        Similarity score normalized to 0-100 scale
    """
    if not text1 or not text2:
        return 0.0
    
    configure_gemini(api_key)
    
    import google.generativeai as genai
    
    # Get embeddings for both texts
    result1 = genai.embed_content(
        model=model,
        content=str(text1),
        task_type="semantic_similarity"
    )
    result2 = genai.embed_content(
        model=model,
        content=str(text2),
        task_type="semantic_similarity"
    )
    
    # Extract embeddings
    emb1 = np.array(result1['embedding'])
    emb2 = np.array(result2['embedding'])
    
    # Compute cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    # Normalize to 0-100 scale
    return float(similarity * 100)


def compute_embedding_similarity(
    text1: str, 
    text2: str,
    provider: str = "local",
    api_key: str = None,
    openai_model: str = "text-embedding-3-small",
    gemini_model: str = "models/text-embedding-004"
) -> float:
    """
    Compute semantic similarity between two texts.
    
    Args:
        text1, text2: Texts to compare
        provider: "local", "openai", or "gemini"
        api_key: Required for OpenAI/Gemini providers
        openai_model: OpenAI model name
        gemini_model: Gemini model name
        
    Returns:
        Similarity score normalized to 0-100 scale
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return compute_openai_embedding_similarity(text1, text2, api_key, openai_model)
    elif provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required")
        return compute_gemini_embedding_similarity(text1, text2, api_key, gemini_model)
    else:
        return compute_local_embedding_similarity(text1, text2)


# ============================================================================
# REFINEMENT LOGIC
# ============================================================================

def apply_ai_refinement(
    df_expand: pd.DataFrame,
    min_score: float = 50,
    max_score: float = 70,
    alpha: float = 0.7,
    high_threshold: float = 80,
    medium_threshold: float = 65,
    provider: str = "local",
    api_key: str = None,
    openai_model: str = "text-embedding-3-small",
    gemini_model: str = "models/text-embedding-004",
    progress_callback=None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply AI embedding refinement to output_expand.
    
    Args:
        df_expand: output_expand DataFrame from Step 2
        min_score: Minimum original score for eligibility
        max_score: Maximum original score for eligibility
        alpha: Weight for original score (0-1)
        high_threshold: Score threshold for "Cao" classification
        medium_threshold: Score threshold for "Trung bình" classification
        provider: "local", "openai", or "gemini"
        api_key: Required if provider is "openai" or "gemini"
        openai_model: OpenAI model name (default: text-embedding-3-small)
        gemini_model: Gemini model name (default: models/text-embedding-004)
        progress_callback: Optional callback function(processed, total)
        
    Returns:
        Tuple of (updated_df, stats_dict)
    """
    df = df_expand.copy()
    
    # Initialize new columns
    df["embedding_score"] = np.nan
    df["totalscore_before_ai"] = df["totalscore"]
    df["totalscore_after_ai"] = df["totalscore"]
    df["muc_do_after_ai"] = df["muc_do"]
    
    # Statistics
    stats = {
        "total_rows": len(df),
        "eligible": 0,
        "excluded_by_score_range": 0,
        "excluded_by_rules": 0,
        "processed": 0,
        "upgraded": 0,
        "downgraded": 0,
        "unchanged": 0
    }
    
    # Find eligible rows (within score range)
    eligible_mask = (df["totalscore"] >= min_score) & (df["totalscore"] <= max_score)
    eligible_indices = df[eligible_mask].index.tolist()
    
    stats["excluded_by_score_range"] = len(df) - len(eligible_indices)
    
    # Process eligible rows
    total_eligible = len(eligible_indices)
    
    for i, idx in enumerate(eligible_indices):
        # Report progress
        if progress_callback:
            progress_callback(i + 1, total_eligible)
        
        row = df.loc[idx]
        
        # Check exclusion rules
        should_exclude, reason = check_all_exclusion_rules(row)
        
        if should_exclude:
            stats["excluded_by_rules"] += 1
            # Keep original values (already set in initialization)
            continue
        
        stats["eligible"] += 1
        
        # Compute embedding similarity
        pl1_tenkt = str(row.get("pl1_tenkt", ""))
        pl2_tenkt = str(row.get("pl2_tenkt", ""))
        
        embedding_score = compute_embedding_similarity(
            pl1_tenkt, 
            pl2_tenkt,
            provider=provider,
            api_key=api_key,
            openai_model=openai_model,
            gemini_model=gemini_model
        )
        
        # Compute refined score
        original_score = float(row["totalscore"])
        refined_score = alpha * original_score + (1 - alpha) * embedding_score
        
        # Reclassify
        if refined_score >= high_threshold:
            new_muc_do = "Cao"
        elif refined_score >= medium_threshold:
            new_muc_do = "Trung bình"
        else:
            new_muc_do = "Thấp"
        
        # Update row
        df.at[idx, "embedding_score"] = round(embedding_score, 2)
        df.at[idx, "totalscore_after_ai"] = round(refined_score, 2)
        df.at[idx, "muc_do_after_ai"] = new_muc_do
        
        stats["processed"] += 1
        
        # Track classification changes
        old_muc_do = row["muc_do"]
        muc_do_order = {"Thấp": 0, "Trung bình": 1, "Cao": 2, "Không tìm thấy": -1}
        
        old_order = muc_do_order.get(old_muc_do, -1)
        new_order = muc_do_order.get(new_muc_do, -1)
        
        if new_order > old_order:
            stats["upgraded"] += 1
        elif new_order < old_order:
            stats["downgraded"] += 1
        else:
            stats["unchanged"] += 1
    
    return df, stats


def classify_after_ai(score: float, high_threshold: float, medium_threshold: float) -> str:
    """Classify score into compatibility level."""
    if score >= high_threshold:
        return "Cao"
    elif score >= medium_threshold:
        return "Trung bình"
    else:
        return "Thấp"

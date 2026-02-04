"""
Matching module for PL2process Step 2.
Implements semantic matching between PL1 and PL2 procedures.

MATCHING LOGIC:
1. Name similarity: TF-IDF (n-gram 1-2) + cosine similarity on normalized_tenkt
2. Chapter compatibility: Rule-based scoring on normalized_chuong
3. Total score: weighted combination of the two scores

OUTPUT STRUCTURE:
- output_expand: One row per PL1-PL2 match (base table)
- output: Aggregated by PL1 (multiple PL2s joined by ";")
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import io


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    "weight_name": 0.7,
    "weight_chuong": 0.3,
    "high_threshold": 80,
    "medium_threshold": 65,
    "limit_pl1": 0,  # 0 = no limit
    "limit_pl2": 0   # 0 = no limit
}


# ============================================================================
# DOMAIN CONSTRAINT: YHCT / PHCN Restriction Rule
# ============================================================================
# RULE: If PL2 belongs to YHCT or PHCN domain, restrict PL1 candidates to:
#       - YHCT, PHCN, or Nhi chapters ONLY
# This is a HARD CONSTRAINT applied at candidate selection (before scoring).
# ============================================================================

# PL2 domains that trigger the restriction
RESTRICTED_PL2_DOMAINS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang"]

# Allowed PL1 chapters when restriction is active
ALLOWED_PL1_CHAPTERS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang", "nhi"]


def is_restricted_domain(normalized_chuong: str) -> bool:
    """
    Check if a normalized_chuong belongs to a restricted domain (YHCT or PHCN).
    
    Args:
        normalized_chuong: Normalized chapter string (lowercase, no diacritics)
        
    Returns:
        True if the chapter belongs to YHCT or PHCN domain
    """
    if not normalized_chuong:
        return False
    
    chuong_lower = str(normalized_chuong).strip().lower()
    
    for domain in RESTRICTED_PL2_DOMAINS:
        if domain in chuong_lower:
            return True
    
    return False


def is_allowed_pl1_chapter(normalized_chuong: str) -> bool:
    """
    Check if a PL1 chapter is allowed for YHCT/PHCN restricted matching.
    
    Allowed chapters: YHCT, PHCN, Nhi
    
    Args:
        normalized_chuong: Normalized PL1 chapter string
        
    Returns:
        True if the PL1 chapter is allowed for restricted matching
    """
    if not normalized_chuong:
        return False
    
    chuong_lower = str(normalized_chuong).strip().lower()
    
    for allowed in ALLOWED_PL1_CHAPTERS:
        if allowed in chuong_lower:
            return True
    
    return False


def get_valid_pl1_candidates(
    df_pl1: pd.DataFrame,
    pl2_normalized_chuong: str
) -> pd.DataFrame:
    """
    Get valid PL1 candidates for a given PL2 record, applying domain constraints.
    
    DOMAIN CONSTRAINT (HARD RULE):
    - If PL2 is YHCT or PHCN → Only allow PL1 from YHCT, PHCN, or Nhi
    - Otherwise → All PL1 records are valid candidates
    
    Args:
        df_pl1: Full PL1 dataframe
        pl2_normalized_chuong: Normalized chapter of the PL2 record
        
    Returns:
        Filtered PL1 dataframe with valid candidates only
    """
    # Check if PL2 belongs to restricted domain
    if is_restricted_domain(pl2_normalized_chuong):
        # Apply restriction: only YHCT, PHCN, Nhi PL1 chapters allowed
        mask = df_pl1["normalized_chuong"].apply(is_allowed_pl1_chapter)
        return df_pl1[mask].copy()
    else:
        # No restriction: all PL1 candidates are valid
        return df_pl1


def calculate_name_similarity_matrix(pl1_names: List[str], pl2_names: List[str]) -> np.ndarray:
    """
    Calculate TF-IDF cosine similarity matrix between PL1 and PL2 procedure names.
    
    Uses n-gram (1, 2) for better Vietnamese text matching.
    
    Args:
        pl1_names: List of normalized PL1 procedure names
        pl2_names: List of normalized PL2 procedure names
        
    Returns:
        Similarity matrix of shape (len(pl2_names), len(pl1_names))
        Each row is a PL2 procedure, each column is a PL1 procedure
    """
    # Combine all names for TF-IDF fitting
    all_names = list(pl1_names) + list(pl2_names)
    
    # Handle empty strings
    all_names = [name if name and str(name).strip() else "empty" for name in all_names]
    pl1_names_clean = [name if name and str(name).strip() else "empty" for name in pl1_names]
    pl2_names_clean = [name if name and str(name).strip() else "empty" for name in pl2_names]
    
    # Create TF-IDF vectorizer with n-grams (1, 2)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer='word',
        lowercase=True,
        max_features=5000
    )
    
    # Fit on all names
    vectorizer.fit(all_names)
    
    # Transform PL1 and PL2 names
    pl1_vectors = vectorizer.transform(pl1_names_clean)
    pl2_vectors = vectorizer.transform(pl2_names_clean)
    
    # Calculate cosine similarity: rows = PL2, columns = PL1
    similarity_matrix = cosine_similarity(pl2_vectors, pl1_vectors)
    
    # Convert to 0-100 scale
    similarity_matrix = similarity_matrix * 100
    
    return similarity_matrix


def calculate_chapter_compatibility(pl1_chuong: str, pl2_chuong: str) -> float:
    """
    Calculate chapter compatibility score using rule-based matching.
    
    Scoring rules:
    - Exact match → 100
    - Partial overlap (containment) → 80
    - Some common words → 60
    - No match → 30
    """
    # Handle empty values
    if not pl1_chuong or not pl2_chuong:
        return 30
    
    pl1_clean = str(pl1_chuong).strip().lower()
    pl2_clean = str(pl2_chuong).strip().lower()
    
    if not pl1_clean or not pl2_clean:
        return 30
    
    # Exact match
    if pl1_clean == pl2_clean:
        return 100
    
    # Containment check
    if pl1_clean in pl2_clean or pl2_clean in pl1_clean:
        return 80
    
    # Word overlap check
    pl1_words = set(pl1_clean.split())
    pl2_words = set(pl2_clean.split())
    
    if not pl1_words or not pl2_words:
        return 30
    
    common_words = pl1_words.intersection(pl2_words)
    overlap_ratio = len(common_words) / max(len(pl1_words), len(pl2_words))
    
    if overlap_ratio >= 0.5:
        return 80
    elif overlap_ratio > 0:
        return 60
    else:
        return 30


def run_matching(
    df_pl1: pd.DataFrame,
    df_pl2: pd.DataFrame,
    weight_name: float = 0.7,
    high_threshold: float = 80,
    medium_threshold: float = 65,
    limit_pl1: int = 0,
    limit_pl2: int = 0,
    progress_callback=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run semantic matching between PL1 and PL2 procedures.
    
    For each PL2 record, finds the best matching PL1 record.
    
    Args:
        df_pl1: Preprocessed PL1 dataframe
        df_pl2: Preprocessed PL2 dataframe
        weight_name: Weight for name similarity (0-1)
        high_threshold: Score threshold for "Cao" classification
        medium_threshold: Score threshold for "Trung bình" classification
        limit_pl1: Limit PL1 records (0 = no limit)
        limit_pl2: Limit PL2 records (0 = no limit)
        progress_callback: Optional callback function(processed, total) for progress updates
        
    Returns:
        Tuple of (output_expand_df, output_df)
        - output_expand: One row per PL1-PL2 match
        - output: Aggregated by PL1
    """
    weight_chuong = 1.0 - weight_name
    
    # Apply record limits (for testing)
    if limit_pl1 > 0:
        df_pl1 = df_pl1.head(limit_pl1).copy()
    else:
        df_pl1 = df_pl1.copy()
    
    if limit_pl2 > 0:
        df_pl2 = df_pl2.head(limit_pl2).copy()
    else:
        df_pl2 = df_pl2.copy()
    
    # Extract normalized names for similarity calculation
    pl1_names = df_pl1["normalized_tenkt"].fillna("").tolist()
    pl2_names = df_pl2["normalized_tenkt"].fillna("").tolist()
    
    # Calculate name similarity matrix
    name_sim_matrix = calculate_name_similarity_matrix(pl1_names, pl2_names)
    
    # ==========================================================================
    # Build output_expand: one row per PL1-PL2 match
    # ==========================================================================
    expand_results = []
    total_pl2 = len(df_pl2)
    
    for processed_count, (pl2_idx, pl2_row) in enumerate(df_pl2.iterrows(), start=1):
        # Report progress
        if progress_callback:
            progress_callback(processed_count, total_pl2)
        
        pl2_chuong_norm = str(pl2_row.get("normalized_chuong", ""))
        
        # ======================================================================
        # DOMAIN CONSTRAINT: Filter PL1 candidates BEFORE scoring
        # If PL2 is YHCT/PHCN → Only allow PL1 from YHCT, PHCN, Nhi
        # ======================================================================
        valid_pl1_candidates = get_valid_pl1_candidates(df_pl1, pl2_chuong_norm)
        
        best_score = -1
        best_pl1_idx = -1
        best_name_score = 0
        best_chuong_score = 0
        
        # If no valid candidates after domain constraint, skip matching
        if len(valid_pl1_candidates) == 0:
            # No suitable match - record as unmatched
            expand_results.append({
                "pl1_stt": "",
                "pl1_chuong": "",
                "pl1_tenkt": "",
                "pl2_stt": str(pl2_row.get("pl2_stt", "")),
                "pl2_tenkt": str(pl2_row.get("pl2_tenkt", "")),
                "totalscore": 0,
                "muc_do": "Không tìm thấy"
            })
            continue
        
        for pl1_idx, pl1_row in valid_pl1_candidates.iterrows():
            # Get positions in the original matrix
            pl1_pos = df_pl1.index.get_loc(pl1_idx)
            pl2_pos = df_pl2.index.get_loc(pl2_idx)
            
            # Name similarity score
            name_score = name_sim_matrix[pl2_pos, pl1_pos]
            
            # Chapter compatibility score
            pl1_chuong_norm = str(pl1_row.get("normalized_chuong", ""))
            chuong_score = calculate_chapter_compatibility(pl1_chuong_norm, pl2_chuong_norm)
            
            # Total weighted score
            total_score = (weight_name * name_score) + (weight_chuong * chuong_score)
            
            if total_score > best_score:
                best_score = total_score
                best_pl1_idx = pl1_idx
                best_name_score = name_score
                best_chuong_score = chuong_score
        
        # Get best matching PL1 record
        if best_pl1_idx >= 0:
            best_pl1 = df_pl1.loc[best_pl1_idx]
            
            # Classify match quality
            if best_score >= high_threshold:
                muc_do = "Cao"
            elif best_score >= medium_threshold:
                muc_do = "Trung bình"
            else:
                muc_do = "Thấp"
            
            expand_results.append({
                "pl1_stt": str(best_pl1.get("pl1_stt", "")),
                "pl1_chuong": str(best_pl1.get("pl1_chuong", "")),
                "pl1_tenkt": str(best_pl1.get("pl1_tenkt", "")),
                "pl2_stt": str(pl2_row.get("pl2_stt", "")),
                "pl2_tenkt": str(pl2_row.get("pl2_tenkt", "")),
                "totalscore": round(best_score, 2),
                "muc_do": muc_do
            })
    
    df_output_expand = pd.DataFrame(expand_results)
    
    # ==========================================================================
    # Build output: aggregated by PL1
    # ==========================================================================
    output_results = []
    
    # Get all PL1 records that were used
    for pl1_idx, pl1_row in df_pl1.iterrows():
        pl1_stt = str(pl1_row.get("pl1_stt", ""))
        pl1_chuong = str(pl1_row.get("pl1_chuong", ""))
        pl1_tenkt = str(pl1_row.get("pl1_tenkt", ""))
        
        # Find all PL2 matches for this PL1
        matches = df_output_expand[df_output_expand["pl1_stt"] == pl1_stt]
        
        if len(matches) > 0:
            # Aggregate PL2 values with ";" separator (consistent order)
            pl2_stts = ";".join(matches["pl2_stt"].astype(str).tolist())
            pl2_tenkts = ";".join(matches["pl2_tenkt"].astype(str).tolist())
            totalscores = ";".join(matches["totalscore"].astype(str).tolist())
            muc_dos = ";".join(matches["muc_do"].astype(str).tolist())
        else:
            # No matches - leave PL2 columns empty
            pl2_stts = ""
            pl2_tenkts = ""
            totalscores = ""
            muc_dos = ""
        
        output_results.append({
            "pl1_stt": pl1_stt,
            "pl1_chuong": pl1_chuong,
            "pl1_tenkt": pl1_tenkt,
            "pl2_stt": pl2_stts,
            "pl2_tenkt": pl2_tenkts,
            "totalscore": totalscores,
            "muc_do": muc_dos
        })
    
    df_output = pd.DataFrame(output_results)
    
    return df_output_expand, df_output


def classify_results(df_expand: pd.DataFrame, high_threshold: float, medium_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-classify results based on new thresholds and regenerate aggregated output.
    
    Args:
        df_expand: output_expand dataframe (base table)
        high_threshold: Score threshold for "Cao"
        medium_threshold: Score threshold for "Trung bình"
        
    Returns:
        Tuple of (updated_expand, updated_output)
    """
    df = df_expand.copy()
    
    def classify(score):
        if score >= high_threshold:
            return "Cao"
        elif score >= medium_threshold:
            return "Trung bình"
        else:
            return "Thấp"
    
    df["muc_do"] = df["totalscore"].apply(classify)
    
    # Regenerate aggregated output
    pl1_stts = df["pl1_stt"].unique()
    output_results = []
    
    for pl1_stt in pl1_stts:
        matches = df[df["pl1_stt"] == pl1_stt]
        first_match = matches.iloc[0]
        
        output_results.append({
            "pl1_stt": pl1_stt,
            "pl1_chuong": first_match["pl1_chuong"],
            "pl1_tenkt": first_match["pl1_tenkt"],
            "pl2_stt": ";".join(matches["pl2_stt"].astype(str).tolist()),
            "pl2_tenkt": ";".join(matches["pl2_tenkt"].astype(str).tolist()),
            "totalscore": ";".join(matches["totalscore"].astype(str).tolist()),
            "muc_do": ";".join(matches["muc_do"].astype(str).tolist())
        })
    
    df_output = pd.DataFrame(output_results)
    
    return df, df_output


def export_to_excel(
    df_output: pd.DataFrame,
    df_output_expand: pd.DataFrame,
    config: Dict,
    timestamp: datetime = None
) -> bytes:
    """
    Export matching results to Excel with three sheets.
    
    Sheets:
    - output: Aggregated by PL1
    - output_expand: One row per match
    - CONFIG_USED: Configuration parameters
    
    Args:
        df_output: Aggregated output dataframe
        df_output_expand: Expanded output dataframe
        config: Configuration dictionary
        timestamp: Execution timestamp
        
    Returns:
        Excel file as bytes
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: output (aggregated)
        df_output.to_excel(writer, sheet_name='output', index=False)
        
        # Sheet 2: output_expand (one row per match)
        df_output_expand.to_excel(writer, sheet_name='output_expand', index=False)
        
        # Sheet 3: CONFIG_USED
        config_df = pd.DataFrame([{
            "weight_name": config.get("weight_name", 0.7),
            "weight_chuong": config.get("weight_chuong", 0.3),
            "high_threshold": config.get("high_threshold", 80),
            "medium_threshold": config.get("medium_threshold", 65),
            "limit_pl1": config.get("limit_pl1", 0),
            "limit_pl2": config.get("limit_pl2", 0),
            "execution_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }])
        config_df.to_excel(writer, sheet_name='CONFIG_USED', index=False)
    
    output.seek(0)
    return output.getvalue()


def get_summary_stats(df_expand: pd.DataFrame) -> Dict:
    """
    Get summary statistics of matching results from output_expand.
    
    Returns:
        Dictionary with counts and percentages by compatibility level
    """
    total = len(df_expand)
    if total == 0:
        return {"total": 0, "cao": 0, "trung_binh": 0, "thap": 0}
    
    counts = df_expand["muc_do"].value_counts()
    
    return {
        "total": total,
        "cao": counts.get("Cao", 0),
        "trung_binh": counts.get("Trung bình", 0),
        "thap": counts.get("Thấp", 0),
        "cao_pct": round(counts.get("Cao", 0) / total * 100, 1),
        "trung_binh_pct": round(counts.get("Trung bình", 0) / total * 100, 1),
        "thap_pct": round(counts.get("Thấp", 0) / total * 100, 1)
    }


def filter_output_by_muc_do(df_output: pd.DataFrame, selected_levels: List[str]) -> pd.DataFrame:
    """
    Filter aggregated output by compatibility levels.
    
    A row is included if ANY of its muc_do values match the selected levels.
    
    Args:
        df_output: Aggregated output dataframe
        selected_levels: List of selected compatibility levels
        
    Returns:
        Filtered dataframe
    """
    if not selected_levels:
        return pd.DataFrame()
    
    def row_matches(muc_do_str):
        if not muc_do_str:
            return False
        muc_dos = str(muc_do_str).split(";")
        return any(m.strip() in selected_levels for m in muc_dos)
    
    mask = df_output["muc_do"].apply(row_matches)
    return df_output[mask].copy()

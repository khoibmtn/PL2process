"""
PL2process - Unified Medical Procedure Matching Application

Streamlit application with a single unified flow:
1. Upload Excel file
2. Select exactly 2 sheets
3. Validate column structure
4. Run preprocessing
5. Configure and run matching
6. Select MASTER sheet
7. View results
8. Download Excel output

COLUMN NAMING CONVENTION:
Each sheet must have columns named with the sheet name as prefix:
- {sheet_name}_stt: Record ID
- {sheet_name}_chuong: Chapter/specialty
- {sheet_name}_tenkt: Procedure name
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

from preprocessing import (
    preprocess_after_lowercase,
    preprocess_after_numeric,
    preprocess_after_diacritics,
    preprocess_name
)
from file_io import (
    get_sheet_names,
    read_two_sheets,
    get_required_columns,
    export_results_to_excel
)

# AI refinement is optional
AI_REFINEMENT_AVAILABLE = False
try:
    from ai_refinement import (
        apply_ai_refinement,
        DEFAULT_AI_CONFIG
    )
    AI_REFINEMENT_AVAILABLE = True
except ImportError:
    DEFAULT_AI_CONFIG = {
        "enabled": False,
        "min_score": 50,
        "max_score": 70,
        "alpha": 0.7
    }


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    "weight_name": 0.7,
    "weight_chuong": 0.3,
    "high_threshold": 80,
    "medium_threshold": 65
}


# ============================================================================
# DOMAIN CONSTRAINT: YHCT / PHCN Restriction Rule
# ============================================================================
RESTRICTED_DOMAINS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang"]
ALLOWED_CHAPTERS = ["yhct", "y hoc co truyen", "phcn", "phuc hoi chuc nang", "nhi"]


def is_restricted_domain(normalized_chuong: str) -> bool:
    """Check if chapter belongs to restricted domain (YHCT or PHCN)."""
    if not normalized_chuong:
        return False
    chuong_lower = str(normalized_chuong).strip().lower()
    return any(domain in chuong_lower for domain in RESTRICTED_DOMAINS)


def is_allowed_chapter(normalized_chuong: str) -> bool:
    """Check if chapter is allowed for YHCT/PHCN restricted matching."""
    if not normalized_chuong:
        return False
    chuong_lower = str(normalized_chuong).strip().lower()
    return any(allowed in chuong_lower for allowed in ALLOWED_CHAPTERS)


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def apply_preprocessing(
    df: pd.DataFrame, 
    sheet_name: str
) -> pd.DataFrame:
    """
    Apply preprocessing to a dataframe.
    
    CORRECT ORDER:
    1. lowercase → {col}_lowercase
    2. numeric conversion (with diacritics) → {col}_num_normalized
    3. diacritics removed → {col}_no_diacritics
    4. final (stopwords removed) → normalized_{col_type}
    
    Note: All column names use lowercase sheet names.
    """
    result_df = df.copy()
    
    # Use lowercase sheet name for column names
    sheet_lower = sheet_name.lower()
    tenkt_col = f"{sheet_lower}_tenkt"
    chuong_col = f"{sheet_lower}_chuong"
    
    # ========================================================================
    # PROCEDURE NAME PREPROCESSING
    # ========================================================================
    
    # Step 1: Lowercase only
    result_df[f'{tenkt_col}_lowercase'] = result_df[tenkt_col].apply(
        lambda x: preprocess_after_lowercase(str(x)) if pd.notna(x) else ""
    )
    
    # Step 2: After numeric conversion (WITH diacritics - CRITICAL)
    result_df[f'{tenkt_col}_num_normalized'] = result_df[tenkt_col].apply(
        lambda x: preprocess_after_numeric(str(x)) if pd.notna(x) else ""
    )
    
    # Step 3: After diacritics removed
    result_df[f'{tenkt_col}_no_diacritics'] = result_df[tenkt_col].apply(
        lambda x: preprocess_after_diacritics(str(x)) if pd.notna(x) else ""
    )
    
    # Step 4: Final (after stopword removal)
    result_df['normalized_tenkt'] = result_df[tenkt_col].apply(
        lambda x: preprocess_name(str(x)) if pd.notna(x) else ""
    )
    
    # ========================================================================
    # CHAPTER NAME PREPROCESSING
    # ========================================================================
    
    # Step 1: Lowercase only
    result_df[f'{chuong_col}_lowercase'] = result_df[chuong_col].apply(
        lambda x: preprocess_after_lowercase(str(x)) if pd.notna(x) else ""
    )
    
    # Step 2: After numeric conversion
    result_df[f'{chuong_col}_num_normalized'] = result_df[chuong_col].apply(
        lambda x: preprocess_after_numeric(str(x)) if pd.notna(x) else ""
    )
    
    # Step 3: After diacritics removed
    result_df[f'{chuong_col}_no_diacritics'] = result_df[chuong_col].apply(
        lambda x: preprocess_after_diacritics(str(x)) if pd.notna(x) else ""
    )
    
    # Step 4: Final
    result_df['normalized_chuong'] = result_df[chuong_col].apply(
        lambda x: preprocess_name(str(x)) if pd.notna(x) else ""
    )
    
    return result_df


# ============================================================================
# MATCHING FUNCTIONS
# ============================================================================

def calculate_name_similarity_matrix(names1: list, names2: list):
    """Calculate TF-IDF cosine similarity matrix between two name lists."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Combine all names for TF-IDF fitting
    all_names = list(names1) + list(names2)
    
    # Handle empty strings
    all_names = [name if name and str(name).strip() else "empty" for name in all_names]
    names1_clean = [name if name and str(name).strip() else "empty" for name in names1]
    names2_clean = [name if name and str(name).strip() else "empty" for name in names2]
    
    # Create TF-IDF vectorizer with n-grams (1, 2)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer='word',
        lowercase=True,
        max_features=5000
    )
    
    # Fit on all names
    vectorizer.fit(all_names)
    
    # Transform both sets
    vectors1 = vectorizer.transform(names1_clean)
    vectors2 = vectorizer.transform(names2_clean)
    
    # Calculate cosine similarity: rows = names2, columns = names1
    similarity_matrix = cosine_similarity(vectors2, vectors1)
    
    # Convert to 0-100 scale
    return similarity_matrix * 100


def calculate_chapter_compatibility(chuong1: str, chuong2: str) -> float:
    """
    Calculate chapter compatibility score using rule-based matching.
    
    Scoring rules:
    - Exact match → 100
    - Partial overlap (containment) → 80
    - Some common words → 60
    - No match → 30
    """
    if not chuong1 or not chuong2:
        return 30
    
    c1 = str(chuong1).strip().lower()
    c2 = str(chuong2).strip().lower()
    
    if not c1 or not c2:
        return 30
    
    # Exact match
    if c1 == c2:
        return 100
    
    # Containment check
    if c1 in c2 or c2 in c1:
        return 80
    
    # Word overlap check
    words1 = set(c1.split())
    words2 = set(c2.split())
    
    if not words1 or not words2:
        return 30
    
    common_words = words1.intersection(words2)
    overlap_ratio = len(common_words) / max(len(words1), len(words2))
    
    if overlap_ratio >= 0.5:
        return 80
    elif overlap_ratio > 0:
        return 60
    else:
        return 30


def run_matching(
    df_master: pd.DataFrame,
    df_match: pd.DataFrame,
    master_sheet: str,
    match_sheet: str,
    weight_name: float = 0.7,
    high_threshold: float = 80,
    medium_threshold: float = 65,
    progress_callback=None
):
    """
    Run semantic matching between MASTER and MATCH sheets.
    
    For each MASTER record, finds matching records from MATCH sheet.
    
    Returns:
        Tuple of (output_expand_df, output_df)
    """
    weight_chuong = 1.0 - weight_name
    
    # Column names (use lowercase sheet names)
    master_lower = master_sheet.lower()
    match_lower = match_sheet.lower()
    master_stt = f"{master_lower}_stt"
    master_chuong = f"{master_lower}_chuong"
    master_tenkt = f"{master_lower}_tenkt"
    match_stt = f"{match_lower}_stt"
    match_tenkt = f"{match_lower}_tenkt"
    
    # Extract normalized names for similarity calculation
    master_names = df_master["normalized_tenkt"].fillna("").tolist()
    match_names = df_match["normalized_tenkt"].fillna("").tolist()
    
    # Calculate name similarity matrix (rows = match, columns = master)
    name_sim_matrix = calculate_name_similarity_matrix(master_names, match_names)
    
    # Build output_expand: one row per MASTER-MATCH pair
    expand_results = []
    total_master = len(df_master)
    
    for processed_count, (master_idx, master_row) in enumerate(df_master.iterrows(), start=1):
        if progress_callback:
            progress_callback(processed_count, total_master)
        
        master_chuong_norm = str(master_row.get("normalized_chuong", ""))
        master_stt_val = str(master_row.get(master_stt, ""))
        master_chuong_val = str(master_row.get(master_chuong, ""))
        master_tenkt_val = str(master_row.get(master_tenkt, ""))
        
        # Get valid MATCH candidates (apply domain constraints)
        if is_restricted_domain(master_chuong_norm):
            valid_mask = df_match["normalized_chuong"].apply(is_allowed_chapter)
            valid_match = df_match[valid_mask].copy()
        else:
            valid_match = df_match
        
        # If no valid candidates, add row with empty match
        if len(valid_match) == 0:
            expand_results.append({
                master_stt: master_stt_val,
                master_chuong: master_chuong_val,
                master_tenkt: master_tenkt_val,
                match_stt: "",
                match_tenkt: "",
                "totalscore": 0,
                "muc_do": "Không tìm thấy"
            })
            continue
        
        # Find best matching record
        best_score = -1
        best_match_idx = -1
        
        for match_idx, match_row in valid_match.iterrows():
            master_pos = df_master.index.get_loc(master_idx)
            match_pos = df_match.index.get_loc(match_idx)
            
            # Name similarity score
            name_score = name_sim_matrix[match_pos, master_pos]
            
            # Chapter compatibility score
            match_chuong_norm = str(match_row.get("normalized_chuong", ""))
            chuong_score = calculate_chapter_compatibility(master_chuong_norm, match_chuong_norm)
            
            # Total weighted score
            total_score = (weight_name * name_score) + (weight_chuong * chuong_score)
            
            if total_score > best_score:
                best_score = total_score
                best_match_idx = match_idx
        
        # Add result
        if best_match_idx >= 0:
            best_match = df_match.loc[best_match_idx]
            
            # Classify match quality
            if best_score >= high_threshold:
                muc_do = "Cao"
            elif best_score >= medium_threshold:
                muc_do = "Trung bình"
            else:
                muc_do = "Thấp"
            
            expand_results.append({
                master_stt: master_stt_val,
                master_chuong: master_chuong_val,
                master_tenkt: master_tenkt_val,
                match_stt: str(best_match.get(match_stt, "")),
                match_tenkt: str(best_match.get(match_tenkt, "")),
                "totalscore": round(best_score, 2),
                "muc_do": muc_do
            })
        else:
            expand_results.append({
                master_stt: master_stt_val,
                master_chuong: master_chuong_val,
                master_tenkt: master_tenkt_val,
                match_stt: "",
                match_tenkt: "",
                "totalscore": 0,
                "muc_do": "Không tìm thấy"
            })
    
    df_output_expand = pd.DataFrame(expand_results)
    
    # Build output: aggregated by MASTER
    output_results = []
    
    for master_idx, master_row in df_master.iterrows():
        master_stt_val = str(master_row.get(master_stt, ""))
        master_chuong_val = str(master_row.get(master_chuong, ""))
        master_tenkt_val = str(master_row.get(master_tenkt, ""))
        
        # Find all matches for this MASTER record
        matches = df_output_expand[df_output_expand[master_stt] == master_stt_val]
        
        if len(matches) > 0:
            match_stts = ";".join(matches[match_stt].astype(str).tolist())
            match_tenkts = ";".join(matches[match_tenkt].astype(str).tolist())
            totalscores = ";".join(matches["totalscore"].astype(str).tolist())
            muc_dos = ";".join(matches["muc_do"].astype(str).tolist())
        else:
            match_stts = ""
            match_tenkts = ""
            totalscores = ""
            muc_dos = ""
        
        output_results.append({
            master_stt: master_stt_val,
            master_chuong: master_chuong_val,
            master_tenkt: master_tenkt_val,
            match_stt: match_stts,
            match_tenkt: match_tenkts,
            "totalscore": totalscores,
            "muc_do": muc_dos
        })
    
    df_output = pd.DataFrame(output_results)
    
    return df_output_expand, df_output


def get_summary_stats(df_expand: pd.DataFrame) -> dict:
    """Get summary statistics of matching results."""
    total = len(df_expand)
    if total == 0:
        return {"total": 0, "cao": 0, "trung_binh": 0, "thap": 0}
    
    counts = df_expand["muc_do"].value_counts()
    
    return {
        "total": total,
        "cao": counts.get("Cao", 0),
        "trung_binh": counts.get("Trung bình", 0),
        "thap": counts.get("Thấp", 0),
        "khong_tim_thay": counts.get("Không tìm thấy", 0),
        "cao_pct": round(counts.get("Cao", 0) / total * 100, 1),
        "trung_binh_pct": round(counts.get("Trung bình", 0) / total * 100, 1),
        "thap_pct": round(counts.get("Thấp", 0) / total * 100, 1)
    }


# ============================================================================
# UI FUNCTIONS
# ============================================================================

def display_config_sidebar():
    """Display configuration controls in sidebar."""
    st.sidebar.header("⚙️ Matching Configuration")
    
    # Weight Settings
    st.sidebar.markdown("### 📊 Weight Settings")
    
    weight_name = st.sidebar.slider(
        "Name similarity weight",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIG["weight_name"],
        step=0.05,
        help="Weight for procedure name similarity."
    )
    weight_chuong = 1.0 - weight_name
    
    st.sidebar.info(f"📌 Chapter weight: **{weight_chuong:.2f}**")
    
    st.sidebar.markdown("---")
    
    # Threshold Settings
    st.sidebar.markdown("### 🎯 Threshold Settings")
    
    high_threshold = st.sidebar.number_input(
        "High compatibility (≥)",
        min_value=0,
        max_value=100,
        value=DEFAULT_CONFIG["high_threshold"],
        step=5
    )
    
    medium_threshold = st.sidebar.number_input(
        "Medium compatibility (≥)",
        min_value=0,
        max_value=int(high_threshold),
        value=min(DEFAULT_CONFIG["medium_threshold"], int(high_threshold)),
        step=5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Classification Legend")
    st.sidebar.markdown(f"""
    - 🟢 **Cao**: score ≥ {high_threshold}
    - 🟡 **Trung bình**: score ≥ {medium_threshold}
    - 🔴 **Thấp**: score < {medium_threshold}
    """)
    
    st.sidebar.markdown("---")
    
    # AI Refinement (optional)
    st.sidebar.markdown("### 🤖 AI Embedding Refinement")
    
    if not AI_REFINEMENT_AVAILABLE:
        st.sidebar.warning("⚠️ AI not available. Install: `pip install sentence-transformers`")
        ai_enabled = False
        ai_config = {}
    else:
        ai_enabled = st.sidebar.checkbox(
            "Apply AI embedding refinement",
            value=DEFAULT_AI_CONFIG["enabled"]
        )
        
        if ai_enabled:
            ai_provider = st.sidebar.radio(
                "Embedding Provider",
                options=["local", "openai", "gemini"],
                format_func=lambda x: {
                    "local": "🖥️ Local",
                    "openai": "☁️ OpenAI",
                    "gemini": "✨ Gemini"
                }.get(x, x)
            )
            
            ai_api_key = None
            if ai_provider in ["openai", "gemini"]:
                ai_api_key = st.sidebar.text_input(
                    f"{ai_provider.title()} API Key",
                    type="password"
                )
            
            ai_min_score = st.sidebar.number_input(
                "Min original score", min_value=0, max_value=100,
                value=DEFAULT_AI_CONFIG["min_score"], step=5
            )
            
            ai_max_score = st.sidebar.number_input(
                "Max original score", min_value=0, max_value=100,
                value=DEFAULT_AI_CONFIG["max_score"], step=5
            )
            
            ai_alpha = st.sidebar.slider(
                "Refinement strength (alpha)",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_AI_CONFIG["alpha"], step=0.05
            )
            
            ai_config = {
                "provider": ai_provider,
                "api_key": ai_api_key,
                "min_score": ai_min_score,
                "max_score": ai_max_score,
                "alpha": ai_alpha
            }
        else:
            ai_config = {}
    
    return {
        "weight_name": weight_name,
        "weight_chuong": weight_chuong,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "ai_enabled": ai_enabled,
        "ai_config": ai_config
    }


def display_summary_stats(stats: dict):
    """Display summary statistics with visual indicators."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total", stats["total"])
    with col2:
        st.metric("🟢 Cao", f"{stats['cao']} ({stats.get('cao_pct', 0)}%)")
    with col3:
        st.metric("🟡 Trung bình", f"{stats['trung_binh']} ({stats.get('trung_binh_pct', 0)}%)")
    with col4:
        st.metric("🔴 Thấp", f"{stats['thap']} ({stats.get('thap_pct', 0)}%)")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="PL2process - Unified Matching",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 PL2process - Medical Procedure Matching")
    st.markdown("""
    **Unified workflow:** Upload → Select sheets → Preprocess → Match → View → Download
    """)
    
    # Sidebar configuration
    config = display_config_sidebar()
    
    # Initialize session state
    if 'sheet_names' not in st.session_state:
        st.session_state.sheet_names = []
    if 'df_sheet1' not in st.session_state:
        st.session_state.df_sheet1 = None
    if 'df_sheet2' not in st.session_state:
        st.session_state.df_sheet2 = None
    if 'preprocessed_sheet1' not in st.session_state:
        st.session_state.preprocessed_sheet1 = None
    if 'preprocessed_sheet2' not in st.session_state:
        st.session_state.preprocessed_sheet2 = None
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'output_expand' not in st.session_state:
        st.session_state.output_expand = None
    if 'selected_sheets' not in st.session_state:
        st.session_state.selected_sheets = []
    if 'master_sheet' not in st.session_state:
        st.session_state.master_sheet = None
    
    st.divider()
    
    # ========================================================================
    # STEP 1: Upload Excel file
    # ========================================================================
    st.header("📁 Step 1: Upload Excel File")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx)",
        type=['xlsx'],
        help="File must contain sheets with columns: {sheet_name}_stt, {sheet_name}_chuong, {sheet_name}_tenkt"
    )
    
    if uploaded_file is not None:
        # Get sheet names
        sheet_names, error = get_sheet_names(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
            st.stop()
        
        st.session_state.sheet_names = sheet_names
        st.success(f"✅ File loaded: {len(sheet_names)} sheets detected")
        
        # ====================================================================
        # STEP 2: Select 2 sheets
        # ====================================================================
        st.divider()
        st.header("📋 Step 2: Select Two Sheets")
        
        selected_sheets = st.multiselect(
            "Select exactly 2 sheets for processing:",
            options=sheet_names,
            max_selections=2,
            help="Choose two sheets to compare and match"
        )
        
        if len(selected_sheets) != 2:
            st.warning("⚠️ Please select exactly 2 sheets to continue.")
            st.stop()
        
        st.session_state.selected_sheets = selected_sheets
        sheet1_name, sheet2_name = selected_sheets
        
        # Show required columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sheet 1: {sheet1_name}**")
            st.markdown(f"Required columns: `{get_required_columns(sheet1_name)}`")
        with col2:
            st.markdown(f"**Sheet 2: {sheet2_name}**")
            st.markdown(f"Required columns: `{get_required_columns(sheet2_name)}`")
        
        # Read and validate sheets
        df_sheet1, df_sheet2, error = read_two_sheets(
            uploaded_file, sheet1_name, sheet2_name
        )
        
        if error:
            st.error(f"❌ {error}")
            st.stop()
        
        st.session_state.df_sheet1 = df_sheet1
        st.session_state.df_sheet2 = df_sheet2
        
        st.success(f"✅ Sheets validated: {sheet1_name} ({len(df_sheet1)} rows), {sheet2_name} ({len(df_sheet2)} rows)")
        
        # ====================================================================
        # STEP 3: Preprocessing
        # ====================================================================
        st.divider()
        st.header("⚙️ Step 3: Run Preprocessing")
        
        if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocess sheet 1
            status_text.text(f"Processing {sheet1_name}...")
            progress_bar.progress(25)
            preprocessed_sheet1 = apply_preprocessing(df_sheet1, sheet1_name)
            st.session_state.preprocessed_sheet1 = preprocessed_sheet1
            
            # Preprocess sheet 2
            status_text.text(f"Processing {sheet2_name}...")
            progress_bar.progress(75)
            preprocessed_sheet2 = apply_preprocessing(df_sheet2, sheet2_name)
            st.session_state.preprocessed_sheet2 = preprocessed_sheet2
            
            progress_bar.progress(100)
            status_text.text("✅ Preprocessing completed!")
            
            st.success("🎉 **Preprocessing completed successfully!**")
        
        # ====================================================================
        # STEP 4: Select MASTER sheet
        # ====================================================================
        if st.session_state.preprocessed_sheet1 is not None:
            st.divider()
            st.header("🎯 Step 4: Select MASTER Sheet")
            
            st.markdown("""
            > **MASTER sheet**: All records from this sheet will appear in the output.  
            > **MATCH sheet**: Used to find corresponding techniques.
            """)
            
            master_sheet = st.radio(
                "Select the MASTER sheet (giữ nguyên danh sách):",
                options=selected_sheets,
                horizontal=True
            )
            
            st.session_state.master_sheet = master_sheet
            match_sheet = sheet2_name if master_sheet == sheet1_name else sheet1_name
            
            st.info(f"📌 **MASTER**: {master_sheet} | **MATCH**: {match_sheet}")
            
            # ================================================================
            # STEP 5: Run Matching
            # ================================================================
            st.divider()
            st.header("🔗 Step 5: Run Matching")
            
            st.markdown(f"""
            **Configuration:**
            - Name weight: **{config['weight_name']:.2f}** | Chapter weight: **{config['weight_chuong']:.2f}**
            - Thresholds: High ≥ **{config['high_threshold']}** | Medium ≥ **{config['medium_threshold']}**
            """)
            
            if st.button("🚀 Run Matching", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Get the correct dataframes based on MASTER selection
                if master_sheet == sheet1_name:
                    df_master = st.session_state.preprocessed_sheet1
                    df_match = st.session_state.preprocessed_sheet2
                else:
                    df_master = st.session_state.preprocessed_sheet2
                    df_match = st.session_state.preprocessed_sheet1
                
                def update_progress(processed, total):
                    pct = processed / total
                    progress_bar.progress(pct)
                    progress_text.markdown(f"**Processing:** {processed} / {total} ({pct*100:.1f}%)")
                
                output_expand, output = run_matching(
                    df_master=df_master,
                    df_match=df_match,
                    master_sheet=master_sheet,
                    match_sheet=match_sheet,
                    weight_name=config['weight_name'],
                    high_threshold=config['high_threshold'],
                    medium_threshold=config['medium_threshold'],
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                progress_text.empty()
                
                st.session_state.output = output
                st.session_state.output_expand = output_expand
                st.session_state.match_sheet = match_sheet
                
                st.success("✅ Matching completed!")
        
        # ====================================================================
        # STEP 6: View Results
        # ====================================================================
        if st.session_state.output is not None:
            st.divider()
            st.header("📊 Step 6: View Results")
            
            # Summary stats
            stats = get_summary_stats(st.session_state.output_expand)
            display_summary_stats(stats)
            
            st.divider()
            
            # Filter controls
            selected_levels = st.multiselect(
                "Show compatibility levels:",
                options=["Cao", "Trung bình", "Thấp", "Không tìm thấy"],
                default=["Cao", "Trung bình", "Thấp"]
            )
            
            # Tabs for different views
            tab1, tab2 = st.tabs(["📋 output (Aggregated)", "🔍 output_expand (Detailed)"])
            
            with tab1:
                st.markdown("**One row per MASTER record. Multiple matches joined with `;`**")
                df_filtered = st.session_state.output.copy()
                if selected_levels:
                    def row_matches(muc_do_str):
                        if not muc_do_str:
                            return False
                        muc_dos = str(muc_do_str).split(";")
                        return any(m.strip() in selected_levels for m in muc_dos)
                    df_filtered = df_filtered[df_filtered["muc_do"].apply(row_matches)]
                
                st.dataframe(df_filtered, use_container_width=True, height=500)
                st.caption(f"Showing {len(df_filtered)} of {len(st.session_state.output)} records")
            
            with tab2:
                st.markdown("**One row per MASTER-MATCH pair.**")
                df_filtered = st.session_state.output_expand[
                    st.session_state.output_expand["muc_do"].isin(selected_levels)
                ].copy()
                st.dataframe(df_filtered, use_container_width=True, height=500)
                st.caption(f"Showing {len(df_filtered)} of {len(st.session_state.output_expand)} records")
            
            # ================================================================
            # STEP 7: Download
            # ================================================================
            st.divider()
            st.header("💾 Step 7: Download Results")
            
            export_config = {
                "master_sheet": st.session_state.master_sheet,
                "match_sheet": st.session_state.match_sheet,
                "weight_name": config['weight_name'],
                "weight_chuong": config['weight_chuong'],
                "high_threshold": config['high_threshold'],
                "medium_threshold": config['medium_threshold'],
                "ai_enabled": config.get('ai_enabled', False),
                "ai_provider": config.get('ai_config', {}).get('provider', 'local')
            }
            
            export_bytes = export_results_to_excel(
                st.session_state.output,
                st.session_state.output_expand,
                export_config,
                datetime.now()
            )
            
            st.download_button(
                label="📥 Download match_results.xlsx",
                data=export_bytes,
                file_name="match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.caption("Excel contains: **output** + **output_expand** + **CONFIG_USED**")


if __name__ == "__main__":
    main()

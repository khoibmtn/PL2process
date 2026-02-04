"""
PL2process - Medical Procedure Data Preprocessing Application

Streamlit UI for uploading Excel files and running preprocessing pipeline.

CORRECT PREPROCESSING ORDER:
1. Lowercase
2. Convert numeric words (WITH diacritics still present)
3. Remove Vietnamese diacritics
4. Remove stopwords
5. Final whitespace normalization
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import time

from preprocessing import (
    preprocess_after_lowercase,
    preprocess_after_numeric,
    preprocess_after_diacritics,
    preprocess_name
)
from file_io import (
    read_excel_sheets, 
    save_preprocessed, 
    get_alias_descriptions,
    PL1_SCHEMA,
    PL2_SCHEMA,
    PL1_INPUT_COLUMNS,
    PL2_INPUT_COLUMNS
)


def apply_preprocessing_with_stages(df: pd.DataFrame, procedure_col: str, chapter_col: str,
                                     sheet_name: str) -> pd.DataFrame:
    """
    Apply preprocessing to a dataframe with explicit intermediate columns.
    
    CORRECT ORDER:
    1. lowercase → {col}_lowercase
    2. numeric conversion (with diacritics) → {col}_num_normalized
    3. diacritics removed → {col}_no_diacritics
    4. final (stopwords removed) → normalized_{col_type}
    """
    result_df = df.copy()
    
    # ========================================================================
    # PROCEDURE NAME PREPROCESSING
    # ========================================================================
    
    # Step 1: Lowercase only
    result_df[f'{procedure_col}_lowercase'] = result_df[procedure_col].apply(
        lambda x: preprocess_after_lowercase(str(x)) if pd.notna(x) else ""
    )
    
    # Step 2: After numeric conversion (WITH diacritics - CRITICAL)
    result_df[f'{procedure_col}_num_normalized'] = result_df[procedure_col].apply(
        lambda x: preprocess_after_numeric(str(x)) if pd.notna(x) else ""
    )
    
    # Step 3: After diacritics removed
    result_df[f'{procedure_col}_no_diacritics'] = result_df[procedure_col].apply(
        lambda x: preprocess_after_diacritics(str(x)) if pd.notna(x) else ""
    )
    
    # Step 4: Final (after stopword removal)
    result_df['normalized_tenkt'] = result_df[procedure_col].apply(
        lambda x: preprocess_name(str(x)) if pd.notna(x) else ""
    )
    
    # ========================================================================
    # CHAPTER NAME PREPROCESSING
    # ========================================================================
    
    # Step 1: Lowercase only
    result_df[f'{chapter_col}_lowercase'] = result_df[chapter_col].apply(
        lambda x: preprocess_after_lowercase(str(x)) if pd.notna(x) else ""
    )
    
    # Step 2: After numeric conversion
    result_df[f'{chapter_col}_num_normalized'] = result_df[chapter_col].apply(
        lambda x: preprocess_after_numeric(str(x)) if pd.notna(x) else ""
    )
    
    # Step 3: After diacritics removed
    result_df[f'{chapter_col}_no_diacritics'] = result_df[chapter_col].apply(
        lambda x: preprocess_after_diacritics(str(x)) if pd.notna(x) else ""
    )
    
    # Step 4: Final
    result_df['normalized_chuong'] = result_df[chapter_col].apply(
        lambda x: preprocess_name(str(x)) if pd.notna(x) else ""
    )
    
    return result_df


def display_schema_info():
    """Display the internal column schema for transparency."""
    alias_desc = get_alias_descriptions()
    
    with st.expander("📐 Column Schema Reference (Position-Based)", expanded=False):
        st.markdown("""
        > **Note:** Column identification uses POSITION, not header names.  
        > Headers in the Excel file are ignored due to formatting issues.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PL1 Sheet (Columns A-E)**")
            for alias, desc in alias_desc["PL1"].items():
                st.markdown(f"- `{alias}`: {desc}")
        
        with col2:
            st.markdown("**PL2 Sheet (Columns A-D)**")
            for alias, desc in alias_desc["PL2"].items():
                if "IGNORED" in desc:
                    st.markdown(f"- ~~`{alias}`~~: {desc}")
                else:
                    st.markdown(f"- `{alias}`: {desc}")


def main():
    # Page configuration
    st.set_page_config(
        page_title="PL2process - Preprocessing",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st.title("🏥 PL2process - Medical Procedure Preprocessing")
    st.markdown("""
    **Purpose:** Preprocess medical procedure data for matching.  
    **Stage:** PREPROCESSING ONLY - This tool stops after preprocessing for debugging.
    """)
    
    # Display schema reference
    display_schema_info()
    
    st.divider()
    
    # Initialize session state
    if 'preprocessed_pl1' not in st.session_state:
        st.session_state.preprocessed_pl1 = None
    if 'preprocessed_pl2' not in st.session_state:
        st.session_state.preprocessed_pl2 = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'output_file_path' not in st.session_state:
        st.session_state.output_file_path = None
    
    # File Upload Section
    st.header("📁 Step 1: Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx) containing PL1 and PL2 sheets",
        type=['xlsx'],
        help="File must contain sheets named 'PL1' and 'PL2'. Columns are identified by POSITION, not headers."
    )
    
    # Processing Section
    if uploaded_file is not None:
        st.success(f"✅ File loaded: {uploaded_file.name}")
        
        st.header("⚙️ Step 2: Run Preprocessing")
        
        if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
            # Reset state
            st.session_state.preprocessed_pl1 = None
            st.session_state.preprocessed_pl2 = None
            st.session_state.processing_complete = False
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ============================================================
                # Stage 1: File loaded
                # ============================================================
                status_text.text("📂 Stage 1/6: Reading Excel sheets (position-based)...")
                progress_bar.progress(10)
                time.sleep(0.3)
                
                # Read Excel file with position-based column mapping
                df_pl1, df_pl2, error = read_excel_sheets(uploaded_file)
                
                if error:
                    st.error(f"❌ Error: {error}")
                    st.stop()
                
                progress_bar.progress(15)
                status_text.text(f"✓ File loaded: PL1 ({len(df_pl1)} rows), PL2 ({len(df_pl2)} rows)")
                time.sleep(0.3)
                
                # ============================================================
                # Stage 2: Lowercase
                # ============================================================
                status_text.text("📝 Stage 2/6: Converting to lowercase...")
                progress_bar.progress(25)
                time.sleep(0.2)
                
                # ============================================================
                # Stage 3: Numeric conversion (BEFORE Unicode normalization!)
                # ============================================================
                status_text.text("🔢 Stage 3/6: Converting numeric words (before Unicode normalization)...")
                progress_bar.progress(40)
                time.sleep(0.2)
                
                # ============================================================
                # Stage 4: Remove diacritics
                # ============================================================
                status_text.text("🔤 Stage 4/6: Removing Vietnamese diacritics...")
                progress_bar.progress(55)
                time.sleep(0.2)
                
                # Process PL1 with all stages
                preprocessed_pl1 = apply_preprocessing_with_stages(
                    df_pl1, 
                    procedure_col="pl1_tenkt",
                    chapter_col="pl1_chuong",
                    sheet_name="pl1"
                )
                progress_bar.progress(65)
                
                # Process PL2 with all stages
                preprocessed_pl2 = apply_preprocessing_with_stages(
                    df_pl2, 
                    procedure_col="pl2_tenkt",
                    chapter_col="pl2_chuong",
                    sheet_name="pl2"
                )
                progress_bar.progress(75)
                
                # ============================================================
                # Stage 5: Remove stopwords
                # ============================================================
                status_text.text("🧹 Stage 5/6: Removing stopwords...")
                progress_bar.progress(85)
                time.sleep(0.2)
                
                # ============================================================
                # Stage 6: Save output
                # ============================================================
                status_text.text("💾 Stage 6/6: Saving preprocessed output...")
                
                # Save to temp directory
                output_dir = tempfile.gettempdir()
                output_path = os.path.join(output_dir, "preprocessed_output.xlsx")
                
                save_error = save_preprocessed(preprocessed_pl1, preprocessed_pl2, output_path)
                if save_error:
                    st.warning(f"⚠️ Could not save file: {save_error}")
                else:
                    st.session_state.output_file_path = output_path
                
                progress_bar.progress(100)
                status_text.text("✅ Preprocessing completed!")
                
                # Store in session state
                st.session_state.preprocessed_pl1 = preprocessed_pl1
                st.session_state.preprocessed_pl2 = preprocessed_pl2
                st.session_state.processing_complete = True
                
                st.success("🎉 **Preprocessing completed successfully!**")
    
    # Results Section
    if st.session_state.processing_complete:
        st.header("📊 Step 3: Review Preprocessed Data")
        
        st.markdown("""
        > ⚠️ **STOP HERE FOR DEBUGGING**  
        > Review **numeric word conversion** column - numbers should be converted  
        > BEFORE diacritics are removed (e.g., "sáu" → "6", NOT "sau" → "6")
        """)
        
        # Download button
        if st.session_state.output_file_path and os.path.exists(st.session_state.output_file_path):
            with open(st.session_state.output_file_path, 'rb') as f:
                st.download_button(
                    label="📥 Download preprocessed_output.xlsx",
                    data=f,
                    file_name="preprocessed_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        st.divider()
        
        # ====================================================================
        # Display PL1
        # ====================================================================
        st.subheader("📋 Preprocessed PL1")
        st.info(f"Total rows: {len(st.session_state.preprocessed_pl1)}")
        
        # Columns showing each step
        display_cols_pl1 = [
            "pl1_stt",
            "pl1_tenkt",                    # Original
            "pl1_tenkt_lowercase",           # After Step 1
            "pl1_tenkt_num_normalized",      # After Step 2 (numbers converted WITH diacritics)
            "pl1_tenkt_no_diacritics",       # After Step 3
            "normalized_tenkt"               # Final
        ]
        available_cols_pl1 = [c for c in display_cols_pl1 if c in st.session_state.preprocessed_pl1.columns]
        
        # Column legend
        with st.expander("🏷️ PL1 Column Legend (Processing Order)"):
            st.markdown("""
            | Column | Step | Description |
            |--------|------|-------------|
            | `pl1_tenkt` | Original | Procedure name (unchanged) |
            | `pl1_tenkt_lowercase` | Step 1 | After lowercase |
            | `pl1_tenkt_num_normalized` | Step 2 | **After numeric conversion (WITH diacritics)** ← Check this! |
            | `pl1_tenkt_no_diacritics` | Step 3 | After diacritics removed |
            | `normalized_tenkt` | Final | After stopwords removed |
            """)
        
        st.dataframe(
            st.session_state.preprocessed_pl1[available_cols_pl1],
            use_container_width=True,
            height=400
        )
        
        st.divider()
        
        # ====================================================================
        # Display PL2
        # ====================================================================
        st.subheader("📋 Preprocessed PL2")
        st.info(f"Total rows: {len(st.session_state.preprocessed_pl2)}")
        
        display_cols_pl2 = [
            "pl2_stt",
            "pl2_tenkt",                    # Original
            "pl2_tenkt_lowercase",           # After Step 1
            "pl2_tenkt_num_normalized",      # After Step 2
            "pl2_tenkt_no_diacritics",       # After Step 3
            "normalized_tenkt"               # Final
        ]
        available_cols_pl2 = [c for c in display_cols_pl2 if c in st.session_state.preprocessed_pl2.columns]
        
        # Column legend
        with st.expander("🏷️ PL2 Column Legend (Processing Order)"):
            st.markdown("""
            | Column | Step | Description |
            |--------|------|-------------|
            | `pl2_tenkt` | Original | Procedure name (unchanged) |
            | `pl2_tenkt_lowercase` | Step 1 | After lowercase |
            | `pl2_tenkt_num_normalized` | Step 2 | **After numeric conversion (WITH diacritics)** ← Check this! |
            | `pl2_tenkt_no_diacritics` | Step 3 | After diacritics removed |
            | `normalized_tenkt` | Final | After stopwords removed |
            """)
        
        st.dataframe(
            st.session_state.preprocessed_pl2[available_cols_pl2],
            use_container_width=True,
            height=400
        )
        
        st.divider()
        
        # ====================================================================
        # Debug info
        # ====================================================================
        with st.expander("🔍 Debug Information"):
            st.write("**PL1 All Columns:**", list(st.session_state.preprocessed_pl1.columns))
            st.write("**PL2 All Columns:**", list(st.session_state.preprocessed_pl2.columns))
            
            st.markdown("### Numeric Conversion Verification")
            st.markdown("""
            **Correct behavior:**
            - `"sáu lần"` → `"6 lần"` (✓ "sáu" is six)
            - `"sau đó"` → `"sau đó"` (✓ "sau" is after, NOT converted)
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**PL1 Sample (first 5):**")
                sample_cols = ["pl1_tenkt", "pl1_tenkt_num_normalized"]
                avail = [c for c in sample_cols if c in st.session_state.preprocessed_pl1.columns]
                st.dataframe(st.session_state.preprocessed_pl1[avail].head())
            with col2:
                st.write("**PL2 Sample (first 5):**")
                sample_cols = ["pl2_tenkt", "pl2_tenkt_num_normalized"]
                avail = [c for c in sample_cols if c in st.session_state.preprocessed_pl2.columns]
                st.dataframe(st.session_state.preprocessed_pl2[avail].head())


if __name__ == "__main__":
    main()

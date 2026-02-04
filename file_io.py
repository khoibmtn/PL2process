"""
File I/O module for PL2process.
Handles Excel reading and writing using POSITION-BASED column identification.

IMPORTANT: Column headers are unreliable due to line breaks and formatting.
All column identification is done by POSITION (0-indexed), not by header names.
"""

import os
from typing import Tuple, Optional, Dict
import pandas as pd

# ============================================================================
# COLUMN SCHEMA - POSITION-BASED (0-indexed)
# ============================================================================

# PL1 Sheet Schema (Columns A-E, positions 0-4)
# - Columns A-C (0-2) are INPUT
# - Columns D-E (3-4) are OUTPUT (may be empty initially)
PL1_SCHEMA = {
    "pl1_stt": 0,       # Column A: PL1 procedure ID
    "pl1_chuong": 1,    # Column B: PL1 specialty / medical field
    "pl1_tenkt": 2,     # Column C: PL1 procedure name
    "pl2_stt": 3,       # Column D: aggregated PL2 IDs (OUTPUT)
    "pl2_tenkt": 4,     # Column E: aggregated PL2 procedure names (OUTPUT)
}

PL1_INPUT_COLUMNS = ["pl1_stt", "pl1_chuong", "pl1_tenkt"]  # Required for input
PL1_MIN_COLUMNS = 3  # Minimum columns required (A, B, C)

# PL2 Sheet Schema (Columns A-D, positions 0-3)
# - Columns A, C, D (0, 2, 3) are used for preprocessing
# - Column B (1) is pl2_stt2 and must be IGNORED
PL2_SCHEMA = {
    "pl2_stt": 0,       # Column A: PL2 procedure ID
    "pl2_stt2": 1,      # Column B: secondary PL2 index (IGNORED)
    "pl2_chuong": 2,    # Column C: PL2 anatomical field / intervention domain
    "pl2_tenkt": 3,     # Column D: PL2 procedure name
}

PL2_INPUT_COLUMNS = ["pl2_stt", "pl2_chuong", "pl2_tenkt"]  # Used for preprocessing (excluding pl2_stt2)
PL2_MIN_COLUMNS = 4  # Minimum columns required (A, B, C, D)


def get_column_letter(index: int) -> str:
    """Convert 0-indexed column position to Excel column letter (A, B, C, ...)"""
    return chr(ord('A') + index)


def read_excel_sheets(file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Read PL1 and PL2 sheets from an Excel file using POSITION-BASED column mapping.
    
    Args:
        file: File path or file-like object (from Streamlit uploader)
        
    Returns:
        Tuple of (df_pl1, df_pl2, error_message)
        If successful, error_message is empty string
        
    Note:
        - Headers are ignored; columns are identified by position only
        - Internal alias names are applied based on position
    """
    try:
        xl = pd.ExcelFile(file)
        
        # Check for required sheets
        if "PL1" not in xl.sheet_names:
            return None, None, "Sheet 'PL1' not found in Excel file"
        if "PL2" not in xl.sheet_names:
            return None, None, "Sheet 'PL2' not found in Excel file"
        
        # Read sheets WITHOUT using headers - we'll assign our own column names
        df_pl1_raw = pd.read_excel(xl, sheet_name="PL1", header=None)
        df_pl2_raw = pd.read_excel(xl, sheet_name="PL2", header=None)
        
        # Skip the first row (original header) and reset index
        df_pl1_raw = df_pl1_raw.iloc[1:].reset_index(drop=True)
        df_pl2_raw = df_pl2_raw.iloc[1:].reset_index(drop=True)
        
        # ====================================================================
        # Validate PL1 column count
        # ====================================================================
        if len(df_pl1_raw.columns) < PL1_MIN_COLUMNS:
            missing_positions = [get_column_letter(i) for i in range(len(df_pl1_raw.columns), PL1_MIN_COLUMNS)]
            return None, None, (
                f"PL1: Expected at least {PL1_MIN_COLUMNS} columns (A-C), "
                f"but found only {len(df_pl1_raw.columns)}. "
                f"Missing column positions: {', '.join(missing_positions)}"
            )
        
        # ====================================================================
        # Validate PL2 column count
        # ====================================================================
        if len(df_pl2_raw.columns) < PL2_MIN_COLUMNS:
            missing_positions = [get_column_letter(i) for i in range(len(df_pl2_raw.columns), PL2_MIN_COLUMNS)]
            return None, None, (
                f"PL2: Expected at least {PL2_MIN_COLUMNS} columns (A-D), "
                f"but found only {len(df_pl2_raw.columns)}. "
                f"Missing column positions: {', '.join(missing_positions)}"
            )
        
        # ====================================================================
        # Apply internal alias names to PL1
        # ====================================================================
        df_pl1 = pd.DataFrame()
        for alias, pos in PL1_SCHEMA.items():
            if pos < len(df_pl1_raw.columns):
                df_pl1[alias] = df_pl1_raw.iloc[:, pos]
            else:
                # Output columns (D, E) may not exist initially
                df_pl1[alias] = ""
        
        # ====================================================================
        # Apply internal alias names to PL2
        # ====================================================================
        df_pl2 = pd.DataFrame()
        for alias, pos in PL2_SCHEMA.items():
            if pos < len(df_pl2_raw.columns):
                df_pl2[alias] = df_pl2_raw.iloc[:, pos]
            else:
                df_pl2[alias] = ""
        
        return df_pl1, df_pl2, ""
        
    except Exception as e:
        return None, None, f"Error reading Excel file: {str(e)}"


def get_alias_descriptions() -> Dict[str, Dict[str, str]]:
    """
    Return descriptions of all internal column aliases for UI display.
    
    Returns:
        Dictionary with 'PL1' and 'PL2' keys, each containing alias -> description mapping
    """
    return {
        "PL1": {
            "pl1_stt": "PL1 procedure ID (Column A)",
            "pl1_chuong": "PL1 specialty / medical field (Column B)",
            "pl1_tenkt": "PL1 procedure name (Column C)",
            "pl2_stt": "Aggregated PL2 IDs - OUTPUT (Column D)",
            "pl2_tenkt": "Aggregated PL2 procedure names - OUTPUT (Column E)",
        },
        "PL2": {
            "pl2_stt": "PL2 procedure ID (Column A)",
            "pl2_stt2": "Secondary PL2 index - IGNORED (Column B)",
            "pl2_chuong": "PL2 anatomical field (Column C)",
            "pl2_tenkt": "PL2 procedure name (Column D)",
        }
    }


def save_preprocessed(df_pl1: pd.DataFrame, df_pl2: pd.DataFrame, output_path: str) -> str:
    """
    Save preprocessed dataframes to an Excel file with two sheets.
    
    Args:
        df_pl1: Preprocessed PL1 dataframe (with internal alias columns)
        df_pl2: Preprocessed PL2 dataframe (with internal alias columns)
        output_path: Full path for the output Excel file
        
    Returns:
        Error message if failed, empty string if successful
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_pl1.to_excel(writer, sheet_name='PL1_PREPROCESSED', index=False)
            df_pl2.to_excel(writer, sheet_name='PL2_PREPROCESSED', index=False)
        return ""
    except Exception as e:
        return f"Error saving Excel file: {str(e)}"


def get_output_path(input_path: str) -> str:
    """
    Generate output path in the same directory as input file.
    
    Args:
        input_path: Path to the input Excel file
        
    Returns:
        Path for the preprocessed output file
    """
    directory = os.path.dirname(input_path)
    return os.path.join(directory, "preprocessed_output.xlsx")

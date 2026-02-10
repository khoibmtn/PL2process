"""
File I/O module for PL2process.
Handles Excel reading and writing with DYNAMIC sheet selection.

COLUMN NAMING CONVENTION (CASE-INSENSITIVE):
Each sheet must have columns named with the sheet name as prefix:
- {sheet_name}_stt: Record ID
- {sheet_name}_chuong: Chapter/specialty
- {sheet_name}_tenkt: Procedure name

Example: Sheet "TT23" needs columns: tt23_stt, tt23_chuong, tt23_tenkt (any case)
"""

import os
from typing import Tuple, Optional, Dict, List
import pandas as pd


def get_sheet_names(file) -> Tuple[List[str], str]:
    """
    Get all sheet names from an Excel file.
    
    Args:
        file: File path or file-like object (from Streamlit uploader)
        
    Returns:
        Tuple of (list of sheet names, error message)
        If successful, error_message is empty string
    """
    try:
        xl = pd.ExcelFile(file)
        return xl.sheet_names, ""
    except Exception as e:
        return [], f"Error reading Excel file: {str(e)}"


def get_required_columns(sheet_name: str) -> List[str]:
    """
    Get the required column names for a given sheet (lowercase).
    
    Args:
        sheet_name: Name of the sheet
        
    Returns:
        List of required column names (lowercase)
    """
    sheet_lower = sheet_name.lower()
    return [
        f"{sheet_lower}_stt",
        f"{sheet_lower}_chuong",
        f"{sheet_lower}_tenkt"
    ]


def find_column_case_insensitive(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """
    Find a column in the dataframe matching target_col (case-insensitive).
    
    Args:
        df: DataFrame to search
        target_col: Target column name (lowercase)
        
    Returns:
        Actual column name if found, None if not found
    """
    target_lower = target_col.lower()
    for col in df.columns:
        if str(col).lower() == target_lower:
            return col
    return None


def validate_sheet_columns(df: pd.DataFrame, sheet_name: str) -> Tuple[str, Dict[str, str]]:
    """
    Validate that the sheet has all required columns (case-insensitive).
    
    Args:
        df: DataFrame loaded from the sheet
        sheet_name: Name of the sheet
        
    Returns:
        Tuple of (error_message, column_mapping)
        - error_message: Empty string if successful
        - column_mapping: Dict mapping required column names to actual column names
    """
    required_cols = get_required_columns(sheet_name)
    column_mapping = {}
    missing_cols = []
    
    for req_col in required_cols:
        actual_col = find_column_case_insensitive(df, req_col)
        if actual_col:
            column_mapping[req_col] = actual_col
        else:
            missing_cols.append(req_col)
    
    if missing_cols:
        return (
            f"Sheet '{sheet_name}' is missing required columns: {missing_cols}. "
            f"Expected columns (case-insensitive): {required_cols}",
            {}
        )
    
    return "", column_mapping


def normalize_columns(df: pd.DataFrame, sheet_name: str, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns to standardized lowercase names.
    
    Args:
        df: Original DataFrame
        sheet_name: Sheet name
        column_mapping: Mapping from required names to actual names
        
    Returns:
        DataFrame with standardized column names
    """
    result_df = df.copy()
    sheet_lower = sheet_name.lower()
    
    # Create rename mapping: actual_name -> standardized_name
    rename_map = {}
    for req_col, actual_col in column_mapping.items():
        # Standardize to lowercase sheet name
        rename_map[actual_col] = req_col
    
    result_df = result_df.rename(columns=rename_map)
    return result_df


def read_sheet(file, sheet_name: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Read a single sheet from an Excel file and validate its structure.
    
    Args:
        file: File path or file-like object
        sheet_name: Name of the sheet to read
        
    Returns:
        Tuple of (DataFrame, error_message)
        If successful, error_message is empty string
    """
    try:
        xl = pd.ExcelFile(file)
        
        if sheet_name not in xl.sheet_names:
            return None, f"Sheet '{sheet_name}' not found in Excel file"
        
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        # Validate required columns (case-insensitive)
        error, column_mapping = validate_sheet_columns(df, sheet_name)
        if error:
            return None, error
        
        # Normalize column names to lowercase
        df = normalize_columns(df, sheet_name, column_mapping)
        
        return df, ""
        
    except Exception as e:
        return None, f"Error reading sheet '{sheet_name}': {str(e)}"


def read_two_sheets(
    file, 
    sheet1_name: str, 
    sheet2_name: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Read two sheets from an Excel file and validate their structure.
    
    Args:
        file: File path or file-like object
        sheet1_name: Name of the first sheet
        sheet2_name: Name of the second sheet
        
    Returns:
        Tuple of (df_sheet1, df_sheet2, error_message)
        If successful, error_message is empty string
    """
    try:
        xl = pd.ExcelFile(file)
        
        # Check both sheets exist
        if sheet1_name not in xl.sheet_names:
            return None, None, f"Sheet '{sheet1_name}' not found in Excel file"
        if sheet2_name not in xl.sheet_names:
            return None, None, f"Sheet '{sheet2_name}' not found in Excel file"
        
        # Read sheets
        df1 = pd.read_excel(xl, sheet_name=sheet1_name)
        df2 = pd.read_excel(xl, sheet_name=sheet2_name)
        
        # Validate columns (case-insensitive)
        error1, col_map1 = validate_sheet_columns(df1, sheet1_name)
        if error1:
            return None, None, error1
        
        error2, col_map2 = validate_sheet_columns(df2, sheet2_name)
        if error2:
            return None, None, error2
        
        # Normalize column names to lowercase
        df1 = normalize_columns(df1, sheet1_name, col_map1)
        df2 = normalize_columns(df2, sheet2_name, col_map2)
        
        return df1, df2, ""
        
    except Exception as e:
        return None, None, f"Error reading Excel file: {str(e)}"


def get_column_mappings(sheet_name: str) -> Dict[str, str]:
    """
    Get column alias mappings for a sheet.
    
    Maps external column names to internal processing names.
    
    Args:
        sheet_name: Name of the sheet
        
    Returns:
        Dictionary mapping external column names to internal names
    """
    sheet_lower = sheet_name.lower()
    return {
        f"{sheet_lower}_stt": "stt",
        f"{sheet_lower}_chuong": "chuong",
        f"{sheet_lower}_tenkt": "tenkt"
    }


def save_preprocessed(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    sheet1_name: str,
    sheet2_name: str,
    output_path: str
) -> str:
    """
    Save preprocessed dataframes to an Excel file with two sheets.
    
    Args:
        df1: Preprocessed dataframe for sheet 1
        df2: Preprocessed dataframe for sheet 2
        sheet1_name: Name of sheet 1
        sheet2_name: Name of sheet 2
        output_path: Full path for the output Excel file
        
    Returns:
        Error message if failed, empty string if successful
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name=f'{sheet1_name}_PREPROCESSED', index=False)
            df2.to_excel(writer, sheet_name=f'{sheet2_name}_PREPROCESSED', index=False)
        return ""
    except Exception as e:
        return f"Error saving Excel file: {str(e)}"


def export_results_to_excel(
    df_output: pd.DataFrame,
    df_output_expand: pd.DataFrame,
    config: Dict,
    timestamp=None
) -> bytes:
    """
    Export matching results to Excel with three sheets.
    
    Sheets:
    - output: Aggregated by MASTER
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
    import io
    from datetime import datetime
    
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
            "master_sheet": config.get("master_sheet", ""),
            "match_sheet": config.get("match_sheet", ""),
            "weight_name": config.get("weight_name", 0.7),
            "weight_chuong": config.get("weight_chuong", 0.3),
            "high_threshold": config.get("high_threshold", 80),
            "medium_threshold": config.get("medium_threshold", 65),
            "ai_enabled": config.get("ai_enabled", False),
            "ai_provider": config.get("ai_provider", "local"),
            "execution_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }])
        config_df.to_excel(writer, sheet_name='CONFIG_USED', index=False)
    
    output.seek(0)
    return output.getvalue()

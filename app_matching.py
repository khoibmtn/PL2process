"""
PL2process Step 2 - Semantic Matching Application

This Streamlit app allows:
- Uploading preprocessed Excel files from Step 1
- Configuring matching weights and thresholds
- Setting record limits for testing
- Running semantic matching between PL1 and PL2 procedures
- Filtering and viewing results by compatibility level
- Exporting results to Excel (output + output_expand)

OUTPUT TABLES:
- output: Aggregated by PL1 (primary display)
- output_expand: One row per match (for debugging)
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from matching import (
    run_matching,
    classify_results,
    export_to_excel,
    get_summary_stats,
    filter_output_by_muc_do,
    DEFAULT_CONFIG
)

# AI refinement is optional - only import if available
AI_REFINEMENT_AVAILABLE = False
try:
    from ai_refinement import (
        apply_ai_refinement,
        DEFAULT_AI_CONFIG
    )
    AI_REFINEMENT_AVAILABLE = True
except ImportError:
    # sentence-transformers not installed
    DEFAULT_AI_CONFIG = {
        "enabled": False,
        "min_score": 50,
        "max_score": 70,
        "alpha": 0.7
    }


def load_preprocessed_file(uploaded_file):
    """
    Load preprocessed Excel file and validate structure.
    """
    try:
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        
        # Check for required sheets
        if "PL1_PREPROCESSED" not in sheet_names:
            return None, None, "Missing sheet: PL1_PREPROCESSED"
        if "PL2_PREPROCESSED" not in sheet_names:
            return None, None, "Missing sheet: PL2_PREPROCESSED"
        
        # Read sheets
        df_pl1 = pd.read_excel(xl, sheet_name="PL1_PREPROCESSED")
        df_pl2 = pd.read_excel(xl, sheet_name="PL2_PREPROCESSED")
        
        # Validate required columns
        required_pl1 = ["pl1_stt", "pl1_chuong", "pl1_tenkt", "normalized_tenkt", "normalized_chuong"]
        required_pl2 = ["pl2_stt", "pl2_chuong", "pl2_tenkt", "normalized_tenkt", "normalized_chuong"]
        
        missing_pl1 = [col for col in required_pl1 if col not in df_pl1.columns]
        missing_pl2 = [col for col in required_pl2 if col not in df_pl2.columns]
        
        if missing_pl1:
            return None, None, f"PL1_PREPROCESSED missing columns: {missing_pl1}"
        if missing_pl2:
            return None, None, f"PL2_PREPROCESSED missing columns: {missing_pl2}"
        
        return df_pl1, df_pl2, None
        
    except Exception as e:
        return None, None, f"Error reading file: {str(e)}"


def display_config_sidebar():
    """
    Display configuration controls in the sidebar.
    """
    st.sidebar.header("⚙️ Matching Configuration")
    
    # ==========================================================================
    # Test Mode: Record Limits
    # ==========================================================================
    st.sidebar.markdown("### 🧪 Test Mode (Record Limits)")
    
    limit_pl1 = st.sidebar.number_input(
        "Limit PL1 records",
        min_value=0,
        max_value=10000,
        value=0,
        step=10,
        help="Limit number of PL1 records for testing (0 = use ALL)"
    )
    
    limit_pl2 = st.sidebar.number_input(
        "Limit PL2 records",
        min_value=0,
        max_value=10000,
        value=0,
        step=10,
        help="Limit number of PL2 records for testing (0 = use ALL)"
    )
    
    if limit_pl1 > 0 or limit_pl2 > 0:
        st.sidebar.warning("⚠️ Test mode active: using limited records")
    
    st.sidebar.markdown("---")
    
    # ==========================================================================
    # Weight Settings
    # ==========================================================================
    st.sidebar.markdown("### 📊 Weight Settings")
    
    weight_name = st.sidebar.slider(
        "Name similarity weight",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIG["weight_name"],
        step=0.05,
        help="Weight for procedure name similarity. Chapter weight = 1 - this value."
    )
    weight_chuong = 1.0 - weight_name
    
    st.sidebar.info(f"📌 Chapter weight: **{weight_chuong:.2f}**")
    
    st.sidebar.markdown("---")
    
    # ==========================================================================
    # Threshold Settings
    # ==========================================================================
    st.sidebar.markdown("### 🎯 Threshold Settings")
    
    high_threshold = st.sidebar.number_input(
        "High compatibility (≥)",
        min_value=0,
        max_value=100,
        value=DEFAULT_CONFIG["high_threshold"],
        step=5,
        help="Score threshold for 'Cao' classification"
    )
    
    medium_threshold = st.sidebar.number_input(
        "Medium compatibility (≥)",
        min_value=0,
        max_value=int(high_threshold),
        value=min(DEFAULT_CONFIG["medium_threshold"], int(high_threshold)),
        step=5,
        help="Score threshold for 'Trung bình' classification"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Classification Legend")
    st.sidebar.markdown(f"""
    - 🟢 **Cao**: score ≥ {high_threshold}
    - 🟡 **Trung bình**: score ≥ {medium_threshold}
    - 🔴 **Thấp**: score < {medium_threshold}
    """)
    
    st.sidebar.markdown("---")
    
    # ==========================================================================
    # AI Embedding Refinement (OPTIONAL)
    # ==========================================================================
    st.sidebar.markdown("### 🤖 AI Embedding Refinement")
    
    if not AI_REFINEMENT_AVAILABLE:
        st.sidebar.warning("⚠️ AI not available. Install: `pip install sentence-transformers`")
        ai_enabled = False
    else:
        ai_enabled = st.sidebar.checkbox(
            "Apply AI embedding refinement",
            value=DEFAULT_AI_CONFIG["enabled"],
            help="Enable AI-based refinement for selected score range"
        )
    
    if ai_enabled:
        st.sidebar.info("⚠️ AI refinement is OPTIONAL and respects all exclusion rules")
        
        # Provider selection
        ai_provider = st.sidebar.radio(
            "Embedding Provider",
            options=["local", "openai", "gemini"],
            format_func=lambda x: {
                "local": "🖥️ Local (sentence-transformers)",
                "openai": "☁️ OpenAI API",
                "gemini": "✨ Google Gemini"
            }.get(x, x),
            help="Choose embedding provider: Local (free), OpenAI, or Gemini"
        )
        
        # Provider-specific settings
        if ai_provider == "openai":
            ai_api_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key (starts with sk-...)"
            )
            
            ai_openai_model = st.sidebar.selectbox(
                "OpenAI Model",
                options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0,
                help="text-embedding-3-small: Faster, cheaper | text-embedding-3-large: Better quality"
            )
            ai_gemini_model = "models/text-embedding-004"
            
            if not ai_api_key:
                st.sidebar.warning("⚠️ Enter OpenAI API key to use")
                
        elif ai_provider == "gemini":
            ai_api_key = st.sidebar.text_input(
                "Google Gemini API Key",
                type="password",
                help="Your Gemini API key from Google AI Studio"
            )
            
            ai_gemini_model = st.sidebar.selectbox(
                "Gemini Model",
                options=["models/text-embedding-004", "models/embedding-001"],
                index=0,
                help="text-embedding-004: Latest model | embedding-001: Legacy"
            )
            ai_openai_model = "text-embedding-3-small"
            
            if not ai_api_key:
                st.sidebar.warning("⚠️ Enter Gemini API key to use")
        else:
            ai_api_key = None
            ai_openai_model = "text-embedding-3-small"
            ai_gemini_model = "models/text-embedding-004"
        
        ai_min_score = st.sidebar.number_input(
            "Min original score",
            min_value=0,
            max_value=100,
            value=DEFAULT_AI_CONFIG["min_score"],
            step=5,
            help="Only refine rows with score >= this value"
        )
        
        ai_max_score = st.sidebar.number_input(
            "Max original score",
            min_value=0,
            max_value=100,
            value=DEFAULT_AI_CONFIG["max_score"],
            step=5,
            help="Only refine rows with score <= this value"
        )
        
        ai_alpha = st.sidebar.slider(
            "Refinement strength (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_AI_CONFIG["alpha"],
            step=0.05,
            help="alpha=1.0 keeps original, alpha=0.0 uses only embedding"
        )
        
        st.sidebar.caption(f"Formula: {ai_alpha:.0%} original + {1-ai_alpha:.0%} embedding")
    else:
        ai_min_score = DEFAULT_AI_CONFIG["min_score"]
        ai_max_score = DEFAULT_AI_CONFIG["max_score"]
        ai_alpha = DEFAULT_AI_CONFIG["alpha"]
        ai_provider = "local"
        ai_api_key = None
        ai_openai_model = "text-embedding-3-small"
        ai_gemini_model = "models/text-embedding-004"
    
    return {
        "weight_name": weight_name,
        "weight_chuong": weight_chuong,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "limit_pl1": limit_pl1,
        "limit_pl2": limit_pl2,
        "ai_enabled": ai_enabled,
        "ai_min_score": ai_min_score,
        "ai_max_score": ai_max_score,
        "ai_alpha": ai_alpha,
        "ai_provider": ai_provider,
        "ai_api_key": ai_api_key,
        "ai_openai_model": ai_openai_model,
        "ai_gemini_model": ai_gemini_model
    }


def display_filter_controls():
    """
    Display compatibility level filter controls.
    """
    st.markdown("### 🔍 Filter Results")
    
    selected_levels = st.multiselect(
        "Show compatibility levels:",
        options=["Cao", "Trung bình", "Thấp"],
        default=["Cao", "Trung bình", "Thấp"],
        help="Select which compatibility levels to display"
    )
    
    return selected_levels


def display_aggregated_table(df_output: pd.DataFrame, selected_levels: list):
    """
    Display the aggregated output table (one row per PL1).
    """
    # Filter by selected levels
    df_filtered = filter_output_by_muc_do(df_output, selected_levels)
    
    if len(df_filtered) == 0:
        st.warning("No results match the selected filter criteria.")
        return
    
    # Prepare display with Vietnamese column names
    df_display = df_filtered[[
        "pl1_chuong",
        "pl1_tenkt",
        "pl2_tenkt",
        "totalscore",
        "muc_do"
    ]].copy()
    
    df_display.columns = [
        "PL1 - Chương",
        "PL1 - Tên kỹ thuật",
        "PL2 - Tên kỹ thuật ghép",
        "Tổng điểm",
        "Mức độ phù hợp"
    ]
    
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500
    )
    
    st.caption(f"Showing {len(df_filtered)} of {len(df_output)} PL1 records")


def display_expanded_table(df_expand: pd.DataFrame, selected_levels: list):
    """
    Display the expanded output table (one row per match).
    """
    # Filter by selected levels
    df_filtered = df_expand[df_expand["muc_do"].isin(selected_levels)].copy()
    
    if len(df_filtered) == 0:
        st.warning("No results match the selected filter criteria.")
        return
    
    # Prepare display
    df_display = df_filtered[[
        "pl1_chuong",
        "pl1_tenkt",
        "pl2_tenkt",
        "totalscore",
        "muc_do"
    ]].copy()
    
    df_display.columns = [
        "PL1 - Chương",
        "PL1 - Tên kỹ thuật",
        "PL2 - Tên kỹ thuật",
        "Tổng điểm",
        "Mức độ phù hợp"
    ]
    
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400
    )
    
    st.caption(f"Showing {len(df_filtered)} of {len(df_expand)} matches")


def display_summary_stats(stats: dict):
    """
    Display summary statistics with visual indicators.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Matches", stats["total"])
    
    with col2:
        st.metric(
            "🟢 Cao",
            f"{stats['cao']} ({stats.get('cao_pct', 0)}%)"
        )
    
    with col3:
        st.metric(
            "🟡 Trung bình",
            f"{stats['trung_binh']} ({stats.get('trung_binh_pct', 0)}%)"
        )
    
    with col4:
        st.metric(
            "🔴 Thấp",
            f"{stats['thap']} ({stats.get('thap_pct', 0)}%)"
        )


def main():
    # Page configuration
    st.set_page_config(
        page_title="PL2process - Step 2: Matching",
        page_icon="🔗",
        layout="wide"
    )
    
    # Title
    st.title("🔗 PL2process - Step 2: Semantic Matching")
    st.markdown("""
    **Purpose:** Match PL2 procedures to PL1 procedures using semantic similarity.  
    **Input:** Preprocessed Excel file from Step 1 (with PL1_PREPROCESSED and PL2_PREPROCESSED sheets).
    """)
    
    # Sidebar configuration
    config = display_config_sidebar()
    
    # Initialize session state
    if 'df_pl1' not in st.session_state:
        st.session_state.df_pl1 = None
    if 'df_pl2' not in st.session_state:
        st.session_state.df_pl2 = None
    if 'output_expand' not in st.session_state:
        st.session_state.output_expand = None
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'last_config' not in st.session_state:
        st.session_state.last_config = None
    
    st.divider()
    
    # ========================================================================
    # Step 1: Upload preprocessed file
    # ========================================================================
    st.header("📁 Step 1: Upload Preprocessed File")
    
    uploaded_file = st.file_uploader(
        "Choose the preprocessed Excel file from Step 1",
        type=['xlsx'],
        help="Must contain sheets: PL1_PREPROCESSED and PL2_PREPROCESSED"
    )
    
    if uploaded_file is not None:
        df_pl1, df_pl2, error = load_preprocessed_file(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
            st.stop()
        
        st.session_state.df_pl1 = df_pl1
        st.session_state.df_pl2 = df_pl2
        
        st.success(f"✅ File loaded: PL1 ({len(df_pl1)} rows), PL2 ({len(df_pl2)} rows)")
        
        # Show preview
        with st.expander("📋 Preview loaded data"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**PL1_PREPROCESSED (first 5 rows)**")
                st.dataframe(df_pl1.head(), use_container_width=True)
            with col2:
                st.markdown("**PL2_PREPROCESSED (first 5 rows)**")
                st.dataframe(df_pl2.head(), use_container_width=True)
    
    # ========================================================================
    # Step 2: Run matching
    # ========================================================================
    if st.session_state.df_pl1 is not None and st.session_state.df_pl2 is not None:
        st.divider()
        st.header("⚙️ Step 2: Run Matching")
        
        # Show current configuration
        limit_info = ""
        if config['limit_pl1'] > 0 or config['limit_pl2'] > 0:
            limit_info = f" | 🧪 **Test mode**: PL1 limit={config['limit_pl1'] or 'ALL'}, PL2 limit={config['limit_pl2'] or 'ALL'}"
        
        st.markdown(f"""
        **Current Configuration:**
        - Name weight: **{config['weight_name']:.2f}** | Chapter weight: **{config['weight_chuong']:.2f}**
        - High threshold: **≥ {config['high_threshold']}** | Medium threshold: **≥ {config['medium_threshold']}**{limit_info}
        """)
        
        if st.button("🚀 Run Matching", type="primary", use_container_width=True):
            # Create progress bar and status text placeholders
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Calculate actual PL2 count (with limits applied)
            total_pl2 = len(st.session_state.df_pl2)
            if config['limit_pl2'] > 0:
                total_pl2 = min(config['limit_pl2'], total_pl2)
            
            # Progress callback function
            def update_progress(processed, total):
                pct = processed / total
                progress_bar.progress(pct)
                progress_text.markdown(
                    f"**Số kỹ thuật PL2 đã xử lý:** {processed} / {total} ({pct*100:.1f}%)"
                )
            
            # Run matching with progress callback
            output_expand, output = run_matching(
                st.session_state.df_pl1,
                st.session_state.df_pl2,
                weight_name=config['weight_name'],
                high_threshold=config['high_threshold'],
                medium_threshold=config['medium_threshold'],
                limit_pl1=config['limit_pl1'],
                limit_pl2=config['limit_pl2'],
                progress_callback=update_progress
            )
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            st.session_state.output_expand = output_expand
            st.session_state.output = output
            st.session_state.last_config = config.copy()
                
            st.success("✅ Matching completed!")
        
        # Re-classify button
        if st.session_state.output_expand is not None:
            if st.session_state.last_config != config:
                st.info("⚠️ Configuration changed. Click 'Re-classify' to update, or 'Run Matching' to re-run with new limits/weights.")
                
                if st.button("🔄 Re-classify with new thresholds", use_container_width=True):
                    output_expand, output = classify_results(
                        st.session_state.output_expand,
                        config['high_threshold'],
                        config['medium_threshold']
                    )
                    st.session_state.output_expand = output_expand
                    st.session_state.output = output
                    st.session_state.last_config = config.copy()
                    st.rerun()
    
    # ========================================================================
    # Step 2.5: AI Embedding Refinement (OPTIONAL)
    # ========================================================================
    if st.session_state.output_expand is not None and config.get('ai_enabled', False):
        st.divider()
        st.header("🤖 Step 2.5: AI Embedding Refinement")
        
        st.markdown(f"""
        **Configuration:**
        - Score range: **{config['ai_min_score']} - {config['ai_max_score']}**
        - Alpha: **{config['ai_alpha']:.2f}** ({config['ai_alpha']:.0%} original + {1-config['ai_alpha']:.0%} embedding)
        """)
        
        if st.button("🚀 Apply AI Refinement", type="secondary", use_container_width=True):
            # Validate API key if needed
            if config.get('ai_provider') == 'openai' and not config.get('ai_api_key'):
                st.error("❌ Please enter your OpenAI API key in the sidebar")
                st.stop()
            if config.get('ai_provider') == 'gemini' and not config.get('ai_api_key'):
                st.error("❌ Please enter your Gemini API key in the sidebar")
                st.stop()
            
            # Check if required package is installed
            try:
                from ai_refinement import apply_ai_refinement
                if config.get('ai_provider') == 'openai':
                    import openai  # Check if openai is installed
                elif config.get('ai_provider') == 'gemini':
                    import google.generativeai  # Check if google-generativeai is installed
            except ImportError as e:
                if 'openai' in str(e):
                    st.error("❌ Please install openai: `pip install openai`")
                elif 'google' in str(e):
                    st.error("❌ Please install google-generativeai: `pip install google-generativeai`")
                else:
                    st.error("❌ Please install sentence-transformers: `pip install sentence-transformers`")
                st.stop()
            
            # Create progress indicators
            ai_progress_bar = st.progress(0)
            ai_progress_text = st.empty()
            
            provider_labels = {"local": "Local", "openai": "OpenAI", "gemini": "Gemini"}
            provider_label = provider_labels.get(config.get('ai_provider'), 'Local')
            
            def update_ai_progress(processed, total):
                if total > 0:
                    pct = processed / total
                    ai_progress_bar.progress(pct)
                    ai_progress_text.markdown(
                        f"**AI refinement ({provider_label}):** {processed} / {total} ({pct*100:.1f}%)"
                    )
            
            # Apply refinement
            spinner_msgs = {
                "local": "Loading local AI model...",
                "openai": "Calling OpenAI API...",
                "gemini": "Calling Gemini API..."
            }
            spinner_msg = spinner_msgs.get(config.get('ai_provider'), "Loading AI model...")
            with st.spinner(spinner_msg):
                try:
                    refined_expand, ai_stats = apply_ai_refinement(
                        st.session_state.output_expand,
                        min_score=config['ai_min_score'],
                        max_score=config['ai_max_score'],
                        alpha=config['ai_alpha'],
                        high_threshold=config['high_threshold'],
                        medium_threshold=config['medium_threshold'],
                        provider=config.get('ai_provider', 'local'),
                        api_key=config.get('ai_api_key'),
                        openai_model=config.get('ai_openai_model', 'text-embedding-3-small'),
                        gemini_model=config.get('ai_gemini_model', 'models/text-embedding-004'),
                        progress_callback=update_ai_progress
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    ai_progress_bar.empty()
                    ai_progress_text.empty()
                    st.stop()
            
            # Clear progress
            ai_progress_bar.empty()
            ai_progress_text.empty()
            
            # Update session state
            st.session_state.output_expand = refined_expand
            st.session_state.ai_stats = ai_stats
            st.session_state.ai_provider_used = config.get('ai_provider', 'local')
            
            st.success(f"✅ AI Refinement completed using {provider_label}!")
        
        # Display AI statistics if available
        if 'ai_stats' in st.session_state and st.session_state.ai_stats:
            stats = st.session_state.ai_stats
            
            st.markdown("### 📊 AI Refinement Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Eligible", stats.get('eligible', 0))
            with col2:
                st.metric("❌ Excluded (rules)", stats.get('excluded_by_rules', 0))
            with col3:
                st.metric("⬆️ Upgraded", stats.get('upgraded', 0))
            with col4:
                st.metric("⬇️ Downgraded", stats.get('downgraded', 0))
            
            st.caption(f"Total rows: {stats.get('total_rows', 0)} | Excluded by score range: {stats.get('excluded_by_score_range', 0)}")
    
    # ========================================================================
    # Step 3: View results
    # ========================================================================
    if st.session_state.output is not None:
        st.divider()
        st.header("📊 Step 3: View Results")
        
        # Summary stats (from output_expand - the base table)
        stats = get_summary_stats(st.session_state.output_expand)
        display_summary_stats(stats)
        
        st.divider()
        
        # Filter controls
        selected_levels = display_filter_controls()
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📋 output (Aggregated by PL1)", "🔍 output_expand (Debug)"])
        
        with tab1:
            st.markdown("""
            **Primary view:** One row per PL1. Multiple matched PL2s are joined with ";".
            """)
            if selected_levels:
                display_aggregated_table(st.session_state.output, selected_levels)
            else:
                st.warning("Please select at least one compatibility level to display.")
        
        with tab2:
            st.markdown("""
            **Debug view:** One row per PL1-PL2 match. Use for detailed analysis.
            """)
            if selected_levels:
                display_expanded_table(st.session_state.output_expand, selected_levels)
            else:
                st.warning("Please select at least one compatibility level to display.")
        
        # ====================================================================
        # Step 4: Export
        # ====================================================================
        st.divider()
        st.header("💾 Step 4: Export Results")
        
        # Prepare export
        export_config = st.session_state.last_config or config
        export_bytes = export_to_excel(
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
        
        st.caption("Excel contains: **output** (aggregated) + **output_expand** (detailed) + **CONFIG_USED**")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from parsing import parse_data  # Assuming this is your existing parsing module
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Askify - Data Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Askify</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Data Analysis & Parsing Tool</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox("Choose a section:", ["Upload & Parse", "Data Analysis", "Results", "About"])
    
    st.header("Settings")
    show_preview = st.checkbox("Show data preview", value=True)
    save_results = st.checkbox("Save results", value=True)

# Main content based on selected page
if page == "Upload & Parse":
    st.header("📤 Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['csv', 'xlsx', 'json', 'txt'],
            help="Upload CSV, Excel, JSON, or text files for analysis"
        )
    
    with col2:
        st.info("""
        **Supported formats:**
        - CSV files (.csv)
        - Excel files (.xlsx)
        - JSON files (.json)
        - Text files (.txt)
        """)
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with st.spinner("Processing file..."):
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            }
            
            # Display file details
            st.subheader("File Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", uploaded_file.name)
            col2.metric("Type", uploaded_file.type)
            col3.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Parse the data
            try:
                # Read file content
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    content = uploaded_file.read().decode('utf-8')
                    df = pd.DataFrame({'content': content.split('\n')})
                
                # Store in session state
                st.session_state['raw_data'] = df
                st.session_state['file_details'] = file_details
                
                st.success("✅ File uploaded and processed successfully!")
                
                if show_preview:
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Total Columns", len(df.columns))
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Upload & Parse' section")
    else:
        df = st.session_state['raw_data']
        
        # Analysis options
        st.subheader("Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Choose analysis type:",
                ["Basic Statistics", "Data Cleaning", "Pattern Detection", "Custom Analysis"]
            )
        
        with col2:
            target_column = st.selectbox("Select target column:", df.columns.tolist())
        
        # Perform analysis based on type
        if analysis_type == "Basic Statistics":
            st.subheader("Basic Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df[target_column].mean():.2f}" if pd.api.types.is_numeric_dtype(df[target_column]) else "N/A")
            with col2:
                st.metric("Median", f"{df[target_column].median():.2f}" if pd.api.types.is_numeric_dtype(df[target_column]) else "N/A")
            with col3:
                st.metric("Std Dev", f"{df[target_column].std():.2f}" if pd.api.types.is_numeric_dtype(df[target_column]) else "N/A")
            
            # Distribution chart
            if pd.api.types.is_numeric_dtype(df[target_column]):
                st.subheader("Distribution")
                st.bar_chart(df[target_column].value_counts().head(20))
        
        elif analysis_type == "Data Cleaning":
            st.subheader("Data Cleaning")
            
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.warning("Missing values detected:")
                st.bar_chart(missing_values[missing_values > 0])
                
                if st.button("Remove missing values"):
                    cleaned_df = df.dropna()
                    st.session_state['cleaned_data'] = cleaned_df
                    st.success(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
            else:
                st.success("✅ No missing values found!")
        
        elif analysis_type == "Pattern Detection":
            st.subheader("Pattern Detection")
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
        
        # Store results
        if st.button("Save Analysis Results"):
            st.session_state['analysis_results'] = {
                'analysis_type': analysis_type,
                'target_column': target_column,
                'timestamp': datetime.now()
            }
            st.success("✅ Analysis results saved!")

elif page == "Results":
    st.header("📊 Results & Export")
    
    # Display saved results
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        st.subheader("Saved Analysis")
        st.json(results)
    
    # Export options
    st.subheader("Export Options")
    
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download as CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"askify_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download as Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"askify_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

elif page == "About":
    st.header("ℹ️ About Askify")
    
    st.markdown("""
    ### Welcome to Askify!
    
    **Askify** is a powerful data analysis tool designed to help you:
    - Upload and parse various data formats
    - Perform comprehensive data analysis
    - Detect patterns and insights
    - Export results in multiple formats
    
    ### Features:
    - 📊 **Multi-format support**: CSV, Excel, JSON, Text
    - 🔍 **Advanced analytics**: Statistics, patterns, correlations
    - 📈 **Visual insights**: Charts and graphs
    - 💾 **Export options**: CSV, Excel formats
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **Backend**: Python (Pandas, NumPy)
    - **Deployment**: Streamlit Cloud
    
    ### Usage:
    1. Upload your data file
    2. Choose analysis type
    3. View results and insights
    4. Export your findings
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: January 2025
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Deployed on Streamlit Cloud</p>
    </div>
    """,
    unsafe_allow_html=True
)

# hf_db.py
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
import os

class HuggingFaceDB:
    def __init__(self, repo_id, token):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        
    def load_table(self, table_name):
        """Load a table from Hugging Face dataset"""
        try:
            file_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"{table_name}.csv",
                repo_type="dataset"
            )
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Creating new {table_name} table: {e}")
            return pd.DataFrame()
    
    def save_table(self, table_name, df):
        """Save a table to Hugging Face dataset"""
        try:
            # Save locally first
            df.to_csv(f"{table_name}.csv", index=False)
            
            # Upload to Hugging Face
            self.api.upload_file(
                path_or_fileobj=f"{table_name}.csv",
                path_in_repo=f"{table_name}.csv",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            # Clean up local file
            if os.path.exists(f"{table_name}.csv"):
                os.remove(f"{table_name}.csv")
            return True
        except Exception as e:
            print(f"Error saving {table_name}: {e}")
            return False

def get_hf_db():
    if 'hf_db' not in st.session_state:
        token = st.secrets.get("HF_TOKEN")
        repo_id = st.secrets.get("HF_REPO_ID", "ElectraNelly/streamlit-app-data")
        if not token:
            st.error("❌ Hugging Face token not found in secrets!")
            return None
        st.session_state.hf_db = HuggingFaceDB(repo_id, token)
    return st.session_state.hf_db

# hybrid_db.py
import sqlite3
import pandas as pd
from datetime import datetime
from hf_db import get_hf_db
import streamlit as st

class HybridDB:
    def __init__(self):
        self.local_db_file = "users_chats.db"
        self.hf_db = get_hf_db()
        
    def get_conn(self):
        return sqlite3.connect(self.local_db_file, check_same_thread=False)
    
    def is_local_db_empty(self):
        """Check if local database has any users"""
        try:
            conn = self.get_conn()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users")
            user_count = cur.fetchone()[0]
            conn.close()
            return user_count == 0
        except:
            return True
    
    def auto_restore_on_startup(self):
        """Automatically restore from Hugging Face if local DB is empty"""
        if self.is_local_db_empty():
            st.info("🔄 No local data found. Restoring from cloud backup...")
            success = self.sync_from_hf()
            if success:
                st.success("✅ Data restored from cloud!")
            else:
                st.warning("❌ Cloud restore failed. Starting with fresh local database.")
            return success
        return True
    
    def sync_to_hf(self):
        """Sync all tables to Hugging Face"""
        try:
            conn = self.get_conn()
            
            # Sync users table
            users_df = pd.read_sql_query("SELECT * FROM users", conn)
            if not users_df.empty:
                self.hf_db.save_table("users", users_df)
            
            # Sync chats table  
            chats_df = pd.read_sql_query("SELECT * FROM chats", conn)
            if not chats_df.empty:
                self.hf_db.save_table("chats", chats_df)
            
            # Sync other tables
            for table in ["courses", "enrollments", "interventions", "learning_metrics", "knowledge_gaps"]:
                try:
                    table_df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    if not table_df.empty:
                        self.hf_db.save_table(table, table_df)
                except:
                    continue
            
            conn.close()
            print("✅ Synced to Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            return False
    
    def sync_from_hf(self):
        """Sync from Hugging Face to local SQLite"""
        try:
            conn = self.get_conn()
            
            # Load tables from HF
            users_df = self.hf_db.load_table("users")
            if not users_df.empty:
                users_df.to_sql("users", conn, if_exists="replace", index=False)
            
            chats_df = self.hf_db.load_table("chats")  
            if not chats_df.empty:
                chats_df.to_sql("chats", conn, if_exists="replace", index=False)
            
            # Sync other tables
            for table in ["courses", "enrollments", "interventions", "learning_metrics", "knowledge_gaps"]:
                try:
                    table_df = self.hf_db.load_table(table)
                    if not table_df.empty:
                        table_df.to_sql(table, conn, if_exists="replace", index=False)
                except:
                    continue
            
            conn.close()
            print("✅ Synced from Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Sync from HF failed: {e}")
            return False

# Global instance
hybrid_db = HybridDB()

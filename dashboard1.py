import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import psycopg2
from psycopg2 import pool
import traceback

# Page configuration
st.set_page_config(
    page_title="Director Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .big-font { font-size: 48px !important; font-weight: bold; }
    .status-badge {
        background: #000; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize connection pool
@st.cache_resource
def init_connection_pool():
    """Initialize PostgreSQL connection pool"""
    try:
        if not hasattr(st, 'secrets'):
            st.error("❌ Streamlit secrets not available. Ensure proper Streamlit environment.")
            return None

        if "database" not in st.secrets:
            st.error("❌ 'database' section not found in secrets.")
            st.info("""
            For local development, add to .streamlit/secrets.toml:
            ```toml
            [database]
            host = "51.178.30.30"
            port = 5433
            database = "rawahel_test"
            user = "readonly_user"
            password = "uJz8o99awc"
            sslmode = "require"
            ```
            """)
            return None

        return psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=st.secrets["database"]["host"],
            port=st.secrets["database"]["port"],
            database=st.secrets["database"]["database"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            sslmode=st.secrets["database"].get("sslmode", "require")
        )
    except Exception as e:
        st.error(f"Error initializing connection pool: {e}")
        st.code(traceback.format_exc())
        return None


def get_connection():
    """Get a connection from the pool"""
    conn_pool = init_connection_pool()
    if conn_pool:
        try:
            return conn_pool.getconn()
        except Exception as e:
            st.error(f"Failed to get connection: {e}")
    return None


def return_connection(conn):
    """Return connection to pool"""
    conn_pool = init_connection_pool()
    if conn_pool and conn:
        conn_pool.putconn(conn)

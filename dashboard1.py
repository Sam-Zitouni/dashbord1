# Fixed database connection management

import streamlit as st
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import traceback

# Initialize connection pool (keep this cached)
@st.cache_resource
def init_connection_pool():
    """Initialize PostgreSQL connection pool"""
    try:
        # Verify secrets exist
        if "database" not in st.secrets:
            st.error("❌ 'database' section not found in secrets.toml")
            return None
        
        # Verify required keys
        required_keys = ["host", "port", "database", "user", "password"]
        missing_keys = [key for key in required_keys if key not in st.secrets["database"]]
        
        if missing_keys:
            st.error(f"❌ Missing keys in secrets.toml: {', '.join(missing_keys)}")
            return None
        
        # Create connection pool with better error handling
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # min and max connections
            host=st.secrets["database"]["host"],
            port=st.secrets["database"]["port"],
            database=st.secrets["database"]["database"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            connect_timeout=10  # Add timeout
        )
        
        if connection_pool:
            st.success("✅ Database connection pool created successfully")
            return connection_pool
            
    except psycopg2.OperationalError as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("Verify: 1) Database is running, 2) Credentials are correct, 3) Network allows connection")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Context manager for safe connection handling
@contextmanager
def get_db_connection():
    """
    Context manager for safe database connections.
    Usage:
        with get_db_connection() as conn:
            # use conn here
    """
    pool = init_connection_pool()
    if not pool:
        raise Exception("Connection pool not available")
    
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            pool.putconn(conn)

# FIXED: Safe data fetching function with parameterized queries
def fetch_bookings_data(date_range):
    """Fetch bookings data with all statuses - SAFE VERSION"""
    try:
        with get_db_connection() as conn:
            # Use parameterized query to prevent SQL injection
            query = """
            SELECT 
                b.id,
                b.created_at,
                b.status,
                b.total_amount,
                b.route_id,
                b.agency_id,
                b.customer_id,
                COALESCE(b.source, 'direct') as booking_source,
                r.name as route_name,
                a.name as agency_name
            FROM bookings b
            LEFT JOIN routes r ON b.route_id = r.id
            LEFT JOIN agencies a ON b.agency_id = a.id
            WHERE b.created_at >= %s 
            AND b.created_at <= %s
            """
            
            # Use pandas with params
            df = pd.read_sql(query, conn, params=(date_range[0], date_range[1]))
            
            if df.empty:
                st.warning(f"⚠️ No bookings found between {date_range[0]} and {date_range[1]}")
            
            return df
            
    except psycopg2.Error as e:
        st.error(f"❌ Database error fetching bookings: {e.pgerror}")
        st.code(f"Error code: {e.pgcode}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error fetching bookings: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

# Test connection function
def test_database_connection():
    """Test database connection and show detailed diagnostics"""
    st.subheader("🔍 Database Connection Test")
    
    try:
        with st.spinner("Testing connection..."):
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Test 1: PostgreSQL version
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                st.success(f"✅ Connected to: {version[:50]}...")
                
                # Test 2: Current database
                cursor.execute("SELECT current_database();")
                db_name = cursor.fetchone()[0]
                st.success(f"✅ Database: {db_name}")
                
                # Test 3: Check if tables exist
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    st.success(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
                else:
                    st.warning("⚠️ No tables found in public schema")
                
                # Test 4: Check bookings table
                if 'bookings' in tables:
                    cursor.execute("SELECT COUNT(*) FROM bookings;")
                    count = cursor.fetchone()[0]
                    st.success(f"✅ Bookings table has {count} records")
                else:
                    st.error("❌ 'bookings' table not found!")
                
                cursor.close()
                
                return True
                
    except Exception as e:
        st.error(f"❌ Connection test failed: {str(e)}")
        st.code(traceback.format_exc())
        return False

# Alternative: Direct connection without pool (for testing)
def get_direct_connection():
    """Get a direct database connection for testing"""
    try:
        conn = psycopg2.connect(
            host=st.secrets["database"]["host"],
            port=st.secrets["database"]["port"],
            database=st.secrets["database"]["database"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            connect_timeout=10
        )
        return conn
    except Exception as e:
        st.error(f"Direct connection failed: {str(e)}")
        return None
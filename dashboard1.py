import streamlit as st
import psycopg2
from psycopg2 import pool
import pandas as pd
import traceback

# ----------------------------------------------------
# 1️⃣ APP CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Rawahel Agency Dashboard",
    page_icon="🧭",
    layout="wide"
)
st.title("🧭 Rawahel Agency Dashboard")
st.caption("Real-time insights powered by PostgreSQL")

# ----------------------------------------------------
# 2️⃣ LOAD DATABASE CONFIG FROM SECRETS
# ----------------------------------------------------
try:
    db_config = st.secrets["database"]
    st.success("✅ Secrets loaded successfully.")
except Exception:
    st.error("❌ Database configuration not found in `.streamlit/secrets.toml`.")
    st.info("Make sure your `.streamlit/secrets.toml` file contains a [database] section.")
    st.stop()

# ----------------------------------------------------
# 3️⃣ INITIALIZE CONNECTION POOL
# ----------------------------------------------------
@st.cache_resource
def init_connection_pool():
    try:
        connection_pool = pool.SimpleConnectionPool(
            1, 10,
            host=db_config.get("host"),
            port=db_config.get("port"),
            database=db_config.get("database"),
            user=db_config.get("user"),
            password=db_config.get("password")
        )
        if connection_pool:
            st.success("✅ Database connection pool created successfully.")
        return connection_pool
    except Exception as e:
        st.error(f"❌ Failed to create connection pool: {e}")
        st.text(traceback.format_exc())
        return None

connection_pool = init_connection_pool()
if not connection_pool:
    st.stop()

# ----------------------------------------------------
# 4️⃣ DATABASE QUERY FUNCTION
# ----------------------------------------------------
def run_query(query):
    conn = None
    try:
        conn = connection_pool.getconn()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    except Exception as e:
        st.error(f"❌ Query failed: {e}")
        st.text(traceback.format_exc())
        return pd.DataFrame()
    finally:
        if conn:
            connection_pool.putconn(conn)

# ----------------------------------------------------
# 5️⃣ LOAD DATA FROM DATABASE
# ----------------------------------------------------
st.markdown("### 📦 Loading Data...")

agency_query = """
SELECT
    agency_name,
    trips,
    cancellations,
    revenue,
    commission,
    growth
FROM agencies
ORDER BY revenue DESC
LIMIT 10;
"""

agency_performance = run_query(agency_query)

if agency_performance.empty:
    st.warning("⚠️ No data returned from the database.")
    st.stop()

st.success("✅ Data loaded successfully!")

# ----------------------------------------------------
# 6️⃣ SUMMARY METRICS
# ----------------------------------------------------
total_revenue = agency_performance["revenue"].sum()
total_commission = agency_performance["commission"].sum()
avg_growth = agency_performance["growth"].mean()
total_trips = agency_performance["trips"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"{total_revenue:,.0f} TND")
col2.metric("🏦 Total Commission", f"{total_commission:,.0f} TND")
col3.metric("📈 Average Growth", f"{avg_growth:.1f}%")
col4.metric("🚗 Total Trips", f"{total_trips:,}")

# ----------------------------------------------------
# 7️⃣ PERFORMANCE CHART
# ----------------------------------------------------
st.markdown("### 📊 Top 10 Agencies by Revenue")

chart_data = agency_performance[["agency_name", "revenue", "growth"]]
st.bar_chart(
    data=chart_data,
    x="agency_name",
    y="revenue",
    use_container_width=True
)

# ----------------------------------------------------
# 8️⃣ TOP PERFORMING AGENCIES SECTION
# ----------------------------------------------------
st.markdown("### 🥇 Top Performing Agencies")

currency = "TND"

for idx, row in agency_performance.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 3])

        with col1:
            st.markdown(f"### 🏅 #{idx + 1}")

        with col2:
            st.markdown(f"**{row['agency_name']}**")
            st.caption(f"{int(row['trips'])} trips | {int(row['cancellations'])} cancellations")
            progress = min(max(row['growth'], 0) / 25, 1.0)
            st.progress(progress)

        with col3:
            st.markdown(f"**{row['revenue']:,.0f} {currency}**")
            st.caption(f"Commission: {row['commission']:,.0f} {currency}")
            st.markdown(
                f"<span style='color: green; font-weight: bold;'>+{row['growth']:.1f}%</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")

# ----------------------------------------------------
# 9️⃣ DEBUG INFO (OPTIONAL)
# ----------------------------------------------------
with st.expander("⚙️ Debug Info"):
    st.write("**Secrets keys found:**", list(st.secrets.keys()))
    st.json({
        "host": db_config.get("host"),
        "port": db_config.get("port"),
        "database": db_config.get("database"),
        "user": db_config.get("user"),
    })
    st.caption("_Password is hidden for security._")

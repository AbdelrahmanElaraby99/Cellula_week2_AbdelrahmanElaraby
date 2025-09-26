import streamlit as st
import sqlite3
from sqlite3 import Connection
import io
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu

# Local imports
from imagecaption import ImageCaptioner
from classifier import classify_text


# --- Style Config ---
st.markdown(
    """
    <style>
    /* Add blue border around the entire page */
    .stApp {
        border: 10px solid #014BB6;  /* Cobalt blue border */
        border-radius: 0px;        /* Optional rounded corners */
        padding: 10px;              /* Space inside the border */
        margin: 15px;               /* Space outside the border */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- DB Setup ---
def get_connection() -> Connection:
    return sqlite3.connect("helping_files/website.db", check_same_thread=False)

def insert_classification(user_id, text_query, image_caption, image_bytes, class_name, confidence_score):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO toxic_classification 
        (user_id, text_query, image_caption, image_src, class, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, text_query, image_caption, image_bytes, class_name, confidence_score))
    conn.commit()
    conn.close()

def create_user(email, name, password):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (user_id, name, password) VALUES (?, ?, ?)",
                    (email, name, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def validate_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE user_id = ? AND password = ?", (email, password))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


# --- Initialize Captioner ---
@st.cache_resource
def load_captioner():
    return ImageCaptioner()

captioner = load_captioner()


# --- Helper: Add logo ---
# def add_logo():
#     st.image("helping_files/cellula.jpg", width=400)
#     # st.image("helping_files/cellula.jpg", use_container_width=True)
def add_logo():
    # Load image normally
    logo = "helping_files/cellula.jpg"
    # Center it with columns
    col1, col2, col3 = st.columns([1, 3, 1])  # middle col is wider
    with col2:
        st.image(logo, width=1000)



# --- Pages ---
def login_page():
    add_logo()
    st.markdown(
        "<h1 style='text-align: center; color: #FCFDFE;'>Welcome to Cellula App</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #FCFDFE;'>Classify text & images securely. Please log in or create an account below.</p>", 
        unsafe_allow_html=True
    )

    ##################
    st.markdown(
        """
        <style>
        /* Target the tab labels */
        .stTabs [role="tab"] {
            background-color: transparent;
            color: #FCFDFE;  /* default text color */
            font-weight: 600;
        }

        /* Hover effect */
        .stTabs [role="tab"]:hover {
            color: #014BB6 !important;  /* Cobalt Blue on hover */
        }

        /* Active (selected) tab */
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;  
            font-weight: 700 !important;
            border-bottom: 3px solid #014BB6; /* underline accent */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab1:
        with st.container():
            email = st.text_input("📧 Email", key="login_email")
            password = st.text_input("🔒 Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True):
                user_name = validate_user(email, password)
                if user_name:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = user_name
                    st.session_state["email"] = email
                    st.success(f"🎉 Welcome back, {user_name}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")

    with tab2:
        with st.container():
            name = st.text_input("👤 Full Name", key="signup_name")
            email = st.text_input("📧 Email", key="signup_email")
            password = st.text_input("🔒 Password", type="password", key="signup_pass")

            if st.button("Create Account", use_container_width=True):
                if create_user(email, name, password):
                    st.success("✅ Account created! Please login now.")
                else:
                    st.warning("⚠️ An account with this email already exists.")


def upload_page():
    add_logo()
    st.title(" 📷🔤 Image and Text  Classification")

    st.write(f"Hello, **{st.session_state['username']}** 👋")

    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    user_text = st.text_area("Enter some text")

    if st.button("Submit"):
        image_caption = None
        image_bytes = None

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            image_bytes = uploaded_file.read()
            with open("helping_files/temp_upload.png", "wb") as f:
                f.write(image_bytes)
            image_caption = captioner.caption("helping_files/temp_upload.png")
            st.write(f"📝 **Image Caption:** {image_caption}")

        combined_text = user_text
        if image_caption:
            combined_text = (user_text + " " + image_caption).strip()

        if combined_text:
            class_name, confidence = classify_text(combined_text)
            st.success(f"🔍 Predicted class: **{class_name}** (Confidence: {confidence:.2f})")

            insert_classification(
                user_id=st.session_state["email"],
                text_query=user_text,
                image_caption=image_caption,
                image_bytes=image_bytes,
                class_name=class_name,
                confidence_score=confidence
            )

            st.info("✅ Results saved to database!")


def database_page():
    add_logo()
    st.title("📊 Database Viewer")

    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT rowid, user_id, text_query, image_caption, class, confidence_score FROM toxic_classification", 
        conn
    )

    if df.empty:
        st.warning("⚠️ No entries found in the database.")
        conn.close()
        return

    st.subheader("Summary")
    st.write(f"**Total entries:** {len(df)}")
    st.write("**Class distribution:**")
    st.bar_chart(df["class"].value_counts())

    st.write("**Average confidence per class:**")
    avg_conf = df.groupby("class")["confidence_score"].mean().reset_index()
    st.dataframe(avg_conf)

    st.subheader("Detailed Records")
    st.dataframe(df)

    row_ids = df["rowid"].tolist()
    selected_row = st.selectbox("Select a rowid:", row_ids)

    cur = conn.cursor()
    cur.execute(
        "SELECT text_query, image_caption, class, confidence_score, image_src "
        "FROM toxic_classification WHERE rowid = ?",
        (selected_row,)
    )
    row = cur.fetchone()
    if row:
        text_query, image_caption, class_name, confidence, image_blob = row
        st.write(f"**Text Query:** {text_query}")
        st.write(f"**Image Caption:** {image_caption}")
        st.write(f"**Class:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}")

        image_slot = st.empty()
        if image_blob:
            image = Image.open(io.BytesIO(image_blob))
            image_slot.image(image, caption="Stored Image", use_container_width=True)
        else:
            image_slot.info("No image stored for this row.")

    conn.close()


def admin_delete_page():
    add_logo()
    st.title("🗑️ Admin Delete Page")

    if st.session_state["email"] != "abdo@gmail.com":
        st.error("⛔ You are not authorized to access this page.")
        return

    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT rowid, user_id, text_query, image_caption, class, confidence_score FROM toxic_classification", 
        conn
    )

    if df.empty:
        st.warning("⚠️ No entries found in the database.")
        conn.close()
        return

    st.dataframe(df)

    row_ids = df["rowid"].tolist()
    selected_row = st.selectbox("Select a rowid to delete:", row_ids)

    if st.button("Delete Selected Row"):
        cur = conn.cursor()
        cur.execute("DELETE FROM toxic_classification WHERE rowid = ?", (selected_row,))
        conn.commit()
        st.success(f"✅ Row {selected_row} deleted successfully!")
        conn.close()
        st.rerun()

    conn.close()


# --- Main App ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    with st.sidebar:
        st.markdown("### 🌐 Navigation")
        if st.session_state["email"] == "abdo@gmail.com":
            page = option_menu(
                None,
                ["Upload", "Database", "Admin Delete", "Logout"],
                icons=["cloud-upload", "bar-chart", "trash", "box-arrow-right"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "#1A1D2E"},
                    "icon": {"color": "#FCFDFE", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "5px",
                        "color": "#FCFDFE",
                        "--hover-color": "#000219",
                    },
                    "nav-link-selected": {"background-color": "#014BB6", "color": "#FCFDFE"},
                },
            )
        else:
            page = option_menu(
                None,
                ["Upload", "Database", "Logout"],
                icons=["cloud-upload", "bar-chart", "box-arrow-right"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "#1A1D2E"},
                    "icon": {"color": "#FCFDFE", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "5px",
                        "color": "#FCFDFE",
                        "--hover-color": "#000219",
                    },
                    "nav-link-selected": {"background-color": "#014BB6", "color": "#FCFDFE"},
                },
            )

    if page == "Upload":
        upload_page()
    elif page == "Database":
        database_page()
    elif page == "Admin Delete":
        admin_delete_page()
    elif page == "Logout":
        st.session_state["logged_in"] = False
        st.rerun()
else:
    login_page()

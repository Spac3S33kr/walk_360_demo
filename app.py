import streamlit as st
import walking_stick
import cv2

# ---- Page Config ----
st.set_page_config(page_title="Walk 360 Demo", layout="wide")

# ---- Custom CSS for background & styling ----
st.markdown(
    """
    <style>
        /* Gradient background */
        .stApp {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }
        

        /* Title */
        h1 {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #ffffff;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #ff7f50;
            color: white;
            border-radius: 12px;
            padding: 0.6em 1.2em;
            font-size: 1.1em;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Title ----
st.title("🚶‍♂️ Walk 360 Demo")

# ---- Instructions ----
st.markdown(
    """
    👋 Welcome to the **AI Smart Stick Demo**!  
    This demo shows how our stick **detects objects** and **guides the user with voice alerts**.  
    """
)

# ---- Layout with Image + Button ----
col1, col2 = st.columns([1, 1])

with col1:
    st.image("stick.png", caption="Our AI Smart Stick", use_container_width=True)  # <-- your stick image file

with col2:
    st.markdown("### 👉 Try the Demo")
    if st.button("▶️ Run Detection"):
        st.info("Running detection... Please wait ⏳")
        result = walking_stick.main()
        st.success("Demo finished ✅")

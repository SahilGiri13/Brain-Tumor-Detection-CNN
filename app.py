# # ============================================================
# # NEUROSCAN AI
# # Brain Tumor Detection and Classification Using CNN
# # Developed by Gaurav Katre
# # ============================================================

# import os
# import io
# import json
# import csv
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf

# from PIL import Image, ImageOps
# from datetime import datetime
# import matplotlib.pyplot as plt


# # ============================================================
# # 1. PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="NeuroScan AI | Brain MRI Analysis",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # ============================================================
# # 2. PREMIUM PROFESSIONAL WEBSITE THEME
# # ============================================================

# st.markdown("""
# <style>

# /* ==========================================
#    1. MAIN BACKGROUND
# ========================================== */

# .stApp {
#     background:
#         linear-gradient(
#             rgba(240, 248, 243, 0.91),
#             rgba(240, 248, 243, 0.94)
#         ),
#         url("https://images.pexels.com/photos/13176356/pexels-photo-13176356.jpeg?auto=compress&cs=tinysrgb&w=2000");

#     background-size: cover;
#     background-position: center;
#     background-attachment: fixed;
# }

# /* ==========================================
#    2. MAIN LAYOUT
# ========================================== */

# .block-container {
#     max-width: 1250px;
#     padding-top: 2rem;
#     padding-bottom: 3rem;
# }

# /* ==========================================
#    3. GLOBAL TEXT
# ========================================== */

# .stApp {
#     color: #294b3b !important;
# }

# .stApp p,
# .stApp label,
# .stApp li {
#     color: #294b3b !important;
#     line-height: 1.7;
# }

# [data-testid="stMarkdownContainer"] p {
#     color: #294b3b !important;
# }

# /* ==========================================
#    4. HEADINGS
# ========================================== */

# .stApp h1 {
#     color: #123d2c !important;
#     font-size: 36px !important;
#     font-weight: 800 !important;
# }

# .stApp h2 {
#     color: #174c38 !important;
#     font-size: 27px !important;
#     font-weight: 750 !important;
# }

# .stApp h3 {
#     color: #216044 !important;
#     font-size: 20px !important;
#     font-weight: 700 !important;
# }

# .stApp h4 {
#     color: #216044 !important;
#     font-weight: 700 !important;
# }

# /* ==========================================
#    5. CAPTIONS
# ========================================== */

# [data-testid="stCaptionContainer"] p {
#     color: #587263 !important;
#     font-size: 12px !important;
# }

# /* ==========================================
#    6. CONTENT CARDS
# ========================================== */

# [data-testid="stVerticalBlockBorderWrapper"] {
#     background: rgba(255, 255, 255, 0.97) !important;
#     border: 1px solid #c9dfd0 !important;
#     border-radius: 15px !important;
#     padding: 16px !important;
#     box-shadow: 0 4px 15px rgba(25, 75, 45, 0.06);
# }

# /* ==========================================
#    7. PRIMARY BUTTONS
# ========================================== */

# .stButton > button {
#     background: linear-gradient(
#         90deg,
#         #238052,
#         #39a875
#     ) !important;

#     color: #ffffff !important;
#     border: none !important;
#     border-radius: 10px !important;
#     height: 48px !important;
#     font-weight: 700 !important;
# }

# .stButton > button p,
# .stButton > button span {
#     color: #ffffff !important;
# }

# .stButton > button:hover {
#     background: #176b45 !important;
#     border: none !important;
# }

# /* ==========================================
#    8. DOWNLOAD BUTTONS
# ========================================== */

# .stDownloadButton > button {
#     background: #eaf6ee !important;
#     color: #174c38 !important;
#     border: 1px solid #b9d9c4 !important;
#     border-radius: 9px !important;
#     font-weight: 700 !important;
# }

# .stDownloadButton > button p,
# .stDownloadButton > button span {
#     color: #174c38 !important;
# }

# .stDownloadButton > button:hover {
#     background: #d7eddf !important;
# }

# /* ==========================================
#    9. FILE UPLOADER
# ========================================== */

# [data-testid="stFileUploader"] {
#     background: #ffffff !important;
#     border: 1px solid #c8dfd0 !important;
#     border-radius: 14px !important;
#     padding: 15px !important;
# }

# /* Upload drop area */

# [data-testid="stFileUploader"] section {
#     background: #f3faf5 !important;
#     border: 2px dashed #83b796 !important;
#     border-radius: 11px !important;
# }

# /* Upload instructions */

# [data-testid="stFileUploader"] section,
# [data-testid="stFileUploader"] section * {
#     color: #244b35 !important;
# }

# /* Browse button */

# [data-testid="stFileUploader"] button {
#     background: #e4f3e9 !important;
#     color: #174c38 !important;
#     border: 1px solid #9ac7a8 !important;
#     border-radius: 8px !important;
# }

# [data-testid="stFileUploader"] button * {
#     color: #174c38 !important;
# }

# /* Uploaded file name card */

# [data-testid="stFileUploaderFile"] {
#     background: #eaf6ee !important;
#     border: 1px solid #c2dfcb !important;
#     border-radius: 9px !important;
# }

# [data-testid="stFileUploaderFile"] * {
#     color: #174c38 !important;
# }

# /* ==========================================
#    10. METRIC CARDS
# ========================================== */

# [data-testid="stMetric"] {
#     background: #f1f8f3 !important;
#     border: 1px solid #cfe4d5 !important;
#     border-radius: 13px !important;
#     padding: 16px !important;
# }

# [data-testid="stMetricLabel"],
# [data-testid="stMetricValue"],
# [data-testid="stMetricDelta"] {
#     color: #174c38 !important;
# }

# /* ==========================================
#    11. SUCCESS MESSAGE
# ========================================== */

# [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
#     background: #e9f7ed !important;
#     border: 1.5px solid #9acbaa !important;
#     border-radius: 12px !important;
# }

# /* Remove dark inner backgrounds */

# [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) > div {
#     background: transparent !important;
# }

# /* Success text */

# [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) p,
# [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) span {
#     color: #205c39 !important;
# }

# /* ==========================================
#    12. WARNING MESSAGE
# ========================================== */

# [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
#     background: #fff8e8 !important;
#     border: 1.5px solid #e7d49c !important;
#     border-radius: 12px !important;
# }

# [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) > div {
#     background: transparent !important;
# }

# [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) p,
# [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) span {
#     color: #72551e !important;
# }

# /* ==========================================
#    13. INFO MESSAGE
# ========================================== */

# [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
#     background: #edf6ff !important;
#     border: 1.5px solid #b8d5ed !important;
#     border-radius: 12px !important;
# }

# [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) > div {
#     background: transparent !important;
# }

# [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) p,
# [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) span {
#     color: #245477 !important;
# }

# /* ==========================================
#    14. GENERAL ALERT FALLBACK
# ========================================== */

# [data-testid="stAlert"] {
#     border-radius: 12px !important;
# }

# /* ==========================================
#    15. DATA TABLE
# ========================================== */

# [data-testid="stDataFrame"] {
#     background: #ffffff !important;
#     border: 1px solid #cfe3d5 !important;
#     border-radius: 12px !important;
#     overflow: hidden;
# }

# /* ==========================================
#    16. TABS
# ========================================== */

# .stTabs [data-baseweb="tab"] {
#     color: #24543c !important;
#     font-weight: 700 !important;
# }

# .stTabs [aria-selected="true"] {
#     color: #176b45 !important;
#     border-bottom-color: #299765 !important;
# }

# /* ==========================================
#    17. INPUT FIELDS
# ========================================== */

# .stTextInput input,
# .stNumberInput input,
# .stSelectbox [data-baseweb="select"],
# .stTextArea textarea {
#     background: #ffffff !important;
#     color: #174c38 !important;
#     border: 1px solid #c9dfd0 !important;
#     border-radius: 9px !important;
# }

# /* ==========================================
#    18. PROGRESS BAR
# ========================================== */

# .stProgress > div > div > div {
#     background-color: #299765 !important;
# }

# /* ==========================================
#    19. SIDEBAR
# ========================================== */

# [data-testid="stSidebar"] {
#     background: #f5faf6 !important;
#     border-right: 1px solid #d6e7db;
# }

# [data-testid="stSidebar"] p,
# [data-testid="stSidebar"] label {
#     color: #244b35 !important;
# }

# /* ==========================================
#    20. DIVIDERS
# ========================================== */

# hr {
#     border-color: #d2e5d8 !important;
# }

# /* ==========================================
#    21. REMOVE DEFAULT FOOTER
# ========================================== */

# footer {
#     visibility: hidden;
# }

# /* ==========================================
#    22. HEADER
# ========================================== */

# header[data-testid="stHeader"] {
#     background: transparent !important;
# }

# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# # 3. MODEL CONFIGURATION
# # ============================================================

# MODEL_PATHS = [
#     "brain_tumor_model_fixed.keras",
#     os.path.join("models", "brain_tumor_model_fixed.keras")
# ]

# CLASS_NAMES = [
#     "Glioma",
#     "Meningioma",
#     "No Tumor",
#     "Pituitary"
# ]

# IMAGE_SIZE = (224, 224)


# # ============================================================
# # 4. LOAD TRAINED MODEL
# # ============================================================


# @st.cache_resource
# def load_trained_model():

#     for model_path in MODEL_PATHS:

#         if os.path.isfile(model_path):

#             try:
#                 loaded_model = tf.keras.models.load_model(
#                     model_path,
#                     compile=False
#                 )

#                 print("Model loaded successfully:", model_path)

#                 return loaded_model, model_path

#             except Exception as e:
#                 st.error(f"Error loading model: {e}")
#                 raise

#     raise FileNotFoundError(
#         f"Model not found. Checked paths: {MODEL_PATHS}"
#     )


# model, loaded_model_path = load_trained_model()



# # ============================================================
# # 5. IMAGE PREPROCESSING
# # ============================================================

# def preprocess_image(image):

#     image = ImageOps.exif_transpose(image)

#     image = image.convert("RGB")

#     image = image.resize(IMAGE_SIZE)

#     image_array = np.array(
#         image,
#         dtype=np.float32
#     )

#     image_array = image_array / 255.0

#     image_array = np.expand_dims(
#         image_array,
#         axis=0
#     )

#     return image_array


# # ============================================================
# # 6. SESSION STATE
# # ============================================================

# if "prediction_history" not in st.session_state:
#     st.session_state.prediction_history = []

# if "latest_prediction" not in st.session_state:
#     st.session_state.latest_prediction = None


# # ============================================================
# # 7. SIDEBAR
# # ============================================================

# with st.sidebar:

#     st.title("🧠 NeuroScan AI")

#     st.caption("BRAIN MRI CLASSIFICATION PLATFORM")

#     st.markdown("---")

#     st.subheader("Navigation")

#     st.write("🏠 Dashboard")
#     st.write("🖼️ MRI Image Analysis")
#     st.write("📊 Model Performance")
#     st.write("📁 Prediction History")
#     st.write("ℹ️ About Project")

#     st.markdown("---")

#     st.subheader("Model Status")

#     if model is not None:

#         st.success("CNN Model Loaded")

#     else:

#         st.error("Model Not Found")

#     st.caption("Input Resolution: 224 × 224")

#     st.caption("Output Categories: 4")


# # ============================================================
# # 8. HEADER
# # ============================================================

# st.title("🧠 NeuroScan AI")

# st.caption(
#     "INTELLIGENT BRAIN MRI IMAGE CLASSIFICATION PLATFORM"
# )

# st.markdown("---")

# st.subheader("Artificial Intelligence | Deep Learning")

# st.header("Brain MRI Classification")

# st.write(
#     """
#     An educational brain MRI image classification platform
#     powered by Convolutional Neural Networks.

#     Upload an MRI image to explore the model's predicted
#     category and class-wise prediction scores.
#     """
# )

# st.success("● CNN Powered Image Classification Platform")


# # ============================================================
# # 9. PROJECT STATISTICS
# # ============================================================

# st.header("📊 Project Statistics")

# st.write(
#     "An overview of the dataset and deep learning architecture."
# )

# stat1, stat2, stat3, stat4 = st.columns(4)

# with stat1:

#     with st.container():

#         st.metric(
#             "Total MRI Images",
#             "7,200"
#         )

#         st.caption("5,600 Training + 1,600 Testing")

# with stat2:

#     with st.container():

#         st.metric(
#             "Classification Classes",
#             "4"
#         )

#         st.caption("Four MRI image categories")

# with stat3:

#     with st.container():

#         st.metric(
#             "CNN Architecture",
#             "3 Layers"
#         )

#         st.caption("32, 64 and 128 filters")

# with stat4:

#     with st.container():

#         st.metric(
#             "Input Resolution",
#             "224 × 224"
#         )

#         st.caption("RGB Image Processing")


# # ============================================================
# # 10. PLATFORM OVERVIEW
# # ============================================================

# st.header("🚀 Platform Overview")

# col1, col2, col3, col4 = st.columns(4, gap="medium")

# with col1:

#     with st.container():

#         st.subheader("🧠 CNN Model")

#         st.write(
#             "Deep learning architecture for brain MRI classification."
#         )

# with col2:

#     with st.container():

#         st.subheader("🖼️ Image Processing")

#         st.write(
#             "Image resizing, RGB conversion and normalization."
#         )

# with col3:

#     with st.container():

#         st.subheader("⚡ AI Prediction")

#         st.write(
#             "Classification across four image categories."
#         )

# with col4:

#     with st.container():

#         st.subheader("📊 Model Scores")

#         st.write(
#             "Visual representation of class-wise prediction scores."
#         )


# # ============================================================
# # 11. MRI IMAGE ANALYSIS
# # ============================================================

# st.markdown("---")

# st.header("🔬 MRI Image Analysis")

# upload_col, info_col = st.columns(
#     [1.25, 0.75],
#     gap="large"
# )

# uploaded_file = None
# image = None

# with upload_col:

#     with st.container():

#         st.subheader("📤 Upload Brain MRI Image")

#         uploaded_file = st.file_uploader(
#             "Choose an MRI image",
#             type=["jpg", "jpeg", "png"],
#             help="Upload a JPG, JPEG or PNG image."
#         )

#         if uploaded_file is not None:

#             try:

#                 image = Image.open(
#                     io.BytesIO(uploaded_file.getvalue())
#                 )

#                 image = ImageOps.exif_transpose(image)

#                 st.image(
#                     image,
#                     caption="Uploaded MRI Image",
#                     use_container_width=True
#                 )

#                 st.caption(
#                     f"Image dimensions: {image.width} × {image.height}"
#                 )

#             except Exception as e:

#                 st.error(f"Unable to open image: {e}")

#         else:

#             st.info(
#                 "Upload a brain MRI image to see its preview."
#             )


# with info_col:

#     with st.container():

#         st.subheader("⚙️ Model Information")

#         st.write("**Architecture:** CNN")

#         st.write("**Convolutional Layers:** 3")

#         st.write("**Filters:** 32, 64, 128")

#         st.write("**Input Size:** 224 × 224")

#         st.write("**Channels:** RGB")

#         st.write("**Output Classes:** 4")

#         st.write("**Normalization:** 0–1")

#         st.markdown("---")

#         st.subheader("Classification Categories")

#         for name in CLASS_NAMES:

#             st.write(f"🟢 {name}")


# # ============================================================
# # 12. PREDICTION
# # ============================================================

# if model is None:

#     st.error(
#         "Trained model not found. Place "
#         "'brain_tumor_model.keras' in the project folder "
#         "or inside the models folder."
#     )

# elif uploaded_file is not None and image is not None:

#     if st.button(
#         "🚀 ANALYZE MRI IMAGE",
#         use_container_width=True
#     ):

#         try:

#             with st.spinner("Analyzing MRI image..."):

#                 processed_image = preprocess_image(image)

#                 predictions = model.predict(
#                     processed_image,
#                     verbose=0
#                 )

#                 if (
#                     predictions.ndim != 2
#                     or predictions.shape[1] != len(CLASS_NAMES)
#                 ):

#                     st.error(
#                         "Model output does not match the four classes."
#                     )

#                     st.stop()

#                 scores = predictions[0]

#                 predicted_index = int(
#                     np.argmax(scores)
#                 )

#                 predicted_class = CLASS_NAMES[predicted_index]

#                 confidence = float(
#                     scores[predicted_index]
#                 ) * 100

#                 result = {
#                     "filename": uploaded_file.name,
#                     "predicted_class": predicted_class,
#                     "confidence": confidence,
#                     "scores": {
#                         CLASS_NAMES[i]: float(scores[i]) * 100
#                         for i in range(len(CLASS_NAMES))
#                     },
#                     "time": datetime.now().strftime(
#                         "%Y-%m-%d %H:%M:%S"
#                     )
#                 }

#                 st.session_state.latest_prediction = result

#                 st.session_state.prediction_history.append(
#                     result
#                 )

#         except Exception as e:

#             st.error(f"Prediction error: {e}")


# # ============================================================
# # 13. DISPLAY PREDICTION RESULTS
# # ============================================================

# result = st.session_state.latest_prediction

# if result is not None:

#     st.markdown("---")

#     st.header("🎯 AI Classification Results")

#     result_col, score_col = st.columns(
#         [1, 1],
#         gap="large"
#     )

#     with result_col:

#         with st.container():

#             st.subheader("Predicted Category")

#             st.title(
#                 result["predicted_class"]
#             )

#             st.metric(
#                 "Prediction Score",
#                 f'{result["confidence"]:.2f}%'
#             )

#             st.caption(
#                 "This score represents the model's output, "
#                 "not a medically validated probability."
#             )

#     with score_col:

#         with st.container():

#             st.subheader("📊 Class-wise Prediction Scores")

#             score_data = pd.DataFrame({
#                 "Category": list(result["scores"].keys()),
#                 "Score (%)": list(result["scores"].values())
#             })

#             st.bar_chart(
#                 score_data.set_index("Category")
                
#             )

#     # Individual class scores

#     st.subheader("Detailed Prediction Scores")

#     score_cols = st.columns(4)

#     for i, class_name in enumerate(CLASS_NAMES):

#         with score_cols[i]:

#             st.metric(
#                 class_name,
#                 f'{result["scores"][class_name]:.2f}%'
#             )

#     # DOWNLOAD REPORT

#     report_text = f"""
# NEUROSCAN AI - MRI CLASSIFICATION REPORT

# Filename: {result["filename"]}

# Prediction Date: {result["time"]}

# Predicted Category: {result["predicted_class"]}

# Prediction Score: {result["confidence"]:.2f}%

# CLASS-WISE SCORES
# """

#     for name, score in result["scores"].items():

#         report_text += f"\n{name}: {score:.2f}%"

#     report_text += """

# IMPORTANT DISCLAIMER:

# This is an educational deep learning project.
# It is not a clinically validated diagnostic system.
# Do not use these predictions for medical diagnosis.
# Consult a qualified medical professional.
# """

#     st.download_button(
#         label="📥 Download Prediction Report",
#         data=report_text,
#         file_name="NeuroScan_AI_Report.txt",
#         mime="text/plain",
#         use_container_width=True
#     )

#     st.success("Image classification completed.")

#     st.warning(
#         "Educational project only. This model is not a "
#         "clinically validated diagnostic system. Its predictions "
#         "must not be used to diagnose or rule out a brain tumor. "
#         "Consult a qualified medical professional."
#     )


# # ============================================================
# # 15. PREDICTION HISTORY
# # ============================================================

# st.markdown("---")

# st.header("📁 Prediction History")

# history = st.session_state.prediction_history

# if history:

#     history_df = pd.DataFrame([
#         {
#             "Image": item["filename"],
#             "Prediction": item["predicted_class"],
#             "Score (%)": round(item["confidence"], 2),
#             "Date": item["time"]
#         }
#         for item in history
#     ])

#     st.dataframe(
#         history_df,
#         use_container_width=True,
#         hide_index=True
#     )

#     csv_data = history_df.to_csv(index=False)

#     st.download_button(
#         "📥 Download Prediction History",
#         data=csv_data,
#         file_name="NeuroScan_Prediction_History.csv",
#         mime="text/csv"
#     )

#     if st.button("Clear Prediction History"):

#         st.session_state.prediction_history = []

#         st.session_state.latest_prediction = None

#         st.rerun()

# else:

#     st.info(
#         "Your prediction history will appear here "
#         "after you analyze an MRI image."
#     )


# # ============================================================
# # 16. ABOUT THE PROJECT
# # ============================================================

# st.markdown("---")

# st.header("ℹ️ About NeuroScan AI")

# about_col1, about_col2 = st.columns(2)

# with about_col1:

#     with st.container():

#         st.subheader("Project Overview")

#         st.write(
#             """
#             NeuroScan AI is an educational deep learning project
#             that uses a Convolutional Neural Network to classify
#             brain MRI images into four categories.

#             The application demonstrates image preprocessing,
#             CNN-based classification and model prediction
#             visualization.
#             """
#         )

# with about_col2:

#     with st.container():

#         st.subheader("Technologies Used")

#         st.write("🐍 Python")

#         st.write("🧠 TensorFlow / Keras")

#         st.write("📊 NumPy and Pandas")
        
#         st.write("🌐 Streamlit")

#         st.write("📈 Matplotlib")


# # ============================================================
# # FOOTER
# # ============================================================


# st.markdown("---")

# st.markdown(
#     """
#     <div style="
#         background-color: white;
#         border: 2px solid green;
#         border-radius: 15px;
#         padding: 30px;
#         text-align: center;
#     ">
#         <h2 style="color: #174c38;">🧠 NeuroScan AI</h2>
#         <p style="color: #294b39;">Brain Tumor Classification Using CNN</p>
#         <p style="color: #174c38;"><b>Developed by Gaurav Katre</b></p>
#         <p style="color: #587263;">Machine Learning & Deep Learning Project</p>
#         <p style="color: #587263;">Educational and Research Purposes Only</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )



# ============================================================
# NEUROSCAN AI
# Brain Tumor Detection and Classification Using CNN
# Modern Healthcare AI Dashboard
# ============================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS - MODERN UI
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #081426, #10243D);
    color: #EAF4FF;
}

/* Main content */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Text */
.stApp p, .stApp label, .stApp li {
    color: #B9CADC;
}

/* Headings */
h1, h2, h3, h4 {
    color: #F0F7FF !important;
    font-weight: 750 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0B192B;
    border-right: 1px solid #233B56;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #C7D8EA !important;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #142942;
    padding: 12px;
    border-radius: 9px;
    margin-bottom: 7px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #152C46, #193752);
    border: 1px solid #2A4C68;
    border-radius: 15px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #9DB8D0 !important;
}

[data-testid="stMetricValue"] {
    color: #53D5E8 !important;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #132942;
    border: 1px solid #284762;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #087EAB, #13B8C7);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #09668C, #0B929F);
    border: none;
}

/* Download buttons */
.stDownloadButton > button {
    background: #193A56;
    color: #55D9E9 !important;
    border: 1px solid #267A91;
    border-radius: 9px;
    font-weight: 650;
}

.stDownloadButton > button:hover {
    background: #20516D;
    border: 1px solid #55D9E9;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #142B44;
    border: 1px dashed #29AFC5;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background: #193651;
    border: 1px dashed #42C4D7;
    border-radius: 10px;
}

[data-testid="stFileUploader"] section * {
    color: #D9F4FF !important;
}

/* Input fields */
.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background: #142B44;
    color: white;
    border: 1px solid #31536F;
    border-radius: 9px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #B9D4E9;
}

.stTabs [aria-selected="true"] {
    color: #55D9E9 !important;
    border-bottom-color: #55D9E9 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #284762;
    border-radius: 12px;
}

/* Progress */
.stProgress > div > div > div {
    background: #21BDD0;
}

/* Divider */
hr {
    border-color: #29445F;
}

/* Header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Custom hero */
.hero {
    background: linear-gradient(120deg, #123B5C, #087F9D);
    padding: 32px;
    border-radius: 20px;
    border: 1px solid #2E8BA7;
    margin-bottom: 25px;
}

.hero-title {
    color: white;
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #D4F4FA;
    font-size: 15px;
}

/* Section labels */
.section-label {
    color: #55D9E9;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
}

/* Information cards */
.info-card {
    background: #142B44;
    border: 1px solid #294B68;
    border-radius: 15px;
    padding: 20px;
    min-height: 145px;
}

.info-card h3 {
    color: #58D7E8;
    font-size: 19px;
}

.info-card p {
    color: #B6CADD;
    font-size: 14px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #123B55, #14536A);
    border: 1px solid #2C91A5;
    border-radius: 16px;
    padding: 25px;
}

.footer {
    text-align: center;
    color: #829BB2;
    padding: 25px;
    border-top: 1px solid #29445F;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = [
    "brain_tumor_model_fixed.keras",
    os.path.join("models", "brain_tumor_model_fixed.keras")
]

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMAGE_SIZE = (224, 224)


# ============================================================
# 4. LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    for path in MODEL_PATHS:

        if os.path.isfile(path):

            loaded_model = tf.keras.models.load_model(
                path,
                compile=False
            )

            return loaded_model, path

    return None, None


try:
    model, loaded_model_path = load_trained_model()

except Exception as e:
    model = None
    loaded_model_path = None
    st.error(f"Model loading error: {e}")


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 6. SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#55D9E9;'>🧠 NeuroScan AI</h1>",
        unsafe_allow_html=True
    )

    st.caption("INTELLIGENT MRI ANALYSIS PLATFORM")

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🔬 MRI Analysis",
            "📊 Model Information",
            "📁 Prediction History",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.subheader("System Status")

    if model is not None:
        st.success("CNN Model Loaded")
    else:
        st.error("CNN Model Not Found")

    st.caption("Input: 224 × 224 RGB")

    st.caption("Classes: 4")

    st.markdown("---")

    st.caption("NEUROSCAN AI")
    st.caption("Deep Learning Project")



# ============================================================
# 8. COMMON HEADER - NEW NEUROSCAN AI DESIGN
# ============================================================




# import streamlit as st

# # Page configuration
# st.set_page_config(
#     page_title="NeuroScan AI",
#     page_icon="🧠",
#     layout="wide"
# )

# NeuroScan AI Header

st.markdown("""
<p style="color: #BDEFFA;
          font-size: 12px;
          letter-spacing: 2px;
          font-weight: bold;">
    ARTIFICIAL INTELLIGENCE • HEALTHCARE TECHNOLOGY
</p>

<h1 style="color: white; font-size: 38px;">
    🧠 NeuroScan AI
</h1>

<p style="color: #D4F4FA; font-size: 16px;">
    Brain MRI Classification Using Convolutional Neural Networks
</p>

<p style="color: #BDEFFA; font-size: 13px;">
    AI-POWERED BRAIN MRI ANALYSIS
</p>
""", unsafe_allow_html=True)




# ============================================================
# 9. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader("Welcome to NeuroScan AI")

    st.write(
        "An intelligent educational platform for exploring "
        "CNN-based brain MRI image classification."
    )

    st.markdown("### 📊 Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Classification Classes",
            "04"
        )

    with col2:
        st.metric(
            "Input Resolution",
            "224 × 224"
        )

    with col3:
        st.metric(
            "Images Analyzed",
            len(st.session_state.prediction_history)
        )

    with col4:
        st.metric(
            "CNN Model",
            "Active" if model is not None else "Unavailable"
        )

    st.markdown("---")

    st.subheader("🚀 Explore Platform Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 MRI Analysis</h3>
            <p>Upload an MRI image and explore CNN classification results.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Prediction Scores</h3>
            <p>Visualize the model output across four classification categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>📁 History</h3>
            <p>Review previous predictions and export the analysis history.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🧠 Classification Categories")

    category_cols = st.columns(4)

    for i, category in enumerate(CLASS_NAMES):

        with category_cols[i]:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{category}</h3>
                    <p>MRI image classification category</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "Select MRI Analysis from the sidebar to upload an image."
    )


# ============================================================
# 10. MRI ANALYSIS
# ============================================================

elif page == "🔬 MRI Analysis":

    st.subheader("🔬 Brain MRI Image Analysis")

    st.write(
        "Upload a brain MRI image to explore the CNN model's output."
    )

    if model is None:

        st.error(
            "CNN model not found. Please place "
            "brain_tumor_model_fixed.keras in your project folder "
            "or inside the models folder."
        )

    else:

        upload_col, detail_col = st.columns(
            [1.4, 0.8],
            gap="large"
        )

        uploaded_file = None
        image = None

        with upload_col:

            st.markdown("### 📤 Upload MRI Image")

            uploaded_file = st.file_uploader(
                "Choose an MRI image",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG and PNG"
            )

            if uploaded_file is not None:

                try:

                    image = Image.open(
                        io.BytesIO(uploaded_file.getvalue())
                    )

                    image = ImageOps.exif_transpose(image)

                    st.image(
                        image,
                        caption="Uploaded Brain MRI Image",
                        use_container_width=True
                    )

                    st.caption(
                        f"Image size: {image.width} × {image.height}"
                    )

                except Exception as e:

                    st.error(f"Image error: {e}")

                    image = None

            else:

                st.info(
                    "Choose an image to start the analysis."
                )

        with detail_col:

            st.markdown("### ⚙️ Model Configuration")

            st.markdown("""
            <div class="info-card">
                <h3>Model Architecture</h3>
                <p>Convolutional Neural Network</p>
                <h3>Input Shape</h3>
                <p>224 × 224 × 3</p>
                <h3>Image Processing</h3>
                <p>RGB Conversion</p>
                <p>Image Resizing</p>
                <p>Normalization (0–1)</p>
                <h3>Output Classes</h3>
                <p>Four MRI Categories</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if uploaded_file is not None and image is not None:

            if st.button(
                "🚀 ANALYZE MRI IMAGE",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Processing MRI image with CNN model..."
                    ):

                        processed_image = preprocess_image(image)

                        predictions = model.predict(
                            processed_image,
                            verbose=0
                        )

                        if (
                            predictions.ndim != 2
                            or predictions.shape[1] != len(CLASS_NAMES)
                        ):

                            st.error(
                                "Model output does not match "
                                "the four expected classes."
                            )

                            st.stop()

                        scores = predictions[0]

                        # This assumes the model outputs
                        # probabilities (such as softmax).
                        # If it outputs logits, apply the
                        # appropriate activation first.

                        if (
                            not np.all(np.isfinite(scores))
                            or np.any(scores < 0)
                            or np.any(scores > 1)
                        ):

                            st.error(
                                "Model output is not a valid "
                                "probability vector. Check the "
                                "model's final activation."
                            )

                            st.stop()

                        predicted_index = int(
                            np.argmax(scores)
                        )

                        predicted_class = CLASS_NAMES[
                            predicted_index
                        ]

                        confidence = float(
                            scores[predicted_index] * 100
                        )

                        result = {
                            "filename": uploaded_file.name,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "scores": {
                                CLASS_NAMES[i]: float(scores[i]) * 100
                                for i in range(len(CLASS_NAMES))
                            },
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        }

                        st.session_state.latest_prediction = result

                        st.session_state.prediction_history.append(
                            result
                        )

                    st.success("Image analysis completed.")

                except Exception as e:

                    st.error(f"Prediction error: {e}")

        # Display latest prediction
        result = st.session_state.latest_prediction

        if result is not None:

            st.markdown("---")

            st.subheader("🎯 Classification Results")

            result_col, confidence_col = st.columns(2)

            with result_col:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="section-label">
                            PREDICTED CATEGORY
                        </div>
                        <h1 style="color:white;">
                            {result['predicted_class']}
                        </h1>
                        <p style="color:#D4F4FA;">
                            Model output for the uploaded image
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with confidence_col:

                st.metric(
                    "Model Output Score",
                    f"{result['confidence']:.2f}%"
                )

                st.progress(
                    min(max(result["confidence"] / 100, 0), 1)
                )

                st.caption(
                    "This is a model score, not a medically "
                    "validated diagnostic probability."
                )

            st.markdown("### 📊 Class-wise Prediction Scores")

            score_df = pd.DataFrame({
                "Category": list(result["scores"].keys()),
                "Score (%)": list(result["scores"].values())
            })

            st.bar_chart(
                score_df.set_index("Category"),
                horizontal=True
            )

            score_cols = st.columns(4)

            for i, category in enumerate(CLASS_NAMES):

                with score_cols[i]:

                    st.metric(
                        category,
                        f"{result['scores'][category]:.2f}%"
                    )

            # Download report
            report = f"""
NEUROSCAN AI
BRAIN MRI CLASSIFICATION REPORT

Filename: {result['filename']}

Date: {result['time']}

Predicted Category: {result['predicted_class']}

Model Output Score: {result['confidence']:.2f}%

CLASS-WISE SCORES
"""

            for name, score in result["scores"].items():

                report += f"\n{name}: {score:.2f}%"

            report += """

DISCLAIMER:
This is an educational deep learning project.
It is not a clinically validated diagnostic system.
Do not use this output for medical diagnosis.
Consult a qualified medical professional.
"""

            st.download_button(
                "📥 Download MRI Analysis Report",
                data=report,
                file_name="NeuroScan_AI_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.warning(
                "Educational use only. This CNN model is not "
                "clinically validated and cannot diagnose or "
                "rule out a brain tumor."
            )


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.subheader("📊 CNN Model Information")

    st.write(
        "Technical overview of the brain MRI classification model."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Architecture")

        st.markdown("""
        <div class="info-card">
            <h3>🧠 Convolutional Neural Network</h3>
            <p>Deep learning image classification model.</p>
            <p>Input: 224 × 224 × 3</p>
            <p>Convolutional filters: 32, 64, 128</p>
            <p>Output categories: 4</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("### Image Processing")

        st.markdown("""
        <div class="info-card">
            <h3>🖼️ Preprocessing Pipeline</h3>
            <p>1. Load MRI image</p>
            <p>2. Convert to RGB</p>
            <p>3. Resize to 224 × 224</p>
            <p>4. Normalize pixel values</p>
            <p>5. Generate CNN prediction</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Classification Labels")

    for category in CLASS_NAMES:
        st.write(f"🔹 {category}")

    if model is not None:

        st.success("Trained model successfully loaded.")

        st.caption(
            f"Model file: {loaded_model_path}"
        )

        with st.expander("View Model Summary"):

            summary_lines = []

            model.summary(
                print_fn=lambda line: summary_lines.append(line)
            )

            st.code(
                "\n".join(summary_lines),
                language="text"
            )

    else:

        st.error("Model is not currently available.")


# ============================================================
# 12. PREDICTION HISTORY
# ============================================================

elif page == "📁 Prediction History":

    st.subheader("📁 Prediction History")

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame([
            {
                "Image": item["filename"],
                "Prediction": item["predicted_class"],
                "Score (%)": round(item["confidence"], 2),
                "Date": item["time"]
            }
            for item in history
        ])

        st.metric(
            "Total Images Analyzed",
            len(history)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False)

        st.download_button(
            "📥 Download History CSV",
            data=csv_data,
            file_name="NeuroScan_History.csv",
            mime="text/csv"
        )

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.prediction_history = []

            st.session_state.latest_prediction = None

            st.rerun()

    else:

        st.info(
            "No prediction records available. "
            "Analyze an MRI image to create your first record."
        )


# ============================================================
# 13. ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About NeuroScan AI")

    st.markdown("""
    ### 🧠 Project Overview

    NeuroScan AI is an educational deep learning application
    designed to classify brain MRI images into four categories
    using a Convolutional Neural Network.

    The platform demonstrates the practical use of image
    preprocessing, CNN classification and prediction
    visualization.

    ### 💻 Technologies Used

    - Python
    - TensorFlow
    - Keras
    - Streamlit
    - NumPy
    - Pandas
    - Pillow

    ### 🎯 Project Objectives

    1. Understand CNN-based image classification.
    2. Process brain MRI images.
    3. Generate model predictions.
    4. Visualize class-wise prediction scores.
    5. Maintain prediction history.

    ### Classification Categories

    - Glioma
    - Meningioma
    - No Tumor
    - Pituitary
    """)

    st.info(
        "This project is intended for educational and research "
        "purposes only. It is not a substitute for professional "
        "medical evaluation."
    )


# ============================================================
# 14. FOOTER
# ============================================================


# ============================================================
# NEUROSCAN AI - FOOTER
# ============================================================


st.markdown("---")

st.markdown(
    "<h2 style='color:#55D9E9;'>🧠 NeuroScan AI</h2>"
    "<p style='color:white;'>Brain MRI Classification Using CNN</p>"
    "<p style='color:#BDEFFA;'>Developed by Sahil Giri</p>"
    "<p style='color:#BDEFFA;'>Machine Learning & Deep Learning Project</p>"
    "<p style='color:#BDEFFA;'>Educational and Research Purposes Only</p>",
    unsafe_allow_html=True
)

st.markdown("---")



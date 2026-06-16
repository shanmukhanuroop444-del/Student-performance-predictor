import streamlit as st        # Streamlit is the framework that turns this Python script into a web app
import numpy as np             # NumPy is used to create the input array we feed into the model
import joblib                  # Pickle lets us load the saved model.pkl file back into memory

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
# st.set_page_config() must be the FIRST Streamlit call in the script.
# title     → text shown on the browser tab
# page_icon → emoji shown on the browser tab
# layout    → "centered" keeps everything in a neat middle column
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD THE TRAINED MODEL
# ─────────────────────────────────────────────
# @st.cache_resource tells Streamlit: "load this once and cache it".
# Without caching, the model would reload from disk on EVERY user interaction,
# which would make the app slow. With caching, it's loaded only once per session.
@st.cache_resource
def load_model():
    # "rb" means "read binary" — pickle files are binary files, not plain text

    model = joblib.load("model.pkl")   # Deserializes the model object back from disk
    return model

model = load_model()   # Call the function; the returned model object is stored here

# ─────────────────────────────────────────────
# APP TITLE AND DESCRIPTION
# ─────────────────────────────────────────────
# st.title() renders a large H1 heading at the top of the page
st.title("🎓 Student Performance Predictor")

# st.markdown() renders any Markdown or plain text as formatted content
# The string here is just a short description explaining what the app does
st.markdown("Fill in the student details below and click **Predict** to see if the student is likely to **pass or fail**.")

# st.divider() draws a horizontal line — purely cosmetic, improves readability
st.divider()

# ─────────────────────────────────────────────
# INPUT SECTION — collecting the 8 features
# ─────────────────────────────────────────────
# We use two columns to keep the form compact and side-by-side
# st.columns(2) returns two column objects; we unpack them into col1 and col2
col1, col2 = st.columns(2)

# ── COLUMN 1 ──
with col1:
    # st.subheader() renders a smaller section heading
    st.subheader("📚 Academic Info")

    # st.slider() creates a draggable slider widget
    # Arguments: label, min_value, max_value, default_value (value=)
    # studytime: weekly study time (1=<2hr, 2=2–5hr, 3=5–10hr, 4=>10hr)
    studytime = st.slider("Study Time (1–4)", min_value=1, max_value=4, value=2)

    # absences: number of school absences (0–93 in the original dataset)
    absences = st.slider("Number of Absences", min_value=0, max_value=93, value=5)

    # failures: number of past class failures (0–4)
    failures = st.slider("Past Class Failures (0–4)", min_value=0, max_value=4, value=0)

# ── COLUMN 2 ──
with col2:
    st.subheader("🏠 Support & Activities")

    # st.selectbox() creates a dropdown menu
    # Arguments: label, list of options
    # The user picks "Yes" or "No"; we'll convert these to 1/0 below
    schoolsup = st.selectbox("School Extra Support?", ["Yes", "No"])
    famsup    = st.selectbox("Family Educational Support?", ["Yes", "No"])
    paid      = st.selectbox("Extra Paid Classes?", ["Yes", "No"])
    activities = st.selectbox("Extracurricular Activities?", ["Yes", "No"])
    higher    = st.selectbox("Wants Higher Education?", ["Yes", "No"])

st.divider()

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
# st.button() renders a clickable button and returns True only when clicked
if st.button("🔍 Predict", use_container_width=True):
    # Convert "Yes"/"No" strings → 1/0 integers using a conditional expression.
    # The model was trained on 1/0 encoded values, so we must match that format.
    schoolsup_val  = 1 if schoolsup  == "Yes" else 0
    famsup_val     = 1 if famsup     == "Yes" else 0
    paid_val       = 1 if paid       == "Yes" else 0
    activities_val = 1 if activities == "Yes" else 0
    higher_val     = 1 if higher     == "Yes" else 0

    # Build the feature array in the EXACT same order used during model training:
    # [studytime, absences, failures, schoolsup, famsup, paid, activities, higher]
    # np.array([...]) creates a 1D array; .reshape(1, -1) reshapes it to
    # shape (1, 8) — i.e., 1 sample with 8 features — which is what sklearn expects
    features = np.array([
        studytime,
        absences,
        failures,
        schoolsup_val,
        famsup_val,
        paid_val,
        activities_val,
        higher_val
    ]).reshape(1, -1)

    # model.predict() returns the predicted class: [0] = Fail, [1] = Pass
    # [0] at the end extracts the single value from the returned array
    prediction = model.predict(features)[0]

    # model.predict_proba() returns the probability of each class: [[prob_0, prob_1]]
    # [0][1] gets the probability of class 1 (Pass)
    # Multiply by 100 and round to 2 decimal places for a clean percentage
    probability = round(model.predict_proba(features)[0][1] * 100, 2)

    # ─────────────────────────────────────────
    # DISPLAY THE RESULT
    # ─────────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        # st.success() renders a green "success" box
        st.success(f"✅ The student is likely to **PASS** with a **{probability}%** pass probability.")
    else:
        # st.error() renders a red "error" box
        st.error(f"❌ The student is likely to **FAIL** with only a **{probability}%** pass probability.")

    # st.metric() renders a clean "metric card" — label on top, value below
    # This gives a nice visual summary of the probability score
    st.metric(label="Pass Probability", value=f"{probability}%")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
# unsafe_allow_html=True lets us render raw HTML inside markdown for styling
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "Built with ❤️ using Scikit-learn & Streamlit | UCI Student Performance Dataset"
    "</p>",
    unsafe_allow_html=True
)
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime

# PAGE CONFIG
st.set_page_config(page_title="Submit Feedback")

# ✅ Load credentials from Streamlit secrets
credentials_info = st.secrets["google"]
credentials = service_account.Credentials.from_service_account_info(
    credentials_info, scopes=["https://www.googleapis.com/auth/documents"]
)
# ✅ Google Docs Setup
DOC_ID = "1BstfrTZjjjCVPcpMuLt9K52g1i8xYl2sHyqDP0Kxg-0"  # Replace with your actual Google Doc ID
docs_service = build("docs", "v1", credentials=credentials)
# 🎨 Streamlit UI
st.title("📝 Feedback & Bug Report Page")
st.write("Share your experience, feedback, and report any issues with the app!")
st.markdown("---")
# 🔹 Organizing Contact Info in Columns
st.header("👤 Your Information")
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Your Name", placeholder="Enter your full name").strip()
with col2:
    lab = st.text_input("Your Lab/Association", placeholder="E.g., WashU, Bioinformatics Team").strip()
with col3:
    email = st.text_input("Your Email (Optional)", placeholder="your.email@example.com").strip()
st.markdown("---")
# 🔹 Experience & Feedback (Dividing into Left/Right Columns)
st.header("💬 Share Your Experience")
left_col, right_col = st.columns([2, 1])  # Left takes 2/3, Right takes 1/3
with left_col:
    experience_rating = st.slider("Rate Your Experience (1-5)", min_value=1, max_value=5, value=3)
    # 🔹 Adding Emojis Below Slider
    st.markdown(
        """
        <style>
        .emoji-row {
            display: flex;
            justify-content: space-between;
            margin-top: -20px;
            padding: 0 20px;
        }
        .emoji {
            font-size: 24px;
        }
        </style>
        <div class="emoji-row">
            <span class="emoji">😢</span>
            <span class="emoji">😞</span>
            <span class="emoji">😐</span>
            <span class="emoji">🙂</span>
            <span class="emoji">😃</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    comment = st.text_area("Your General Feedback", placeholder="Write your feedback here...").strip()
with right_col:
    # 🔨 Bug Report (Collapsible)
    with st.expander("🔨 Report a Bug or Issue (Optional)"):
        bug_report = st.text_area("Describe any issues or unexpected behavior...").strip()
    # 🚀 Additional Features (Collapsible)
    with st.expander("🚀 Suggest Additional Features"):
        st.write("Suggest **biological data types, tools, or general features** you would like BioREMix to integrate.")
        additional_features = st.text_area("🔍 Feature Suggestion", placeholder="E.g., 'Integration with Ensembl API'").strip()
st.markdown("---")
# 🔹 Submit Feedback Button (Full Width) - Only One Button Now
submit = st.button("Submit Feedback", use_container_width=True)
# 🔹 Handling Submission
if submit:  # Only triggered when the wide button is clicked
    if comment or bug_report or additional_features:  # Ensure at least one field is filled
        # 🕒 Generate a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        # 📝 Define feedback structure
        separator = "―" * 40  # Creates a separator with exactly 40 dashes
        feedback_parts = [
            ("📅 Submitted on: ", timestamp),
            ("👤 Name: ", name if name else "Anonymous"),
            ("🏛️ Lab: ", lab if lab else "N/A"),
            ("📧 Email: ", email if email else "N/A"),  # Optional email field
            ("⭐ Experience Score: ", f"{experience_rating}/5"),
            ("💬 Feedback: ", comment if comment else "No feedback provided"),
            ("🛠️ Bug Report: ", bug_report if bug_report else "No issues reported"),
            ("🔍 Additional Features: ", additional_features if additional_features else "No suggestions provided"),
            (separator, ""),  # Separator (40 dashes)
        ]
        # 🔢 Compute text insertion and track styling positions
        insert_text = ""
        styling_requests = []
        start_idx = 1  # Start at index 1 in the Google Doc
        for label, value in feedback_parts:
            label_start = start_idx
            label_end = label_start + len(label)  # Ensure the full label is bold
            value_start = label_end  # Start of user input text
            value_end = value_start + len(value)  # End of user input text
            insert_text += f"{label}{value}\n"  # Append label + value with newline
            start_idx += len(label) + len(value) + 1  # Move index forward (+1 for newline)
            # ✅ Apply bold formatting to both labels AND user input text
            if label != separator:  # Ensure separator is NOT bold
                styling_requests.append({
                    "updateTextStyle": {
                        "range": {"startIndex": label_start, "endIndex": value_end},  # ✅ Bolding now covers the entire entry
                        "textStyle": {"bold": True},
                        "fields": "bold",
                    }
                })
        # 📌 Insert text into Google Doc
        insert_request = {
            "insertText": {
                "location": {"index": 1},  # Insert at the top
                "text": insert_text,
            }
        }
        docs_service.documents().batchUpdate(documentId=DOC_ID, body={"requests": [insert_request]}).execute()
        # 🎨 Apply bold styling to everything
        if styling_requests:  # Ensure we don't send an empty request
            docs_service.documents().batchUpdate(documentId=DOC_ID, body={"requests": styling_requests}).execute()
        st.success("✅ Your feedback, bug report, and/or feature suggestions have been submitted to our development team! Refresh tab to submit another comment.")
    else:
        st.error("⚠️ Please enter feedback, report a bug, or suggest a feature before submitting.")

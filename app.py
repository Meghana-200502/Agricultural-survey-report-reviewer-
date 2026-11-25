
import streamlit as st
import pandas as pd

from modules.report_checker import AgriculturalReportChecker
from modules.gemini_service import GeminiService


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Agricultural Survey Report Reviewer",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND REPORT CHECKER
# --------------------------------------------------

@st.cache_resource
def get_report_checker():
    # Train the neural network once and reuse it
    return AgriculturalReportChecker()


@st.cache_resource
def get_gemini_service():
    return GeminiService()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🌾 Agricultural Survey Report Reviewer")

st.write(
    "An AI-assisted system for checking agricultural "
    "statistics and identifying discrepancies in reports."
)

st.divider()


# --------------------------------------------------
# LOAD REFERENCE DATA
# --------------------------------------------------

try:
    checker = get_report_checker()
    data = checker.data

except Exception as e:
    st.error(f"Could not initialize the report checker: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    This system reviews agricultural statistics using:

    - Official reference dataset
    - Python rule-based validation
    - PyTorch Neural Network
    - Gemini AI explanations
    """
)

st.sidebar.divider()

st.sidebar.metric(
    "Confirmed reference records",
    len(data)
)

st.sidebar.caption(
    "Only confirmed records are used for report verification."
)


# --------------------------------------------------
# AGRICULTURAL REPORT INPUT
# --------------------------------------------------

st.header("1. Enter Agricultural Statistics")

st.write(
    "Enter the figures from the agricultural report "
    "you want to verify."
)

crop_options = sorted(data["crop"].dropna().unique().tolist())

with st.form("agricultural_review_form"):

    crop = st.selectbox(
        "Select Crop",
        crop_options
    )

    crop_data = data[data["crop"] == crop]

    season_options = sorted(
        crop_data["season"].dropna().unique().tolist()
    )

    season = st.selectbox(
        "Select Season",
        season_options
    )

    season_data = crop_data[
        crop_data["season"] == season
    ]

    district_options = sorted(
        season_data["district"].dropna().unique().tolist()
    )

    district = st.selectbox(
        "Select District",
        district_options
    )

    district_data = season_data[
        season_data["district"] == district
    ]

    year_options = sorted(
        district_data["year"].dropna().astype(str).unique().tolist()
    )

    year = st.selectbox(
        "Select Agricultural Year",
        year_options
    )

    st.subheader("Reported Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area (hectares)",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    with col2:
        production = st.number_input(
            "Production (tonnes)",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    with col3:
        reported_yield = st.number_input(
            "Yield (tonnes/hectare)",
            min_value=0.01,
            value=1.0,
            step=0.01
        )

    submitted = st.form_submit_button(
        "🔍 Review Agricultural Report",
        use_container_width=True
    )


# --------------------------------------------------
# RUN REPORT CHECKER
# --------------------------------------------------

if submitted:

    claim = {
        "crop": crop,
        "season": season,
        "district": district,
        "year": year,
        "area": area,
        "production": production,
        "yield": reported_yield
    }

    try:
        with st.spinner("Checking agricultural statistics..."):

            result = checker.check_claim(claim)

        st.session_state["review_result"] = result
        st.session_state["review_claim"] = claim
        st.session_state.pop("gemini_explanation", None)

    except Exception as e:
        st.error(f"Report checking failed: {e}")


# --------------------------------------------------
# DISPLAY REVIEW RESULTS
# --------------------------------------------------

if "review_result" in st.session_state:

    result = st.session_state["review_result"]

    st.divider()
    st.header("2. Report Review Results")

    if result["status"] == "NOT_VERIFIABLE":

        st.warning("This report claim could not be verified.")

        st.write(result.get("message", "No matching reference record found."))

    else:

        if result["status"] == "REVIEW_REQUIRED":

            st.error("⚠️ REVIEW REQUIRED")

        else:

            st.success("✅ NO RULE-BASED DISCREPANCY DETECTED")

        # Show machine learning prediction
        st.subheader("Machine Learning Prediction")

        st.info(
            f"Predicted category: {result['ml_prediction']}"
        )

        # Show findings
        st.subheader("Detected Issues")

        findings = result["findings"]

        if findings:

            for finding in findings:

                st.warning(finding["type"])

                col1, col2 = st.columns(2)

                if "reported_value" in finding:

                    col1.metric(
                        "Reported Value",
                        f"{finding['reported_value']:,.2f}"
                    )

                    col2.metric(
                        "Official Value",
                        f"{finding['official_value']:,.2f}"
                    )

                else:

                    col1.metric(
                        "Reported Yield",
                        f"{finding['reported_yield']:.4f}"
                    )

                    col2.metric(
                        "Calculated Yield",
                        f"{finding['calculated_yield']:.4f}"
                    )

                st.write(
                    f"Difference: "
                    f"{finding['difference_percent']:.2f}%"
                )

        else:

            st.success("No discrepancies were detected by the validation rules.")

        # Reference values
        st.subheader("Official Reference Data")

        reference = result["reference"]

        ref_col1, ref_col2, ref_col3 = st.columns(3)

        ref_col1.metric(
            "Official Area",
            f"{reference['area_ha']:,.2f} ha"
        )

        ref_col2.metric(
            "Official Production",
            f"{reference['production_tonnes']:,.2f} tonnes"
        )

        ref_col3.metric(
            "Official Yield",
            f"{reference['yield_ton_per_ha']:.4f}"
        )

        st.metric(
            "Calculated Yield from Report",
            f"{result['calculated_yield']:.4f} tonnes/ha"
        )

        # --------------------------------------------------
        # GEMINI EXPLANATION
        # --------------------------------------------------

        st.divider()
        st.header("3. AI Explanation and Recommendations")

        st.write(
            "Gemini explains the findings generated by the "
            "Python validation system."
        )

        if st.button(
            "✨ Generate AI Explanation",
            use_container_width=True
        ):

            try:
                gemini = get_gemini_service()

                with st.spinner("Gemini is preparing the explanation..."):

                    explanation = gemini.explain_findings(result)

                st.session_state["gemini_explanation"] = explanation

            except Exception as e:

                st.error(
                    "Gemini could not generate an explanation. "
                    "Your numerical review results are still available."
                )

                st.caption(f"Technical details: {e}")

        if "gemini_explanation" in st.session_state:

            st.subheader("Gemini Review")

            st.markdown(
                st.session_state["gemini_explanation"]
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Agricultural Survey Report Reviewer | "
    "Python + PyTorch + Gemini + Streamlit"
)

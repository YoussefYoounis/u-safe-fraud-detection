# app.py

"""

Streamlit application for fraud detection predictions.

Modern, interactive dashboard for enterprise use.

"""



import logging

import streamlit as st

import pandas as pd

import plotly.graph_objects as go

from datetime import datetime

from src.predict import FraudPredictor

from src.config import (

    APP_TITLE, APP_DESCRIPTION, APP_ICON,

    USER_FACING_FEATURES, COLOR_LEGITIMATE, COLOR_FRAUD

)

logger = logging.getLogger(__name__)



# ============== U-SAFE BRAND CONSTANTS (display-only, no logic impact) ==============

USAFE_BRAND_NAME = "U-SAFE"

USAFE_TAGLINE = "Intelligent Real-Time Risk & Fraud Intelligence System"

USAFE_ACCENT_LIGHT = "#00E5A3"   # vibrant teal/cyan (Sarmad)

USAFE_PRIMARY_LIGHT = "#2B1B54"  # deep royal purple/navy (Sarmad)

USAFE_ACCENT_DARK = "#38BDF8"    # glowing cyan/teal



# ============== PAGE CONFIGURATION ==============

st.set_page_config(

    page_title=f"{USAFE_BRAND_NAME} | {APP_TITLE}",

    page_icon=APP_ICON,

    layout="wide",

    initial_sidebar_state="expanded"

)



# Custom CSS — modern iOS-style, dual light/dark theme, glassmorphism

st.markdown("""

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');



    html, body, [class*="css"], .stApp, .stMarkdown, .stButton button, .stTextInput input,

    .stNumberInput input, .stSelectbox, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {

        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Apple Color Emoji", "Inter", sans-serif !important;

    }



    .main { padding: 1.5rem 2rem 3rem 2rem; }



    /* ---------- Shared structural classes ---------- */

    .usafe-hero {

        border-radius: 24px;

        padding: 2.2rem 2.5rem;

        margin-bottom: 1.8rem;

        position: relative;

        overflow: hidden;

    }

    .usafe-badge {

        display: inline-flex;

        align-items: center;

        gap: 0.4rem;

        padding: 0.3rem 0.9rem;

        border-radius: 999px;

        font-size: 0.78rem;

        font-weight: 700;

        letter-spacing: 0.06em;

        text-transform: uppercase;

        margin-bottom: 0.9rem;

    }

    .usafe-hero-title {

        font-size: 2.4rem;

        font-weight: 800;

        margin: 0 0 0.35rem 0;

        letter-spacing: -0.02em;

    }

    .usafe-hero-tagline {

        font-size: 1.05rem;

        font-weight: 500;

        opacity: 0.85;

        margin: 0 0 0.2rem 0;

    }

    .usafe-hero-sub {

        font-size: 0.9rem;

        opacity: 0.65;

        margin: 0;

    }



    .usafe-card, .feature-input-section {

        border-radius: 20px;

        padding: 1.75rem;

        margin-bottom: 1.6rem;

        backdrop-filter: blur(24px);

        -webkit-backdrop-filter: blur(24px);

    }



    .usafe-section-title {

        font-size: 1.25rem;

        font-weight: 700;

        margin: 0.4rem 0 1rem 0;

        display: flex;

        align-items: center;

        gap: 0.5rem;

    }



    .usafe-status-card {

        text-align: center;

        padding: 2.4rem 1.5rem;

        border-radius: 22px;

        position: relative;

        backdrop-filter: blur(20px);

        -webkit-backdrop-filter: blur(20px);

    }

    .usafe-status-emoji { font-size: 2.6rem; display: block; margin-bottom: 0.3rem; }

    .usafe-status-label {

        font-size: 2.1rem;

        font-weight: 800;

        letter-spacing: 0.02em;

        margin: 0.2rem 0;

    }

    .usafe-status-confidence {

        font-size: 1.05rem;

        font-weight: 500;

        opacity: 0.85;

        margin: 0;

    }



    .usafe-pill {

        display: inline-block;

        font-size: 0.78rem;

        font-weight: 600;

        padding: 0.35rem 0.9rem;

        border-radius: 999px;

        margin: 0 0.25rem;

    }



    .usafe-footer {

        border-radius: 20px;

        padding: 1.8rem;

        text-align: center;

        margin-top: 2rem;

        backdrop-filter: blur(20px);

        -webkit-backdrop-filter: blur(20px);

    }

    .usafe-footer-title {

        font-size: 1.15rem;

        font-weight: 700;

        margin-bottom: 0.6rem;

        display: flex;

        align-items: center;

        justify-content: center;

        gap: 0.5rem;

    }

    .usafe-footer-sub { font-size: 0.9rem; margin-bottom: 1rem; opacity: 0.8; }



    div[data-testid="stMetric"] {

        border-radius: 16px;

        padding: 1rem 1.2rem;

        backdrop-filter: blur(16px);

        -webkit-backdrop-filter: blur(16px);

    }



    .stButton button[kind="primary"] {

        border: none;

        border-radius: 14px;

        font-weight: 700;

        letter-spacing: 0.02em;

    }

    .stButton button {

        border-radius: 14px;

        font-weight: 600;

    }



    hr { opacity: 0.15; }



    /* ---------- LIGHT MODE — Sarmad luxury palette ---------- */

    @media (prefers-color-scheme: light) {

        .stApp { background: linear-gradient(180deg, #F8F9FD 0%, #F1EFFA 100%); }

        .usafe-hero {

            background: linear-gradient(135deg, #2B1B54 0%, #3D2874 55%, #2B1B54 100%);

            border: 1px solid rgba(0,229,163,0.25);

            box-shadow: 0 20px 45px rgba(43,27,84,0.25);

        }

        .usafe-hero-title, .usafe-hero-tagline { color: #FFFFFF; }

        .usafe-hero-sub { color: #E4DFFA; }

        .usafe-badge { background: rgba(0,229,163,0.16); color: #00E5A3; border: 1px solid rgba(0,229,163,0.4); }



        .usafe-card, .feature-input-section {

            background: rgba(255,255,255,0.72);

            border: 1px solid rgba(43,27,84,0.08);

            box-shadow: 0 8px 32px rgba(43,27,84,0.08);

            color: #1A1330;

        }

        .usafe-section-title { color: #2B1B54; }



        div[data-testid="stMetric"] {

            background: rgba(255,255,255,0.72);

            border: 1px solid rgba(43,27,84,0.08);

            box-shadow: 0 6px 20px rgba(43,27,84,0.06);

        }



        .usafe-pill {

            background: rgba(43,27,84,0.06);

            color: #2B1B54;

            border: 1px solid rgba(43,27,84,0.12);

        }

        .usafe-pill.accent {

            background: rgba(0,229,163,0.14);

            color: #009e73;

            border: 1px solid rgba(0,229,163,0.3);

        }



        .usafe-footer {

            background: rgba(43,27,84,0.04);

            border: 1px solid rgba(43,27,84,0.08);

            color: #2B1B54;

        }

        .usafe-footer-sub { color: #5B4B85; }



        .stButton button[kind="primary"] {

            background: linear-gradient(135deg, #2B1B54 0%, #00E5A3 130%);

            color: #FFFFFF;

        }

    }



    /* ---------- DARK MODE — obsidian / cyan glow ---------- */

    @media (prefers-color-scheme: dark) {

        .stApp { background: #0F172A; }

        .usafe-hero {

            background: linear-gradient(135deg, #0F172A 0%, #1E293B 65%, #0F172A 100%);

            border: 1px solid rgba(56,189,248,0.25);

            box-shadow: 0 20px 45px rgba(0,0,0,0.5);

        }

        .usafe-hero-title, .usafe-hero-tagline { color: #F1F5F9; }

        .usafe-hero-sub { color: #94A3B8; }

        .usafe-badge { background: rgba(56,189,248,0.14); color: #38BDF8; border: 1px solid rgba(56,189,248,0.4); }



        .usafe-card, .feature-input-section {

            background: rgba(30,41,59,0.6);

            border: 1px solid rgba(56,189,248,0.12);

            box-shadow: 0 8px 32px rgba(0,0,0,0.45);

            color: #E2E8F0;

        }

        .usafe-section-title { color: #38BDF8; }



        div[data-testid="stMetric"] {

            background: rgba(30,41,59,0.6);

            border: 1px solid rgba(56,189,248,0.12);

            box-shadow: 0 6px 20px rgba(0,0,0,0.35);

        }



        .usafe-pill {

            background: rgba(56,189,248,0.1);

            color: #38BDF8;

            border: 1px solid rgba(56,189,248,0.25);

        }

        .usafe-pill.accent {

            background: rgba(56,189,248,0.18);

            color: #7DD3FC;

            border: 1px solid rgba(56,189,248,0.4);

        }



        .usafe-footer {

            background: rgba(30,41,59,0.5);

            border: 1px solid rgba(56,189,248,0.12);

            color: #E2E8F0;

        }

        .usafe-footer-sub { color: #94A3B8; }



        .stButton button[kind="primary"] {

            background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%);

            color: #0F172A;

        }

    }

    </style>

    """, unsafe_allow_html=True)





# ============== INITIALIZE SESSION STATE ==============

@st.cache_resource

def load_predictor():

    """Load predictor (cached to avoid reloading)."""

    try:

        predictor = FraudPredictor()

        return predictor, None

    except Exception as e:

        return None, str(e)





# ============== HELPER FUNCTIONS ==============

def create_gauge_chart(fraud_probability):

    """Create a gauge chart for fraud probability."""

    color = COLOR_FRAUD if fraud_probability >= 0.5 else COLOR_LEGITIMATE

    status = "FRAUD" if fraud_probability >= 0.5 else "LEGITIMATE"



    fig = go.Figure(go.Indicator(

        mode="gauge+number+delta",

        value=fraud_probability * 100,

        title={'text': "Fraud Probability", 'font': {'size': 16, 'color': '#64748B'}},

        number={'font': {'color': color, 'size': 36}},

        delta={'reference': 50, 'suffix': "%", 'font': {'size': 14}},

        gauge={

            'axis': {'range': [0, 100], 'tickcolor': '#64748B', 'tickfont': {'color': '#64748B'}},

            'bar': {'color': color, 'thickness': 0.28},

            'bgcolor': 'rgba(0,0,0,0)',

            'borderwidth': 0,

            'steps': [

                {'range': [0, 20], 'color': "rgba(0,229,163,0.25)"},

                {'range': [20, 40], 'color': "rgba(0,229,163,0.45)"},

                {'range': [40, 60], 'color': "rgba(148,163,184,0.35)"},

                {'range': [60, 80], 'color': "rgba(56,189,248,0.35)"},

                {'range': [80, 100], 'color': "rgba(231,76,60,0.45)"}

            ],

            'threshold': {

                'line': {'color': "#2B1B54", 'width': 3},

                'thickness': 0.8,

                'value': 50

            }

        }

    ))



    fig.update_layout(

        height=300,

        font_size=12,

        paper_bgcolor='rgba(0,0,0,0)',

        plot_bgcolor='rgba(0,0,0,0)',

        margin=dict(l=30, r=30, t=60, b=20),

        font=dict(color='#64748B')

    )

    return fig





def create_feature_importance_chart(driving_factors):

    """Create a horizontal bar chart for feature importance."""

    if not driving_factors:

        return None



    df = pd.DataFrame(driving_factors)



    n = len(df)

    gradient_colors = []

    for i in range(n):

        t = i / max(n - 1, 1)

        r = int(0x2B + t * (0x00 - 0x2B))

        g = int(0x1B + t * (0xE5 - 0x1B))

        b = int(0x54 + t * (0xA3 - 0x54))

        gradient_colors.append(f"rgb({r},{g},{b})")



    fig = go.Figure(go.Bar(

        x=df['importance'],

        y=df['feature'],

        orientation='h',

        marker=dict(

            color=gradient_colors,

            line=dict(color='rgba(56,189,248,0.4)', width=1)

        ),

        text=[f"{val:.2f}" for val in df['importance']],

        textposition='auto',

    ))



    fig.update_layout(

        title=dict(text="Top Driving Factors for Prediction", font=dict(color='#64748B', size=15)),

        xaxis_title="Importance Score",

        yaxis_title="Feature",

        height=300,

        showlegend=False,

        paper_bgcolor='rgba(0,0,0,0)',

        plot_bgcolor='rgba(0,0,0,0)',

        font=dict(color='#64748B'),

        xaxis=dict(gridcolor='rgba(148,163,184,0.2)'),

        yaxis=dict(gridcolor='rgba(148,163,184,0.2)'),

        margin=dict(l=10, r=10, t=50, b=10)

    )



    return fig





def format_prediction_card(result):

    """Create formatted prediction result card."""

    prediction = result['prediction']

    fraud_prob = result['fraud_probability']

    confidence = result['confidence']



    if prediction == "FRAUD":

        card_class = "fraud-card"

        status_color = COLOR_FRAUD

        status_emoji = "🚨"

    else:

        card_class = "legitimate-card"

        status_color = COLOR_LEGITIMATE

        status_emoji = "💎"



    return {

        'status': prediction,

        'emoji': status_emoji,

        'color': status_color,

        'fraud_prob': fraud_prob,

        'confidence': confidence,

        'class': card_class

    }





# ============== SIDEBAR ==============

with st.sidebar:

    st.markdown("### ⚙️ Settings")

    st.divider()



    st.markdown("### ✨ About U-SAFE")

    st.info(

        "U-SAFE uses a LightGBM machine learning model trained on "

        "the IEEE-CIS Fraud Detection dataset to assess whether a credit card "

        "transaction is fraudulent or legitimate in real-time."

    )



    st.markdown("### 📈 Model Performance")

    st.markdown("""

    - **ROC-AUC Score**: ~0.93

    - **Training Samples**: 590,540

    - **Fraud Cases**: 3.5% of data

    - **Model Type**: LightGBM with SMOTE balancing

    """)



    st.markdown("### 🔍 User Guide")

    st.markdown("""

    1. Enter transaction details in the form

    2. Click "Predict" to get instant results

    3. View the probability gauge and key driving factors

    4. Complex features are auto-filled with typical values

    """)



    st.divider()

    st.caption(f"🛡️ U-SAFE • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")





# ============== MAIN APPLICATION ==============

# Header — hero banner

st.markdown(
    """
    <div class="usafe-hero-card">
        <span class="usafe-badge">🛡️ U-SAFE Engine</span>
        <div class="usafe-hero-title">🔐 U-SAFE</div>
        <p class="usafe-hero-tagline">Intelligent Real-Time Risk & Fraud Intelligence System</p>
        <p class="usafe-hero-sub">U-SAFE Engine — Real-time credit card transaction fraud detection powered by advanced Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)



# Load predictor

predictor, load_error = load_predictor()



if load_error:

    st.error(

        f"❌ Failed to load the U-SAFE model:\n\n{load_error}\n\n"

        "Please ensure train.py has been executed successfully and model files exist."

    )

else:

    # ============== INPUT FORM ==============

    st.markdown('<p class="usafe-section-title">📋 Transaction Details</p>', unsafe_allow_html=True)

    st.markdown('<div class="feature-input-section">', unsafe_allow_html=True)



    user_input = {}



    # Create input fields based on user-facing features

    cols = st.columns(2)

    col_idx = 0



    for feature_name, feature_config in USER_FACING_FEATURES.items():

        col = cols[col_idx % 2]



        with col:

            label = feature_config['label']

            description = feature_config.get('description', '')



            if feature_config['type'] == 'number':

                value = st.number_input(

                    label=label,

                    value=feature_config['default'],

                    min_value=feature_config['min'],

                    max_value=feature_config['max'],

                    step=feature_config['step'],

                    help=description

                )

                user_input[feature_name] = value



            elif feature_config['type'] == 'categorical':

                value = st.selectbox(

                    label=label,

                    options=feature_config['options'],

                    index=0,

                    help=description

                )

                user_input[feature_name] = value



        col_idx += 1



    st.markdown('</div>', unsafe_allow_html=True)



    # ============== PREDICTION BUTTON ==============

    st.divider()



    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:

        predict_button = st.button(

            "⚡ Predict",

            width="stretch",

            type="primary"

        )



    with col2:

        reset_button = st.button(

            "🔄 Reset",

            width="stretch"

        )



    if reset_button:

        st.rerun()



    # ============== PREDICTION RESULTS ==============

    if predict_button:

        with st.spinner("⚡ Analyzing transaction..."):

            try:

                # Make prediction

                result = predictor.predict(user_input)



                if not result['success']:

                    st.error("Prediction could not be completed. Check the application logs and try again.")

                else:

                    # Format result

                    formatted = format_prediction_card(result)



                    # Display main prediction

                    st.divider()

                    st.markdown('<p class="usafe-section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)



                    # Status badge

                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:

                        st.markdown(

                            f"<div class='usafe-status-card' style='"

                            f"background: {formatted['color']}14; "

                            f"border: 2px solid {formatted['color']}; "

                            f"box-shadow: 0 0 40px {formatted['color']}55, 0 0 0 1px {formatted['color']}22 inset;'>"

                            f"<span class='usafe-status-emoji'>{formatted['emoji']}</span>"

                            f"<h1 class='usafe-status-label' style='color: {formatted['color']};'>"

                            f"{formatted['status']}</h1>"

                            f"<p class='usafe-status-confidence'>"

                            f"Confidence: {formatted['confidence']*100:.1f}%</p>"

                            f"</div>",

                            unsafe_allow_html=True

                        )



                    st.divider()



                    # Metrics in columns

                    col1, col2, col3 = st.columns(3)



                    with col1:

                        st.metric(

                            "🚨 Fraud Probability",

                            f"{result['fraud_probability']*100:.2f}%",

                            delta=f"{result['fraud_probability']*100 - 50:+.1f}%"

                        )





                    with col2:

                        st.metric(

                            "💎 Legitimate Probability",

                            f"{result['legitimate_probability']*100:.2f}%",

                            delta=f"{result['legitimate_probability']*100 - 50:+.1f}%"

                        )



                    with col3:

                        st.metric(

                            "✨ Model Confidence",

                            f"{result['confidence']*100:.2f}%"

                        )



                    st.divider()



                    # Gauge chart

                    st.markdown('<p class="usafe-section-title">📊 Probability Gauge</p>', unsafe_allow_html=True)

                    gauge_fig = create_gauge_chart(result['fraud_probability'])

                    st.plotly_chart(gauge_fig, width="stretch")



                    # Feature importance

                    st.markdown('<p class="usafe-section-title">🔍 Key Driving Factors</p>', unsafe_allow_html=True)



                    if result['driving_factors']:

                        # Display as table

                        factors_df = pd.DataFrame(result['driving_factors'])

                        factors_df = factors_df[['feature', 'importance', 'value']]

                        factors_df.columns = ['Feature', 'Importance Score', 'Value']



                        st.dataframe(

                            factors_df,

                            width="stretch",

                            hide_index=True

                        )



                        # Chart

                        importance_fig = create_feature_importance_chart(result['driving_factors'])

                        if importance_fig:

                            st.plotly_chart(importance_fig, width="stretch")

                    else:

                        st.info("No driving factors available. Features were auto-filled with defaults.")



                    st.divider()



                    # Input summary

                    st.markdown('<p class="usafe-section-title">📝 Transaction Summary</p>', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)



                    with col1:

                        st.markdown("**User-Provided Features:**")

                        user_summary = pd.DataFrame([

                            {'Feature': k, 'Value': v}

                            for k, v in list(user_input.items())[:6]

                        ])

                        st.dataframe(user_summary, width="stretch", hide_index=True)



                    with col2:

                        st.markdown("**Prediction Metadata:**")

                        metadata = {

                            'Prediction Type': formatted['status'],

                            'Fraud Probability': f"{result['fraud_probability']:.4f}",

                            'Model Confidence': f"{result['confidence']:.4f}",

                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                            'Model Type': 'LightGBM'

                        }

                        metadata_df = pd.DataFrame(list(metadata.items()), columns=['Metric', 'Value'])

                        st.dataframe(metadata_df, width="stretch", hide_index=True)



                    # Success message

                    st.success("✅ Prediction completed successfully!")



            except Exception:

                logger.exception("Unexpected error while rendering a prediction")
                st.error("An unexpected error occurred. Please try again later.")






# ============== FOOTER ==============
st.divider()

st.markdown(f"""
    <div class="usafe-footer">
        <div class="usafe-footer-title">
            🛡️ U-SAFE Platform
        </div>
        <div class="usafe-footer-sub">
            Powered by <b>LightGBM</b> &amp; <b>SMOTE</b> &nbsp;|&nbsp; IEEE-CIS Dataset
        </div>
        <span class="usafe-pill accent">✨ Real-Time Intelligence</span>
        <span class="usafe-pill">📊 LightGBM Core</span>
        <span class="usafe-pill">⚠️ Prototype — Demo Only</span>
    </div>
""", unsafe_allow_html=True)

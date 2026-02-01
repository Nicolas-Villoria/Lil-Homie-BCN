"""
UI Components for Lil Homey - Barcelona Real Estate Valuator

A modern, professional design with full-page background and centered card.
"""
import streamlit as st
import base64
import os
from app.utils.formatting import format_euro

# Get the assets directory path
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def get_base64_image(image_path: str) -> str:
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def render_background_and_styles():
    """Injects full-page background image and custom CSS."""
    
    # Load and encode the background image
    bg_image_path = os.path.join(ASSETS_DIR, "comedor-patio.jpg")
    
    bg_css = ""
    if os.path.exists(bg_image_path):
        img_base64 = get_base64_image(bg_image_path)
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(26, 26, 46, 0.75);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
            z-index: 0;
        }}
        """
    
    st.markdown(f"""
    <style>
        /* Import Playfair Display font */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');

        /* ALL fonts use Playfair Display */
        html, body, [class*="css"], .stMarkdown, .stSelectbox, .stNumberInput, .stCheckbox, 
        .stButton, input, select, textarea, label, p, span, div {{
            font-family: 'Playfair Display', serif !important;
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Background Image */
        {bg_css}

        /* Color Palette */
        :root {{
            --primary: #1a1a2e;
            --secondary: #16213e;
            --gold: #c9a227;
            --text-primary: #1a1a2e;
            --text-secondary: #64748b;
            --card-bg: rgba(255, 255, 255, 0.95);
            --border: rgba(0, 0, 0, 0.08);
        }}

        /* Main Container - Center everything and make it the card */
        .stApp > div > div > div > div.block-container {{
            padding: 2rem 2.5rem !important;
            max-width: 720px !important;
            margin: 2rem auto !important;
            background: var(--card-bg) !important;
            border-radius: 24px !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
            position: relative;
            z-index: 1;
        }}

        /* Form section header - GOLD color */
        .form-section-title {{
            font-family: 'Playfair Display', serif !important;
            font-size: 1.3rem;
            font-weight: 600;
            color: #c9a227 !important;
            margin: 1.5rem 0 1rem 0;
            padding: 0.5rem 0;
        }}

        /* Card Header */
        .card-header {{
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}

        .card-title {{
            font-family: 'Playfair Display', serif !important;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0;
        }}

        .card-title-accent {{
            color: #c9a227;
        }}

        .card-subtitle {{
            font-family: 'Playfair Display', serif !important;
            font-size: 1rem;
            color: white;
            margin-top: 0.5rem;
            color: white !important;
            margin-top: 0.5rem;
        }}

        /* All paragraph elements white */
        .stApp p {{
            color: white !important;
        }}
        

        /* Form Styling */
        .stSelectbox label, .stNumberInput label, .stCheckbox label {{
            font-family: 'Playfair Display', serif !important;
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
        }}

        .stSelectbox > div > div, .stNumberInput > div > div > input {{
            font-family: 'Playfair Display', serif !important;
            border-radius: 12px !important;
            border: 1.5px solid var(--border) !important;
        }}

        .stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within {{
            border-color: #c9a227 !important;
            box-shadow: 0 0 0 3px rgba(201, 162, 39, 0.15) !important;
        }}

        /* Button Styling */
        .stButton > button {{
            font-family: 'Playfair Display', serif !important;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.875rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 14px rgba(26, 26, 46, 0.3) !important;
            margin-top: 1rem !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(26, 26, 46, 0.4) !important;
        }}

        /* Footer */
        .card-footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}

        .footer-badge {{
            font-family: 'Playfair Display', serif !important;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* Checkbox styling */
        .stCheckbox {{
            padding: 0.25rem 0 !important;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .card-title {{
                font-size: 2rem;
            }}
            .stApp > div > div > div > div.block-container {{
                margin: 1rem !important;
                padding: 1.5rem !important;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)


def render_card_header():
    """Renders the header inside the main card."""
    st.markdown("""
    <div class="card-header">
        <h1 class="card-title">Lil <span class="card-title-accent">Homey</span></h1>
        <p class="card-subtitle">AI-powered property valuations for Barcelona</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_title(icon: str, title: str):
    """Renders a form section title with gold color."""
    st.markdown(f'<div class="form-section-title">{icon} {title}</div>', unsafe_allow_html=True)


def render_result_card(prediction: float, range_pct: float = 0.10, features: dict = None):
    """Renders the valuation result using native Streamlit components."""
    if prediction <= 0:
        st.warning("Could not generate a valid prediction. Please check your inputs.")
        return

    lower_bound = prediction * (1 - range_pct)
    upper_bound = prediction * (1 + range_pct)
    
    # Extract feature values
    size = features.get('size', 1) if features else 1
    ppm2 = prediction / size
    neighborhood = features.get('neighborhood', 'N/A') if features else 'N/A'
    property_type = features.get('propertyType', 'flat').title() if features else 'Flat'
    rooms = features.get('rooms', 0) if features else 0
    
    # Use Streamlit success container for the result
    st.markdown("---")
    
    # Main price display
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="
            text-transform: uppercase;
            letter-spacing: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #c9a227;
            margin-bottom: 0.5rem;
        ">Estimated Market Value</div>
        <div style="
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.5rem;
        ">{format_euro(prediction)}</div>
        <div style="
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
        ">{format_euro(lower_bound)} – {format_euro(upper_bound)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats using Streamlit columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Per m²", value=f"€{ppm2:,.0f}")
    
    with col2:
        st.metric(label="Total m²", value=f"{size}")
    
    with col3:
        st.metric(label="Bedrooms", value=f"{rooms}")


def render_card_footer(model_version: str = None):
    """Renders the footer with credits."""
    version_text = f" • Model v{model_version}" if model_version else ""
    st.markdown(f"""
    <div class="card-footer">
        <div class="footer-badge">
            ✓ Powered by 50,000+ property listings{version_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

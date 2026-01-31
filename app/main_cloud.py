"""
Lil Homey - Barcelona Property Valuator

A modern Streamlit application for predicting Barcelona property prices.
Powered by a cloud-deployed ML API.
"""
import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Lil Homey | Barcelona Property Valuator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Import components
from app.components.ui import render_background_and_styles, render_card_header, render_result_card, render_card_footer, render_section_title
from app.services.data_service import get_districts, get_neighborhoods, get_neighborhood_id, get_socio_metrics
from app.services.api_client import get_api_client, check_api_health

# Property types
PROPERTY_TYPES = ['flat', 'penthouse', 'duplex', 'studio', 'chalet', 'countryHouse']


def main():
    """Main application entry point."""
    
    # Render background image and global styles
    render_background_and_styles()
    
    # Card header with logo
    render_card_header()
    
    # === LOCATION SECTION ===
    render_section_title("📍", "Location")
    
    col1, col2 = st.columns(2)
    
    districts = get_districts()
    
    with col1:
        district = st.selectbox("District", options=districts, label_visibility="collapsed", 
                               help="Select the district")
    
    with col2:
        neighborhoods = get_neighborhoods(district) if district else []
        neighborhood = st.selectbox("Neighborhood", options=neighborhoods, label_visibility="collapsed",
                                   help="Select the neighborhood")
    
    # Get socioeconomic data
    neighborhood_id = get_neighborhood_id(neighborhood) if neighborhood else None
    socio_metrics = get_socio_metrics(neighborhood_id, district=district) if neighborhood_id else {'income': 15374, 'density': 200}
    
    # === PROPERTY DETAILS SECTION ===
    render_section_title("🏠", "Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        size = st.number_input("Size (m²)", min_value=15, max_value=1000, value=75)
    
    with col2:
        rooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
    
    with col3:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prop_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)
    
    with col2:
        floor = st.number_input("Floor", min_value=0, max_value=50, value=2, help="0 = Ground floor")
    
    # === AMENITIES SECTION ===
    render_section_title("✨", "Amenities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        has_elevator = st.checkbox("Elevator", value=True)
    
    with col2:
        has_parking = st.checkbox("Parking", value=False)
    
    with col3:
        has_ac = st.checkbox("Air Conditioning", value=False)
    
    st.markdown("---")
    
    # === SUBMIT BUTTON ===
    submitted = st.button("Get Valuation", use_container_width=True)
    
    # === RESULTS SECTION ===
    if submitted:
        # Build features dict
        features = {
            "size": size,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "neighborhood": neighborhood,
            "propertyType": prop_type,
            "district": district,
            "avg_income_index": socio_metrics.get('income', 15374),
            "density_val": socio_metrics.get('density', 200),
            "has_lift": has_elevator,
            "has_parking": has_parking,
            "has_ac": has_ac,
            "floor": floor
        }
        
        # Call the API
        with st.spinner("Analyzing market data..."):
            client = get_api_client()
            response = client.predict(features)
        
        if response and response.is_valid:
            # Display result
            render_result_card(
                prediction=response.predicted_price,
                range_pct=0.10,
                features=features
            )
            
            # Footer with model version
            render_card_footer(model_version=response.model_version)
        else:
            st.error("Could not generate prediction. Please try again.")
    else:
        # Show footer when no prediction yet
        render_card_footer()


if __name__ == "__main__":
    main()

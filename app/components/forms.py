import streamlit as st
from app.services.data_service import get_districts, get_neighborhoods, get_neighborhood_id, get_socio_metrics

PROPERTY_TYPES = ['flat', 'penthouse', 'duplex', 'studio', 'chalet', 'countryHouse']

def render_input_form():
    """
    Renders the property input form.
    Returns:
        dict: A dictionary of features if form is submitted, else None.
    """
    # Section 1: The Basics (Location)
    st.markdown("### Location")
    st.caption("First things first, where is this place?")
    
    districts = get_districts()
    if not districts:
        st.error("Data error: No districts found.")
        return None
        
    col_loc1, col_loc2 = st.columns(2)
    with col_loc1:
        district = st.selectbox("District", options=districts)
    with col_loc2:
        neighborhoods = get_neighborhoods(district)
        neighborhood = st.selectbox("Neighborhood", options=neighborhoods)

    # Get hidden features (pass district for fallback lookup)
    neighborhood_id = get_neighborhood_id(neighborhood)
    socio_metrics = get_socio_metrics(neighborhood_id, district=district)

    st.markdown("---")

    # Section 2: The Specs (Property Details)
    st.markdown("### The Details")
    st.caption("What are we looking at?")

    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Total Area (m²)", min_value=15, max_value=1000, value=75)
        rooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        floor = st.number_input("Floor Level", min_value=0, max_value=50, value=1, help="0 = Ground Floor")
        
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
        prop_type = st.selectbox("Type", options=PROPERTY_TYPES, index=0)
        
        st.write("") # Spacer
        st.write("Amenities")
        has_lift = st.checkbox("Elevator", value=True)
        has_parking = st.checkbox("Parking", value=False)
            
    st.markdown("---")

    # Call to Action
    submit = st.button("Calculate Value", type="primary", use_container_width=True)
    
    if submit:
        return {
            "size": size,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "neighborhood": neighborhood,
            "propertyType": prop_type,
            "district": district,
            "avg_income_index": socio_metrics.get('income', 0),
            "density_val": socio_metrics.get('density', 0),
            # Extras
            "has_lift": has_lift, 
            "has_parking": has_parking,
            "floor": floor
        }
    return None

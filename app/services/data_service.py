import pandas as pd
import streamlit as st
import os

# Get project root directory (two levels up from this file: app/services/ -> app/ -> project_root/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths to data (using absolute paths based on project root)
LOOKUP_PATH = os.path.join(PROJECT_ROOT, "datasets/lookup_tables/idealista_extended.csv")
INCOME_PATH = os.path.join(PROJECT_ROOT, "data_lake/silver/income_clean.csv")
DENSITY_PATH = os.path.join(PROJECT_ROOT, "data_lake/silver/density_clean.csv")

@st.cache_data
def load_lookup_table():
    """Loads the Neighborhood <-> District mapping table."""
    if not os.path.exists(LOOKUP_PATH):
        st.error(f"Lookup table not found at {LOOKUP_PATH}")
        return pd.DataFrame()
    return pd.read_csv(LOOKUP_PATH)

@st.cache_data
def load_socioeconomic_data():
    """
    Loads Income and Density data to create a lookup dictionary for model features.
    Returns:
        dict: {neighborhood_id: {'income': val, 'density': val}}
    """
    data_map = {}
    
    def get_csv_from_path(path):
        """Helper to handle Spark 'directory' CSVs or normal files."""
        if not os.path.exists(path):
            return None
        if os.path.isdir(path):
            # Find the part-*.csv file
            for f in os.listdir(path):
                if f.startswith("part-") and f.endswith(".csv"):
                    return os.path.join(path, f)
            return None
        return path

    # Load Income
    income_file = get_csv_from_path(INCOME_PATH)
    if income_file:
        try:
            df_income = pd.read_csv(income_file)
            # Get latest year per neighborhood
            # Assuming higher year is better. 
            # We need avg_income_index. The silver file has 'income' column. 
            # In exploitation zone, we used 'income' as 'avg_income_index'.
            for _, row in df_income.iterrows():
                n_id = row['neighborhood_id']
                # Simple overwriting with latest/last seen value. 
                # Ideally we pick specific year, but for now latest is fine.
                if n_id not in data_map:
                    data_map[n_id] = {}
                data_map[n_id]['income'] = row['income']
        except Exception as e:
            print(f"Error loading income data: {e}")
            
    # Load Density
    density_file = get_csv_from_path(DENSITY_PATH)
    if density_file:
        try:
            df_density = pd.read_csv(density_file)
            for _, row in df_density.iterrows():
                n_id = row['neighborhood_id']
                if n_id not in data_map:
                    data_map[n_id] = {}
                data_map[n_id]['density'] = row['density_val']
        except Exception as e:
             print(f"Error loading density data: {e}")
            
    return data_map

def get_districts():
    """Returns unique districts."""
    df = load_lookup_table()
    if df.empty: return []
    return sorted(df['district'].unique().tolist())

def get_neighborhoods(district):
    """Returns neighborhoods for a given district."""
    df = load_lookup_table()
    if df.empty: return []
    filtered = df[df['district'] == district]
    return sorted(filtered['neighborhood'].unique().tolist())

def get_neighborhood_id(neighborhood_name):
    """Returns ID for a neighborhood name."""
    df = load_lookup_table()
    if df.empty: return None
    row = df[df['neighborhood'] == neighborhood_name]
    if not row.empty:
        return row.iloc[0]['neighborhood_id']
    return None

def get_district_from_neighborhood(neighborhood_name):
    """Returns district for a neighborhood name."""
    df = load_lookup_table()
    if df.empty: return None
    row = df[df['neighborhood'] == neighborhood_name]
    if not row.empty:
        return row.iloc[0]['district']
    return None


# District-level fallback values (calculated from actual data)
# Used when a neighborhood is missing from socioeconomic data
DISTRICT_FALLBACKS = {
    "Sarrià-Sant Gervasi": {"income": 19510.4, "density": 161.75},
    "Eixample": {"income": 18500.0, "density": 350.0},
    "Gràcia": {"income": 16000.0, "density": 290.0},
    "Les Corts": {"income": 20000.0, "density": 200.0},
    "Sant Martí": {"income": 14000.0, "density": 220.0},
    "Ciutat Vella": {"income": 12000.0, "density": 400.0},
    "Sants-Montjuïc": {"income": 13000.0, "density": 180.0},
    "Horta-Guinardó": {"income": 14500.0, "density": 170.0},
    "Nou Barris": {"income": 11000.0, "density": 250.0},
    "Sant Andreu": {"income": 13500.0, "density": 200.0},
}

# Barcelona city-wide average (ultimate fallback)
BCN_AVERAGE = {"income": 15374.0, "density": 200.0}


def get_socio_metrics(neighborhood_id, district=None):
    """
    Returns income and density for a neighborhood ID.
    Falls back to district average if neighborhood data is missing.
    Falls back to city average if district data is also missing.
    """
    data = load_socioeconomic_data()
    metrics = data.get(neighborhood_id)
    
    if metrics and metrics.get('income', 0) > 0:
        # Found valid neighborhood data
        return {
            'income': metrics.get('income', BCN_AVERAGE['income']),
            'density': metrics.get('density', BCN_AVERAGE['density'])
        }
    
    # Fallback to district average
    if district and district in DISTRICT_FALLBACKS:
        return DISTRICT_FALLBACKS[district]
    
    # Ultimate fallback: Barcelona city average
    return BCN_AVERAGE

"""
Team11C_airflow_setup.py
----------------
Setup script for Apache Airflow environment.

This script initializes Airflow with proper configuration for the BDA project.

Usage:
    python Team11C_airflow_setup.py
"""

import os
import subprocess
import sys

def setup_airflow():
    """
    Initializes Airflow environment for the BDA project.
    
    Steps:
    1. Set AIRFLOW_HOME environment variable
    2. Initialize Airflow database (SQLite by default)
    3. Create admin user
    4. Start Airflow webserver and scheduler
    """
    
    # Set Airflow home to project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    airflow_home = os.path.join(project_root, "airflow")
    
    os.environ['AIRFLOW_HOME'] = airflow_home
    os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = os.path.join(project_root, 'dags')
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    
    print(f"Setting AIRFLOW_HOME to: {airflow_home}")
    print(f"DAGs folder: {os.environ['AIRFLOW__CORE__DAGS_FOLDER']}")
    
    # Create airflow directory if it doesn't exist
    os.makedirs(airflow_home, exist_ok=True)
    
    # Initialize Airflow database
    print("\n[1/2] Initializing Airflow database...")
    subprocess.run([sys.executable, "-m", "airflow", "db", "migrate"], check=True)
    
    # Instructions for starting Airflow
    print("\n[2/2] Setup complete!")
    print("\n" + "="*60)
    print("AIRFLOW SETUP COMPLETE")
    print("="*60)
    
    # Check if running from venv
    venv_path = os.path.join(project_root, 'venv', 'bin')
    using_venv = os.path.exists(venv_path) and os.path.dirname(sys.executable).startswith(venv_path)
    
    print("\nTo start Airflow:")
    
    if using_venv or os.path.exists(venv_path):
        # Virtual environment exists
        print(f"  export PATH={venv_path}:$PATH")
        print(f"  export AIRFLOW_HOME={airflow_home}")
        print(f"  export AIRFLOW__CORE__DAGS_FOLDER={os.environ['AIRFLOW__CORE__DAGS_FOLDER']}")
        print("\n  # Then run:")
        print("  airflow standalone")
        print("\n  # Or simply use the convenience script:")
        print("  ./start_airflow.sh")
        print("\n  ⚠️  IMPORTANT: PATH must include venv/bin for standalone mode to work")
    else:
        # No virtual environment (system Python)
        print(f"  export AIRFLOW_HOME={airflow_home}")
        print(f"  export AIRFLOW__CORE__DAGS_FOLDER={os.environ['AIRFLOW__CORE__DAGS_FOLDER']}")
        print("\n  # Then run:")
        print("  airflow standalone")
        print("\n  # Or use the convenience script:")
        print("  ./start_airflow.sh")
    
    print("\n  # Alternative - Separate services:")
    print("  airflow webserver --port 8080  # (terminal 1)")
    print("  airflow scheduler              # (terminal 2)")
    print("\n  # Access UI at: http://localhost:8080")
    print("  # Standalone mode will display username/password on first run")
    print("="*60)

if __name__ == "__main__":
    setup_airflow()

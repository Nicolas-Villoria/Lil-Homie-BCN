#!/bin/bash
# ==============================================================================
# start_airflow.sh
# ==============================================================================
# Convenience script to start Airflow webserver and scheduler.
#
# Usage:
#   chmod +x start_airflow.sh
#   ./start_airflow.sh
# ==============================================================================

# Set Airflow home to project directory
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# CRITICAL: Add venv/bin to PATH if it exists
# This is required for 'airflow standalone' to find the airflow command in subprocesses
if [ -d "$(pwd)/venv/bin" ]; then
    export PATH="$(pwd)/venv/bin:$PATH"
    echo "Using virtual environment: $(pwd)/venv"
else
    echo "Using system Python (no venv found)"
fi

echo "=========================================="
echo "Starting Apache Airflow"
echo "=========================================="
echo "AIRFLOW_HOME: $AIRFLOW_HOME"
echo "DAGs Folder: $AIRFLOW__CORE__DAGS_FOLDER"
echo ""

# Check if Airflow is initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Airflow not initialized. Running setup..."
    python airflow_setup.py
    echo ""
fi

# Configure Airflow webserver to bind to all interfaces (network accessible)
export AIRFLOW__WEBSERVER__WEB_SERVER_HOST=0.0.0.0
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8080

# Detect local IP for display
if command -v ip &> /dev/null; then
    LOCAL_IP=$(ip route get 8.8.8.8 2>/dev/null | awk -F"src " 'NR==1{split($2,a," ");print a[1]}')
elif command -v ipconfig &> /dev/null; then
    LOCAL_IP=$(ipconfig.exe | grep -A 5 "WSL" | grep "IPv4" | awk '{print $NF}' | tr -d '\r')
else
    LOCAL_IP="<your-ip>"
fi

echo "Starting Airflow in standalone mode..."
echo ""
echo "This will start webserver, scheduler, and triggerer in one process."
echo "Username and password will be displayed on first run."
echo ""
echo "Access Airflow UI at:"
echo "  - Local:  http://localhost:8080"
echo "  - Remote: http://$LOCAL_IP:8080"
echo ""
echo "Share the Remote URL with your team!"
echo ""
echo "Press Ctrl+C to stop Airflow"
echo "=========================================="

# Start Airflow standalone (includes webserver + scheduler + triggerer)
# Use airflow from PATH (either venv or system)
airflow standalone

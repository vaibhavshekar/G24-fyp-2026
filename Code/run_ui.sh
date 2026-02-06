#!/bin/bash
# Run the SITS Forecasting UI

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost

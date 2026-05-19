#!/bin/bash
cd "$(dirname "$0")"

# Try pip3 first, then pip
if command -v pip3 &>/dev/null; then
    pip3 install -r requirements.txt --quiet
elif command -v pip &>/dev/null; then
    pip install -r requirements.txt --quiet
else
    echo "pip not found, trying python -m pip..."
    python3 -m pip install -r requirements.txt --quiet
fi

# Try streamlit directly, then via python3 -m
if command -v streamlit &>/dev/null; then
    streamlit run app.py
else
    python3 -m streamlit run app.py
fi

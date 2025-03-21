#!/bin/bash

echo "Setting up Resume Optimizer environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK resources
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

# Download datasets and process them
echo "Downloading datasets..."
python scripts/dataset/load_data.py
# Use the correct filename
python scripts/dataset/preprocessing.py

echo "Setup complete!"

# Ask user if they want to run models and application
read -p "Do you want to run the models now? (y/n): " RUN_MODELS
if [[ $RUN_MODELS == "y" || $RUN_MODELS == "Y" ]]; then
    # run the naive approach
    echo "Running the naive approach..."
    python scripts/naive/model.py

    # run the traditional approach
    echo "Running the traditional approach..."
    python scripts/traditional/traditional_combined.py

    # Ask if they want to run the app
    read -p "Do you want to run the Streamlit app? (y/n): " RUN_APP
    if [[ $RUN_APP == "y" || $RUN_APP == "Y" ]]; then
        echo "Running the Streamlit app..."
        streamlit run main.py
    fi
fi
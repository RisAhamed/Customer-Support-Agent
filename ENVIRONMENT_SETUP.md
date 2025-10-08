# Environment Setup Guide

## NumPy Version Issue Fix

You're facing a NumPy version compatibility issue. The error occurs because:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

This means your current environment has NumPy 2.x, but pandas and other dependencies were built with NumPy 1.x.

## Solution: Create a Compatible Environment

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment with Python 3.8 and NumPy 1.x
conda create -n customer-support python=3.8 numpy=1.24.0 pandas scikit-learn matplotlib seaborn dill mlflow -y

# Activate the environment
conda activate customer-support

# Install additional dependencies
pip install python-dotenv langchain-core langchain-groq

# Install the project in development mode
cd C:\Users\riswa\Desktop\AI\Customer-Support-Agent
pip install -e .
```

### Option 2: Using pip with venv

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# Install specific NumPy version first
pip install numpy==1.24.0

# Install other dependencies
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

## Running the Project

After creating and activating your environment, you can run any file directly:

```bash
# Run model trainer
python src/components/model_trainer.py

# Run training pipeline
python src/pipelines/train_pipeline.py

# Make predictions
python src/pipelines/prediction_pipeline.py
```

The import issues have been fixed, and all modules should now be accessible from anywhere in the project.
"""
Run Customer Support Agent

This script provides a simple interface to:
1. Set up the right environment
2. Train the model
3. Test predictions

Usage:
    python run.py --train  # Train the model
    python run.py --predict "I need help with my bill"  # Make a prediction
"""

import os
import sys
import argparse
import subprocess

def check_environment():
    """Check if the environment is correctly set up."""
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        
        if major_version >= 2:
            print("\n⚠️ WARNING: You have NumPy 2.x installed, but we need NumPy 1.x")
            print("Please follow the instructions in ENVIRONMENT_SETUP.md to fix this")
            return False
        
        import pandas
        import sklearn
        import dill
        print("✅ Environment looks good!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please follow the instructions in ENVIRONMENT_SETUP.md to fix this")
        return False

def train_model():
    """Train the customer support agent model."""
    if not check_environment():
        return
    
    print("\n🔄 Training model...")
    script_path = os.path.join("src", "pipelines", "train_pipeline.py")
    subprocess.run([sys.executable, script_path])
    print("✅ Training complete!")

def make_prediction(text):
    """Make a prediction using the trained model."""
    if not check_environment():
        return
    
    try:
        # Import here to avoid issues if environment isn't set up
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.pipelines.prediction_pipeline import PredictionPipeline
        
        pipeline = PredictionPipeline()
        result = pipeline.predict(text)
        
        print("\n✨ Prediction Result:")
        print(f"Category: {result['category']}")
        print(f"Summary: {result['summary']}")
        print(f"Entities: {result['entities']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def main():
    """Main function to parse arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="Customer Support Agent CLI")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Text to predict")
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.predict:
        make_prediction(args.predict)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
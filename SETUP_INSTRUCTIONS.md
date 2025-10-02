# Environment Setup Instructions

To fix the NumPy version compatibility issues, please follow these steps:

## Option 1: Using conda (Recommended)

1. Create a new conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate customer-support
```

3. Run the training pipeline:

```bash
python src/pipelines/train_pipeline.py
```

## Option 2: Using pip with specific NumPy version

1. Create a new virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install NumPy 1.24.0 specifically:

```bash
pip install numpy==1.24.0
```

4. Install the other requirements:

```bash
pip install -r requirements.txt
```

5. Run the training pipeline:

```bash
python src/pipelines/train_pipeline.py
```

## Explanation of the Issue

The error occurs because:
- Your system is running NumPy 2.x
- Pandas and other libraries were compiled with NumPy 1.x
- This version mismatch causes compatibility issues

Using either of the methods above will ensure all packages are using a compatible version of NumPy, which should resolve the error.
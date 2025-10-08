from setuptools import find_packages, setup

setup(
    name="customer-support-agent",
    version="0.1.0",
    author="RisAhamed",
    author_email="your.email@example.com",
    description="Customer support agent with ML classification and LLM integration",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "mlflow",
        "dill",
        "langchain-core",
        "langchain-groq",
        "python-dotenv"
    ],
    python_requires=">=3.8",
)
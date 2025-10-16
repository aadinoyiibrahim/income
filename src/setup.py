from setuptools import setup, find_packages

setup(
    name="ml_prediction_package",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        'pandas', 'numpy', 'scikit-learn', 'matplotlib',
        'xgboost', 'optuna', 'shap', 'gensim', 'fastapi', 'uvicorn', 'joblib'
    ],
    author="Abdullahi Adinoyi Ibrahim",
    description="A tool for training and predicting transactions.",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

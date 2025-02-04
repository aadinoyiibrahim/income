from setuptools import setup, find_packages

setup(
    name="ml_prediction_package",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scikit-learn==1.3.1",
        "matplotlib==3.4.3",
        "joblib==1.3.2",
    ],
    author="Abdullahi Adinoyi Ibrahim",
    description="A machine learning package for training and predicting transactions.",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

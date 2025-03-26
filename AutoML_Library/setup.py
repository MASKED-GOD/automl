from setuptools import setup, find_packages

# Safe README loading
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "An advanced AutoML library with dataset visualization and ML algorithms."

setup(
    name="AutoML_Library",
    version="0.1.1",
    author="Arnav Upadhyay",
    author_email="upadhyayarnav2004@gmail.com",
    description="An advanced AutoML library with dataset visualization and ML algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MASKED-GOD/AutoML_Library",
    packages=find_packages(where="."),  # Ensures correct folder structure
    include_package_data=True,           # Ensures non-Python files are included
    install_requires=[
        "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
        "xgboost", "lightgbm", "catboost"
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Pytest"
    ],
    python_requires=">=3.12",  # Ensures compatibility with Python 3.12
)

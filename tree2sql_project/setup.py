from setuptools import setup, find_packages

setup(
    name="tree2sql",
    version="1.0.0",
    description="Convert scikit-learn decision trees to native SQL CASE expressions for accelerated DuckDB inference.",
    author="Tree2SQL Contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "duckdb>=0.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "data": ["ucimlrepo>=0.0.6"],
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0", "ipykernel>=6.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
)

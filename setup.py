from setuptools import setup, find_packages

setup(
    name="apnet",
    version="1.0.0",
    description="APNet: Adaptive Prototype Network Framework for Metric Learning & Classification",
    author="Reo Rioll",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)
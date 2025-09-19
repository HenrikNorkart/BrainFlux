from setuptools import setup, find_packages

setup(
    name="BrainFlux",
    version="0.1.0",
    description="A package for BrainFlux project.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "prettytable",
        "seaborn",
        "matplotlib",
        "pandas",
        "tqdm",
        "python-dotenv",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

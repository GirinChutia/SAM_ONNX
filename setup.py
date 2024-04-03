from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sam_onnx",
    version="0.1",
    author="Girin Chutia",
    author_email="girin.iitm@gmail.com",
    description="A simple package for using SAM with ONNX without pytorch dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/mypackage",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

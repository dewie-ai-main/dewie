from setuptools import setup, find_packages

setup(
    name="dewie",
    version="0.1.0",
    description="Agent-native knowledge retrieval for small models",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.24",
        "pydantic>=2.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
    ],
)

from setuptools import setup, find_packages

setup(
    name="medrag_retriever",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-core", "langchain-huggingface", "qdrant-client",
        "sentence-transformers", "fastembed", "pydantic", "transformers", "torch","bitsandbytes"
    ]
)

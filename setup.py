from setuptools import setup, find_packages

setup(
    name="LightUniLLM",
    version="0.1.0",
    description="LightUniLLM",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ascorblack",
    author_email="a@scorblack.ru",
    url="https://github.com/ascorblack/LightUniLLM",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12'
)

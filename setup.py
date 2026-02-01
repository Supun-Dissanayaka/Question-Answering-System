# If We used "from src.helper import llm_pipeline" in app.py, we need to have setup.py to make src a package.
# If we checked pip list in terminal, src is not a package by default., whenever we create a folder named src, it is not a package until we create setup.py file.
from setuptools import setup, find_packages

setup(
    name= "QUESTIONANSWERING_SYSYTEM",
    author= "Supun_Dissanayaka",
    packages= find_packages(),
    version= "0.0.1",
    install_requires= []
)
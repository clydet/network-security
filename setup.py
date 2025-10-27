'''
Setup script for the project
'''

from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    '''
    Get the requirements from the requirements.txt file
    '''
    requirements: List[str] = []
    try:
        with open("requirements.txt", "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                line = line.strip()
                if line and line != "-e .":
                    requirements.append(line)
    except FileNotFoundError as e:
        print(f"Error getting requirements: {e}")
        return []

    return requirements

setup(
    name="network-security",
    version="0.0.1",
    author="Clyde Tedrick",
    author_email="me@iamcly.de",
    description="Network Security",
    packages=find_packages(),
    install_requires=get_requirements(),
)

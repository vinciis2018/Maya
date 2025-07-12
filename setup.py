from setuptools import setup, find_packages

setup(
    name="hedes",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'requests',
    ],
)

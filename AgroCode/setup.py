from setuptools import setup, find_packages


with open("README.md") as file:
    read_me_description = file.read()

setup(
    packages=find_packages(where="."),
    long_description=read_me_description,
    include_package_data=True
)
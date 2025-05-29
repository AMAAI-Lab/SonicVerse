from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="sonicverse",
    version="1.0.0",
    description="",
    url="https://github.com/amaai-lab/SonicVerse",
    author="Anuradha Chopra",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)

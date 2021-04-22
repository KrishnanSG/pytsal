from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pytsal',
    version='1.0.2',
    packages=find_packages(exclude=['tests', 'tests.*', '.github']),
    install_requires=[],
    url='https://github.com/KrishnanSG/pytsal',
    author='Krishnan S G',
    description="An open source low-code time series analysis library in Python",
    author_email='krishsg525@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="time, series, analysis, tsa, visualization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)

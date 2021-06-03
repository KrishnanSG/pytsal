from setuptools import setup, find_packages

import pytsal

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name='pytsal',
    version=pytsal.__version__,
    packages=find_packages(exclude=['tests', 'tests.*', '.github', 'examples']),
    install_requires=requirements,
    url='https://github.com/KrishnanSG/pytsal',
    author='Krishnan S G',
    description="An easy to use open-source python framework for Time Series analysis, visualization and forecasting along with AutoTS",
    author_email='krishsg525@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="time, series, analysis, tsa, visualization, autots, datascience",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.6'
)

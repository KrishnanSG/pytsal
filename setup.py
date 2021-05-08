from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name='pytsal',
    version='1.1.0',
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
    ],
    python_requires='>=3.6'
)

# pytsal

![logo](https://raw.githubusercontent.com/KrishnanSG/pytsal/master/pytsal-logo.JPG)

[![CodeFactor](https://www.codefactor.io/repository/github/krishnansg/pytsal/badge)](https://www.codefactor.io/repository/github/krishnansg/pytsal) 
[![Downloads](https://pepy.tech/badge/pytsal)](https://pepy.tech/project/pytsal)
[![Downloads](https://pepy.tech/badge/pytsal/month)](https://pepy.tech/project/pytsal)
[![PyPI version](https://badge.fury.io/py/pytsal.svg)](https://pypi.org/project/pytsal)


An easy to use open-source python framework for Time Series analysis, visualization, forecasting along with AutoTS.

## Why was pytsal created?

I was deeply inspired by **[pycaret](https://github.com/pycaret/pycaret)**  library which is an amazing library for **Machine Learning** and I wanted to create a similar library for **Time Series Analysis**.

Therefore the interface and features provided are very similar to pycaret but focused and customized towards **Time Series**.

### What does pytsal mean?

pystal is the abbreviation for **Py**thon **T**ime **S**eries **A**nalysis **L**ibrary


## Overview

![Features](https://raw.githubusercontent.com/pycaret/pycaret/master/pycaret2-features.png)

*[Image source](https://raw.githubusercontent.com/pycaret/pycaret/master/pycaret2-features.png)*


## Features

Checklist of features the library currently offers and plans to offer.

> Convention used below: Feature [status]

- Time series data loaders [partial]
- Time series preprocessing [partial]
- Time series modelling
  - Forecasting
    - Holt Winter [completed]
    - ARIMA [in progress]
    - Facebook Prophet [planned]
  - Classification [planned]
  - Anomaly Detection
    - Brutlag [completed]
- Time series visualization [v1 completed]
- Time series validation [v1 completed]
- AutoTS
  - Forecasting [v1 completed]


## Getting Started


The following instructions will get you a copy of the project and ready for use for your python projects.

### Installation

#### Quick Access
  - Download from PyPi.org
  
    ```bash
    pip install pytsal
    ```
  
#### Developer Style
  - Requires Python version >=3.6
  - Clone this repository using the command:

    ```bash
    git clone https://github.com/KrishnanSG/Nutshell.git
    cd Nutshell
    ```
  - Then install the library using the command:

    ```bash
    python setup.py install
    ```

### Examples & Tutorials

The tutorial on how to the library can be found under the [examples folder](https://github.com/KrishnanSG/pytsal/tree/master/examples)

The tutorials clearly explain how to use the library and also provide basic guide to understand time series analysis.

- [Forecasting tutorial](https://github.com/KrishnanSG/pytsal/blob/master/examples/101_forecasting.ipynb)
- [Anomaly detection tutorial](https://github.com/KrishnanSG/pytsal/blob/master/examples/101_anomaly_detection.ipynb)


## Stability

The library isn't mature or stable for production use yet. 

The best of the library currently would be for **non production use and rapid prototyping**.


## Current Contributors
<a href="https://github.com/KrishnanSG/pytsal/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=KrishnanSG/pytsal" />
</a>

*Made with [contributors-img](https://contrib.rocks).*

## Contribution

Contributions are always welcomed, it would be great to have people use and contribute to this project to help users understand and benefit from the library.

### How to contribute
- **Create an issue:** If you have a new feature in mind, feel free to open an issue and add some short description on what that feature could be.
- **Create a PR**: If you have a bug fix, enhancement or new feature addition, create a Pull Request and the maintainers of the repo, would review and merge them.

### What can be contributed?
- Datasets
- Source code enhancement
- Documentation

## Author

* **Krishnan S G** - [@KrishnanSG](https://github.com/KrishnanSG)
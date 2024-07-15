# POGEMA Toolbox

[![Downloads](https://static.pepy.tech/badge/pogema-toolbox)](https://pepy.tech/project/pogema-toolbox)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
![PyPI](https://img.shields.io/pypi/v/pogema-toolbox?color=blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XcLr-EmcgctKta3H1-zac_mnPqmG4Xxj?usp=sharing)


The POGEMA Toolbox is a comprehensive framework designed to facilitate the testing of learning-based approaches within the POGEMA environment. This toolbox offers a unified interface that enables the seamless execution of any learnable MAPF algorithm in POGEMA. 

- Firstly, the toolbox provides robust management tools for custom maps, allowing users to register and utilize these maps effectively within POGEMA. 
- Secondly, it enables the concurrent execution of multiple testing instances across various algorithms in a distributed manner, leveraging Dask for scalable processing. The results from these instances are then aggregated for analysis. 
- Lastly, the toolbox includes visualization capabilities, offering a convenient method to graphically represent aggregated results through detailed plots.
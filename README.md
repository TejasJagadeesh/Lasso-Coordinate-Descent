# Lasso-Coordinate-Descent

In these examples, the LASSO problem with L1-Regularization is solved using Cyclic and Random Coordinate Descent Algorithms.

The general form of the algorithm is:

!(https://github.com/TejasJagadeesh/Lasso-Coordinate-Descent/blob/master/img/algo.jpg)

# Examples:
There are three examples in this repository:
1) Demo using Real Dataset
2) Demo using Simulated Dataset
3) Comparison of my algorithm with Sklearn

# Datasets:
Two datasets are used:

1) Hitters Dataset
It contains Major League Baseball Data from the 1986 and 1987 seasons
Link: https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv)

2) Simulated Dataset

# Structure:
The files for each of the examples are located in the /src folder. Overall structure is:
|- README.md
  |- src/
     |- demo_simulated_dataset.py
     |- demo_real_dataset.py
     |- compare_sklearn_mylasso.py
  |- img/
     |- algo.jpg

# Execution:
To execute the examples, you can download the three .py files in /src folder and use the following python commands to run:
1) python demo_simulated_dataset.py
2) python demo_real_dataset.py
3) python compare_sklearn_mylasso.py

# Required Packages:
1) Numpy
2) Pandas
3) Sklearn
4) Matplotlib

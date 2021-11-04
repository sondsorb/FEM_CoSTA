# FEM_CoSTA

This is the repo for my preproject, which is about combining neural networks with FEM to solve the heat equation.


### Notation used in the code
Np: number of nodes in the triangulization
p: degree of basis polynomials, usually p=1 is used, i.e. linear interpolation
Ne: number of elements (so N = MP+1)


## File description

 - FEM.py : finite element source code. Includes a class "Heat" for solving the heat equation using backwards euler method. Heat.step() takes one temporal step, using an optional correction, this is used in the CoSTA approach.

 - functions.py : manufactured solutions to test the models on

 - isoFEM.py : currently empty, may in the future be an isogeometric version of FEM.py

 - plotsol.py : plot solutions from function.py for visualization

 - quadrature.py : Source code for numerical (gaussian) integration, used in FEM.py

 - replot.sh : Shell script for running testing.py in several configurations (to create plots of solutions)

 - solvers.py : Deep learning methods for solving the heat equation. Methods are DNN, LSTM, and CoSTAs with those networks. The class "Solvers" is used for training and testing these models.

 - testing.ipynb : Notebook version of testing.py

 - testing.py : script for testing the methods, with chosen configurations.

 - tests.py : Tests to see that the code works as intended. Not many tests have been implemented, mostly for the FEM code.

 - tuner.py : script for finding optimal configurations.

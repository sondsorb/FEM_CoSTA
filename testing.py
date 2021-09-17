import numpy as np
import quadrature
import FEM
import solvers
from matplotlib import pyplot as plt

model = solvers.Costa()

model.train(True)

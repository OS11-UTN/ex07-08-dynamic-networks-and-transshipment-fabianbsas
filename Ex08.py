#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabian
"""
import numpy
import sys
from Utils import transform_NN_to_NA
from scipy.optimize import linprog


# Array with node's name
node_name = ["Plant 1 product A", "Plant 1 product B", 
             "Plant 2 product A", "Plant 2 product B", 
             "Plant 3 product A", "Plant 3 product B", 
             "Stock plant 1 product A", "Stock plant 1 product B", 
             "Stock plant 2 product A", "Stock plant 2 product B", 
             "Sales point 1 product A", "Sales point 1 productB", 
             "Sales point 2 productA", "Sales point 2 productB", 
             "Sales point 3 productA", "Sales point 3 productB"]


# This matrix represent a Nodo-Nodo graph
matrix_node_node = numpy.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# Transform Node-Node matrix to Node-Archs
# Aeq matrix Node-Archs
# arc_idxs list of archs
matrix_node_arch, arc_idxs = transform_NN_to_NA(matrix_node_node)


# We need to create 2 different Node-Arch matrices , one will represent the 
# plants ant the other one will represent the sales point

# This matrix represent the plant production
# The nodes belong to stock and sales will be set to 0
matrix_node_arch_ub = matrix_node_arch.copy()
matrix_node_arch_ub[6:, :] = 0 
bub = numpy.array([20, 30, 10, 40, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# This matrix represent the sales point demand
# The nodes belong to the plant will be set to 0
matrix_node_arch_eq = matrix_node_arch.copy()
matrix_node_arch_eq[:6, :] = 0
beq = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -30, -40, -10, -20, -20, -20])

# Cost vector 
cost_vector = numpy.array([100, 100, 200, 200, 150, 150, 150, 150, 200, 200, 100,
                           100, 100, 150, 200, 200, 150, 100, 100, 150, 200, 200, 150, 100])


bounds = tuple([(0, None) for arcs in range(0, matrix_node_arch.shape[1])])

if len(cost_vector) != len(arc_idxs):
    print("The quantity of arches and the cost vector don't match")
    sys.exit()
    
# check if the arch has the correct cost    
for index, arch in enumerate(arc_idxs):
    print("The arch: {} has a cost of: {}".format(arch, cost_vector[index]))



# Optimize
result = linprog(cost_vector, 
                 A_eq=matrix_node_arch_eq, b_eq=beq, 
                 A_ub=matrix_node_arch_ub, b_ub=bub, 
                 bounds=bounds, 
                 method="simplex")



print("\n\n## Results ## \n\n")


for i in range(len(arc_idxs)):
    print("The optimun distribution for {} is {}".format(arc_idxs[i], result.x[i].astype(int)))
    








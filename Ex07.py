#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:24:43 2020

@author: fabian
"""
import numpy
import sys
from Utils import transform_NN_to_NA
from scipy.optimize import linprog

# This matrix represent a Nodo-Nodo graph
matrix_node_node = numpy.array([[ 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #s0
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #s1
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #s2
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #s3
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #s4
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #s5
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #s6
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #a1
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], #a2
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], #a3
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], #a4
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #a5
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #a6
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #b1
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #b2
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #b3
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #b4
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #b5
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #b6
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t1
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t2
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t3
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t4
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t5
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #t6
                                [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])#t0)


# Add a new arch from node t to s
# The capacity of this new arch will represent the maximum flow 
number_nodes = len(matrix_node_node)
matrix_node_node[number_nodes-1][0] = 1

# Transform Node-Node matrix to Node-Archs
# Aeq matrix Node-Archs
# arc_idxs list of archs
Aeq, arc_idxs = transform_NN_to_NA(matrix_node_node)
 
# Cost vector 
cost_vector = numpy.zeros(len(arc_idxs))

# the cost of the new arch that connects t to s is -1
cost_vector[-1] = -1

# Demand-Supply vector
beq = numpy.zeros(number_nodes)

# Capacity vector
max_capacity = numpy.array([None, None, None, None, None, None, 5, 10, 5, 10, 
                            5, 10, 5, 5, 6, 3, 6, 3, 6, 3, 6, 3, 3, 3, 3, 3, 3, 
                            3, None, None, None, None, None, None, None])

bounds = tuple([(0, max_capacity[arch]) for arch in range(0, Aeq.shape[1])])
#print(bounds)

if len(max_capacity) != len(arc_idxs):
    print("The quantity of arches and the capacity vector don't match")
    sys.exit()
    
# check if the arch has the correct cost    
for index, arch in enumerate(arc_idxs):
    print("The arch: {} has a cost of: {}".format(arch, max_capacity[index]))


print("\n\n## Optimazer inputs ## \n\n"
      "Cost vector: {} \n"
      "Node-Arch matrix: \n {} \n"
      "Demand-Supply vector: {} \n"
      "Bounds of each arch: {} \n".format(cost_vector, Aeq, beq, bounds))

    
# Optimize
result = linprog(cost_vector, A_eq = Aeq, b_eq=beq, bounds=bounds, method="simplex")

# Resuts 

print("\n\n## Results ## \n\n")
print("Maximum flow send through each arc: ")
for i in range(len(result.x)):
    print("\t{} -> {}".format(arc_idxs[i], result.x[i].astype(int)))

max_flow = result.fun * - 1 # Mult x -1 since the new arch added has negative cost 
print("\n\tThe maximum flow is {:.2f} \n".format(max_flow))




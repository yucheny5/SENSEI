#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rwr_utils import reader, normalizer
import numpy as np
import pickle

class PyRWR:
    normalized = False

    def __init__(self):
        pass

    def read_graph(self, input_path, graph_type):
        '''
        Read a graph from the edge list at input_path

        inputs
            input_path : str
                path for the graph data
            graph_type : str
                type of graph {'directed', 'undirected', 'bipartite'}
        '''

        self.A, self.base = reader.read_graph(input_path, graph_type)
        self.m, self.n = self.A.shape
        self.node_ids = np.arange(0, self.n) + self.base
        self.normalize()

    def read_csr_matrix(self, csr_matrix):
        self.A = csr_matrix
        self.m, self.n = self.A.shape
        self.base = 0
        self.normalize()
        self.node_ids = np.arange(0, self.n) + self.base

    def read_matrix(self, adj_matrix):
        self.A = adj_matrix
        self.m, self.n = self.A.shape
        self.base = 0


    def normalize(self):
        '''
        Perform row-normalization of the adjacency matrix
        '''
        if self.normalized is False:
            nA = normalizer.row_normalize(self.A)
            self.nAT = nA.T
            self.normalized = True

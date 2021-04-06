# EvoBlock

import random
import copy

from typing import List, Optional, Tuple, Union, Set
import torch
import torch.nn as nn

# from monai.networks.layers.factories import Act, Dropout, Norm, split_args
from monai.utils import has_option
from .evonorm_primitives import PRIMITIVES

class EvonormParameters():
    def __init__(self):
        self.nodes: List
        self.adjacency_list: List


class EvoNormLayer(nn.Module):
    """
    Class that implements an EvoNormLayer.

    Reference:
        H. Liu, A. Brock, K. Simonyan, Q. Le
        “Evolving Normalization-Activation Layers,”,
        arXiv e-prints, 2020.
        https://arxiv.org/pdf/2004.02967.pdf
    """
    def __init__(
        self,
        in_channels: int,
        parameters: Optional[EvonormParameters] = None,
        max_nodes: int = 14,
        max_arity: int = 2,
    ):
        """Initializes the EvoNorm Graph

        Args:
            in_channels: number of input channels
            parameters: specifies the parameters of the evonorm layer
            max_nodes: maximum number of nodes in the graph
            max_arity: max number of arguments that a node can operate on
        """
        super().__init__()

        self.in_channels = in_channels
        self.max_nodes = max_nodes
        self.max_arity = max_arity

        self.v0 = nn.Parameter(torch.zeros(in_channels), requires_grad=True).view(1, in_channels, 1, 1, 1)
        self.v1 = nn.Parameter(torch.ones(in_channels), requires_grad=True).view(1, in_channels, 1, 1, 1)

        # def change_v0_if_needed(x):
        #     if x.size(1) !=  self.in_channels:
        #         import pdb; pdb.set_trace()
        #         self.in_channels = x.size(1)
        #         self.v0 = nn.Parameter(torch.zeros(in_channels), requires_grad=True).view(1, in_channels, 1, 1)
        #         self.v1 = nn.Parameter(torch.ones(in_channels), requires_grad=True).view(1, in_channels, 1, 1)
        #
        #     return self.v0
        #
        # def change_v1_if_needed(x):
        #     if x.size(1) !=  self.in_channels:
        #         import pdb; pdb.set_trace()
        #         self.in_channels = x.size(1)
        #         self.v0 = nn.Parameter(torch.zeros(in_channels), requires_grad=True).view(1, in_channels, 1, 1)
        #         self.v1 = nn.Parameter(torch.ones(in_channels), requires_grad=True).view(1, in_channels, 1, 1)
        #
        #     return self.v1

        self.nodes = [nn.Identity(),
                      lambda x: nn.Parameter(torch.zeros(in_channels), requires_grad=True).view(1, in_channels, 1, 1, 1),
                      lambda x: nn.Parameter(torch.ones(in_channels), requires_grad=True).view(1, in_channels, 1, 1, 1),
                      lambda x: nn.Parameter(torch.zeros_like(x), requires_grad=False)]

        if parameters:
            self.nodes += parameters.nodes
            self.adjacency_list = parameters.adjacency_list
        else:
            self.adjacency_list = [[] for _ in self.nodes]

    @property
    def parameters(self) -> EvonormParameters:
        parameters = EvonormParameters()
        parameters.nodes = copy.deepcopy(self.nodes[4:])
        parameters.adjacency_list = copy.deepcopy(self.adjacency_list)
        return parameters

    # def __str__(self):
    #     forward_nodes, _ = self.get_forward_nodes_()
    #     # outstr = "Forward nodes: " + str([self.nodes[i] for i in forward_nodes]) + "\n"
    #     outstr += f"Channel dim: {self.in_channels}"
    #     return outstr
    #

    def mutate(self):
        """Mutates the graph
        """
        new_node = random.choice(PRIMITIVES)
        if type(new_node) is list:
            new_node = random.choice(new_node)

        new_node = new_node()
        num_parents = new_node.arity
        new_node_index = len(self.nodes)

        parent_indexes = random.sample([i for i in range(len(self.nodes))], num_parents)

        while True:
            for index in parent_indexes:
                if index == 0 or index in self.adjacency_list[0]:
                    break
            else:
                parent_indexes = random.sample([i for i in range(len(self.nodes))], num_parents)
                continue
            break

        for index in parent_indexes:
            self.adjacency_list[index].append(new_node_index)

        self.adjacency_list.append([])
        self.nodes.append(new_node)
        self.add_module(str(new_node), new_node)

    def forward(self, x):
        """Performs a forward-pass over the computational graph
        """
        output_node_indexes, inputs_dict = self.get_forward_nodes_()
        if len(output_node_indexes) == 0:
            return self.nodes[0](x)

        intermediate_outs = {}
        for node in output_node_indexes:
            if node not in inputs_dict:
                intermediate_outs[node] = self.nodes[node](x)
                continue
            inputs = [intermediate_outs[i] for i in inputs_dict[node]]
            intermediate_outs[node] = self.nodes[node](*inputs)
        output_node = max(output_node_indexes)

        return intermediate_outs[output_node]

    def get_forward_nodes_(self):
        """Gets a list of nodes that are connected to the output

        Returns:
            output_nodes: A sorted list of all of the nodes by index that form the input-output chain
            inputs_dict: A dictionary that maps each node index to a list of its output
        """
        input_index = 0
        output_index = len(self.nodes)-1
        node_paths = []
        for input_index, _ in enumerate(self.nodes):
            node_paths += self.traverse_dag_(input_index)

        output_paths = [path for path in node_paths if output_index in path]
        output_nodes = set()
        inputs_dict = {}

        for path in output_paths:
            path.reverse()
            for i, pi in enumerate(path[1:],start=1):
                if path[i-1] not in inputs_dict:
                    inputs_dict[path[i-1]] = set()
                inputs_dict[path[i-1]].add(pi)
                output_nodes.add(path[i-1])
                output_nodes.add(pi)

        output_nodes = list(output_nodes)
        output_nodes.sort()

        return output_nodes, inputs_dict

    def traverse_dag_(self,
                      node: int,
                      current_path: Optional[List] = None,
                      paths: Optional[List] = None) -> List:
        """Recursive DAG traversal implementation
        """
        if len(self.adjacency_list[node]) > 0:
            for next_node in self.adjacency_list[node]:
                if current_path is None:
                    current_path = []
                    paths = []
                current_path.append(node)
                self.traverse_dag_(next_node, current_path, paths)
                current_path.pop()
        else:
            if current_path is None:
                current_path = []
                paths = []
            paths.append(current_path + [node])

        if paths is not None:
            return paths
        return []

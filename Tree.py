import numpy as np
from queue import LifoQueue
import itertools


class Node:
    def __init__(self, id, lemma, form=None, sent_name=None):
        self.id = id
        self.lemma = lemma
        self.form = form
        self.sent_name = sent_name


class Edge:
    def __init__(self, node_id_from, node_id_to, weight):
        self.node_from = node_id_from
        self.node_to = node_id_to
        self.weight = weight  # relation type


class Tree:
    def __init__(self):
        self.edges = []
        self.nodes = []
        self.heights = []
        self.edges_dict_from = {}
        self.edges_dict_to = {}
        self.nodes_dict_id = {}

    def set_help_dict(self):
        self.edges_dict_from = {k: list(v) for k, v in itertools.groupby(sorted(self.edges, key=lambda x: x.node_from),
                                                                         key=lambda x: x.node_from)}
        self.nodes_dict_id = {node.id: node for node in self.nodes}
        self.edges_dict_to = {k: list(v)[0] for k, v in itertools.groupby(self.edges, key=lambda x: x.node_to)}

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_id):
        return self.nodes_dict_id.get(node_id)  # return Node class instance

    def get_children(self, node_id):
        edges = self.edges_dict_from.get(node_id)
        return list(map(lambda x: x.node_to, edges if edges is not None else []))

    def get_edge(self, to_id):
        return self.edges_dict_to.get(to_id)

    def remove_edge(self, to_id):
        self.edges = list(filter(lambda x: x.node_to != to_id, self.edges))

    def calculate_heights(self):
        visited = np.full(len(self.nodes), False, dtype=bool)
        self.heights = np.full(len(self.nodes), -1, dtype=int)  # all heights are -1 initially
        stack = LifoQueue()
        stack.put(0)
        prev = None
        while stack.qsize() > 0:
            curr = stack.get()
            stack.put(curr)
            if not visited[curr]:
                visited[curr] = True
            children = self.get_children(curr)
            if len(children) == 0:
                self.heights[curr] = 0
                prev = curr
                stack.get()
            else:
                all_visited_flag = True
                for child in children:
                    if not visited[child]:
                        all_visited_flag = False
                        stack.put(child)
                if all_visited_flag:
                    if len(children) > 1:
                        max_height = -1
                        for child in children:
                            if self.heights[child] > max_height:
                                max_height = self.heights[child]
                        curr_height = max_height + 1
                    else:
                        curr_height = self.heights[prev] + 1
                    self.heights[curr] = curr_height
                    prev = curr
                    stack.get()

    def simple_dfs(self, vertex):
        sequence = []
        node = self.get_node(vertex)
        if node is not None:
            sequence.append(tuple([vertex, node.form, node.sent_name]))
            visited = np.full(len(self.nodes), False, dtype=bool)
            stack = [vertex]
            while len(stack) > 0:
                curr = stack[-1]
                if not visited[curr]:
                    visited[curr] = True
                children = self.get_children(curr)
                if len(children) == 0:
                    stack.pop()
                else:
                    all_visited_flag = True
                    for child in children:
                        if not visited[child]:
                            all_visited_flag = False
                            stack.append(child)
                            node = self.get_node(child)
                            sequence.append(tuple([child, node.form, node.sent_name]))
                    if all_visited_flag:
                        stack.pop()
        return sequence

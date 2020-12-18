import itertools
from queue import LifoQueue


class Node:
    def __init__(self, id, lemma=None, form=None, sent_name=None, is_included=False):
        self.id = id
        self.lemma = lemma
        self.form = form
        self.sent_name = sent_name
        self.is_included = is_included


class Edge:
    def __init__(self, node_id_from, node_id_to, weight):
        self.node_from = node_id_from
        self.node_to = node_id_to
        self.weight = weight  # relation type


class Tree:

    def __init__(self):
        self.edges = []
        self.nodes = []
        self.inactive = set()
        self.created = set()
        self.heights = {}
        self.edges_dict_from = {}
        self.edges_dict_to = {}
        self.nodes_dict_id = {}
        self.additional_nodes = set()
        self.similar_lemmas = {}
        self.global_similar_mapping = {}

    def set_help_dict(self):
        self.edges_dict_from = {k: list(v) for k, v in itertools.groupby(sorted(self.edges, key=lambda x: x.node_from),
                                                                         key=lambda x: x.node_from)}
        self.nodes_dict_id = {node.id: node for node in self.nodes}
        self.edges_dict_to = {k: list(v) for k, v in itertools.groupby(self.edges, key=lambda x: x.node_to)}
        # filtered_edges = list(filter(lambda x: x.node_to not in self.additional_nodes and x.node_from not in self.additional_nodes, self.edges))
        # self.edges_dict_to = {k: list(set(v) - self.additional_nodes) for k, v in itertools.groupby(filtered_edges, key=lambda x: x.node_to)}

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_id):
        return self.nodes_dict_id.get(node_id)  # return Node class instance

    def get_children(self, node_id):
        edges = self.edges_dict_from.get(node_id)
        # only_active = list(filter(lambda edge: edge.node_to not in self.inactive and edge.node_from not in self.inactive, edges if edges is not None else []))
        # return list(map(lambda x: x.node_to, only_active if only_active is not None else []))
        return set(map(lambda x: x.node_to, edges if edges is not None else []))

    def get_edge(self, to_id):
        return self.edges_dict_to.get(to_id)

    def remove_edge(self, to_id):
        self.edges = list(filter(lambda x: x.node_to != to_id, self.edges))

    def copy_node_details(self, existing_node, id_count):
        new_node = Node(id=id_count,
                    form=existing_node.form,
                    sent_name=existing_node.sent_name,
                    is_included=existing_node.is_included)
        return new_node

    def add_new_edges(self, new_node_id, children):
        for child_id in children:
            new_edge = Edge(new_node_id, child_id, self.get_edge(child_id)[0].weight)
            if child_id in self.edges_dict_to.keys():
                self.edges_dict_to[child_id].append(new_edge)
            else:
                self.edges_dict_to[child_id] = [new_edge]
            self.edges.append(new_edge)
            if new_node_id in self.edges_dict_from.keys():
                self.edges_dict_from[new_node_id].append(new_edge)
            else:
                self.edges_dict_from[new_node_id] = [new_edge]
            # if new_edge.node_to not in self.edges_dict_to.keys():
            #     self.edges_dict_to[new_edge.node_to] = [new_edge]
            # else:
            #     self.edges_dict_to[new_edge.node_to].append(new_edge)

    def add_edge_to_dict(self, edge):
        # if edge.node_to not in self.edges_dict_to.keys():
        #     self.edges_dict_to[edge.node_to] = [edge]
        # else:
        #     self.edges_dict_to[edge.node_to].append(edge)
        self.edges_dict_to[edge.node_to] = [edge]
        if edge.node_from in self.edges_dict_from.keys():
            self.edges_dict_from[edge.node_from].append(edge)
        else:
            self.edges_dict_from[edge.node_from] = edge
        self.edges.append(edge)

    def add_node_to_dict(self, node):
        self.nodes_dict_id[node.id] = node
        self.created.add(node.id)
        self.nodes.append(node)

    def add_inactive(self, node_id):
        self.inactive.add(node_id)

    def get_children_nodes(self, children):
        children_nodes = {
            str(Tree.get_edge(self, child)[0].weight) + str(Tree.get_node(self, child).lemma): child for
            child in children}
        return children_nodes

    #
    # def remove_node(self, existing_node, existing_edge):
    #     self.nodes.remove(existing_node)
    #     self.nodes_dict_id.pop(existing_node.id)
    #     self.edges_dict_from.pop(existing_node.id)
    #     self.edges_dict_to.pop(existing_node.id)
    #     old_edge = list(filter(lambda x: x.node_to == existing_node.id, self.edges_dict_from[existing_edge.node_from]))
    #     if len(old_edge) > 0:
    #         self.edges_dict_from[existing_edge.node_from].remove(old_edge)

    # def calculate_heights(self):
    #     visited = np.full(len(self.nodes), False, dtype=bool)
    #     self.heights = np.full(len(self.nodes), -1, dtype=int)  # all heights are -1 initially
    #     stack = LifoQueue()
    #     stack.put(0)
    #     prev = None
    #     while stack.qsize() > 0:
    #         curr = stack.get()
    #         stack.put(curr)
    #         if not visited[curr]:
    #             visited[curr] = True
    #         children = self.get_children(curr)
    #         if len(children) == 0:
    #             self.heights[curr] = 0
    #             prev = curr
    #             stack.get()
    #         else:
    #             all_visited_flag = True
    #             for child in children:
    #                 if not visited[child]:
    #                     all_visited_flag = False
    #                     stack.put(child)
    #             if all_visited_flag:
    #                 if len(children) > 1:
    #                     max_height = -1
    #                     for child in children:
    #                         if self.heights[child] > max_height:
    #                             max_height = self.heights[child]
    #                     curr_height = max_height + 1
    #                 else:
    #                     curr_height = self.heights[prev] + 1
    #                 self.heights[curr] = curr_height
    #                 prev = curr
    #                 stack.get()

    def calculate_heights(self):
        # visited = np.full(len(self.nodes), False, dtype=bool)
        visited = {node.id: False for node in self.nodes}
        # self.heights = np.full(len(self.nodes), -1, dtype=int)  # all heights are -1 initially
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
                self.heights[curr] = [0]
                prev = curr
                stack.get()
            else:
                all_visited_flag = True
                children_filtered = set(children) - self.additional_nodes
                for child in children_filtered:
                    try:
                        if not visited[child]:
                            all_visited_flag = False
                            stack.put(child)
                    except IndexError as ie:
                        ff = []
                if all_visited_flag:
                    curr_height = []
                    if len(children_filtered) > 1:
                        for child in children_filtered:
                            for child_height in self.heights[child]:
                                curr_height.append(child_height + 1)
                    else:
                        try:
                            curr_height = [h + 1 for h in self.heights[prev]]
                        except KeyError as e:
                            efl = []
                    self.heights[curr] = list(set(curr_height))
                    prev = curr
                    stack.get()
    #
    # def simple_dfs(self, vertex):
    #     sequence = []
    #     node = self.get_node(vertex)
    #     if node is not None:
    #         sequence.append(tuple([vertex, node.form, node.sent_name]))
    #         visited = np.full(len(self.nodes), False, dtype=bool)
    #         stack = [vertex]
    #         while len(stack) > 0:
    #             curr = stack[-1]
    #             if not visited[curr]:
    #                 visited[curr] = True
    #             children = self.get_children(curr)
    #             if len(children) == 0:
    #                 stack.pop()
    #             else:
    #                 all_visited_flag = True
    #                 for child in children:
    #                     if not visited[child]:
    #                         all_visited_flag = False
    #                         stack.append(child)
    #                         node = self.get_node(child)
    #                         sequence.append(tuple([child, node.form, node.sent_name]))
    #                 if all_visited_flag:
    #                     stack.pop()
    #     return sequence

    def simple_dfs(self, vertex, subtree_vertices):
        sequence = []
        node = self.get_node(vertex)
        if node is not None:
            sequence.append(tuple([vertex, node.lemma, node.form, node.sent_name]))
            # visited = np.full(len(self.nodes), False, dtype=bool)
            visited = {node.id: False for node in self.nodes}
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
                        if child in subtree_vertices and not visited[child]:
                            all_visited_flag = False
                            stack.append(child)
                            node = self.get_node(child)
                            sequence.append(tuple([child, node.lemma, node.form, node.sent_name]))
                    if all_visited_flag:
                        stack.pop()
        return sequence

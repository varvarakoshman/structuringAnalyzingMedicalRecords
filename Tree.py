from queue import LifoQueue


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
        self.heights = {}

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_id):
        return list(filter(lambda x: x.id == node_id, self.nodes))[0]  # return Node class instance

    def get_children(self, node_id):
        return list(map(lambda x: x.node_to, filter(lambda x: x.node_from == node_id, self.edges)))

    def get_edge(self, to_id):
        optional_edge = list(filter(lambda x: x.node_to == to_id, self.edges))
        if len(optional_edge) != 0:
            return optional_edge[0]
        else:
            return None

    def calculate_heights(self):
        # visited = np.full(len(self.nodes), False, dtype=bool)
        visited = {node.id: False for node in self.nodes}
        # self.heights = np.full(len(self.nodes), -1, dtype=int)  # all heights are -1 initially
        # heights = {i: -1 for i in list(map(lambda x: x.id, self.nodes))}
        stack = LifoQueue() # push fictional root on top
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
            curr_edge = self.get_edge(vertex)
            if curr_edge is not None:
                sequence.append(tuple([vertex, node.lemma, node.form, node.sent_name, curr_edge.weight]))
            visited = []
            stack = [vertex]
            while len(stack) > 0:
                curr = stack[-1]
                if curr not in visited:
                    visited.append(curr)
                children = self.get_children(curr)
                if len(children) == 0:
                    stack.pop()
                else:
                    all_visited_flag = True
                    for child in children:
                        if child not in visited:
                            all_visited_flag = False
                            stack.append(child)
                            node = self.get_node(child)
                            sequence.append(tuple([child, node.lemma, node.form, node.sent_name, self.get_edge(child).weight]))
                    if all_visited_flag:
                        stack.pop()
        return sequence
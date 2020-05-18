import os
from collections import defaultdict

import numpy as np
import pandas as pd


class Node:
    def __init__(self, id, lemma, form, sent_name):
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

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_id):
        return list(filter(lambda x: x.id == node_id, self.nodes))[0]  # return Node class instance

    def get_children(self, node):
        return list(map(lambda x: x.node_to, filter(lambda x: x.node_from == node.id, self.edges)))

    def get_edge(self, to_id):
        optional_edge = list(filter(lambda x: x.node_to == to_id, self.edges))
        if len(optional_edge) != 0:
            return optional_edge[0]
        else:
            return None

    def calculate_heights(self):
        visited = np.full(len(self.nodes), False, dtype=bool)
        # visited = {i: False for i in list(map(lambda x: x.id, self.nodes))}
        self.heights = np.full(len(self.nodes), -1, dtype=int)  # all heights are -1 initially
        # heights = {i: -1 for i in list(map(lambda x: x.id, self.nodes))}
        stack = [0]  # push fictional root on top
        prev = None
        while len(stack) > 0:
            curr = stack[-1]
            if not visited[curr]:
                visited[curr] = True
            children = self.get_children(self.get_node(curr))
            if len(children) == 0:
                self.heights[curr] = 0
                prev = curr
                stack.pop()
            else:
                all_visited_flag = True
                for child in children:
                    if not visited[child]:
                        all_visited_flag = False
                        stack.append(child)
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
                    stack.pop()


# input: map_dict - {v.id: str value of mapped string}
# returns: sorted_res - {v.id: str value of mapped string} in sorted order
def radix_sort(map_dict):
    if len(map_dict) == 0:
        return map_dict
    map_dict_rev = defaultdict(list)  # reversed: {str value of mapped string: [v.id]}
    for key, val in map_dict.items():
        map_dict_rev[val].append(key)
    n_digits = max(list(map(lambda x: len(x), map_dict_rev.keys())))  # max N of digits
    numbers = list(map_dict_rev.keys())
    b = numbers.copy()  # array for intermediate results
    upper_bound = 10
    for i in range(n_digits):
        c = np.zeros(upper_bound, dtype=int)  # temp array
        for j in range(len(numbers)):
            d = int(numbers[j]) // pow(10, i)
            c[d % 10] += 1
        cnt = 0
        for j in range(upper_bound):
            tmp = c[j]
            c[j] = cnt
            cnt += tmp
        for j in range(len(numbers)):
            d = int(numbers[j]) // pow(10, i)
            b[c[d % 10]] = numbers[j]
            c[d % 10] += 1
        numbers = b.copy()
    sorted_res = {list_item: number for number in numbers for list_item in map_dict_rev[number]}
    return sorted_res


# from https://stackoverflow.com/questions/18262306/quicksort-with-python
def quick_sort(array):
    less = []
    equal = []
    greater = []
    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        return quick_sort(less) + equal + quick_sort(greater)
    else:
        return array


def remap_s(str_dict):
    m = 0
    remapped = {key: "" for key in str_dict.keys()}  # key : node id, value: mapped string
    already_seen = {}
    for tpl in str_dict.items():
        for char in tpl[1]:
            if char in already_seen.keys():
                mapping = already_seen[char]
            else:
                m = m + 1
                already_seen[char] = m
                mapping = m
            remapped[tpl[0]] += str(mapping)
    remapped = {item[0]: np.array([int(item[1][p]) for p in range(len(item[1]))]) for item in remapped.items()}
    return remapped


# input: remapped_dict - dictionary {v.id: int array of mapped string}
# returns: sorted_res - dictionary {v.id: str value of mapped string}
def sort_strings_inside(remapped_dict):
    sorted_res = {}
    for item in remapped_dict.items():
        sorted_array = quick_sort(item[1])
        sorted_res[item[0]] = "".join([str(digit) for digit in sorted_array])
    return sorted_res


DATA_PATH = r'parus_results'
files = os.listdir(DATA_PATH)

# Чтение деревьев в дата фрейм
trees_df = pd.DataFrame(columns=['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])

for file in files:
    full_dir = os.path.join(DATA_PATH, file)
    name = file.split('.')[0]
    with open(full_dir, encoding='utf-8') as f:
        this_df = pd.read_csv(f, sep='\t',
                              names=['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])
        this_df['sent_name'] = name
        trees_df = pd.concat([trees_df, this_df], ignore_index=True)

# delete useless data
trees_df = trees_df.drop(columns=['upostag', 'xpostag', 'feats'], axis=1)
trees_df.drop(index=[11067], inplace=True)
trees_df.loc[13742, 'deprel'] = 'разъяснит'

# delete relations of type PUNC and reindex
trees_df_filtered = trees_df[trees_df.deprel != 'PUNC']
trees_df_filtered = trees_df_filtered.reset_index(drop=True)
trees_df_filtered.index = trees_df_filtered.index + 1

# trees_df_filtered.head()
# TEST - тест на первых 3х предложениях
# trees_df_filtered = trees_df_filtered.head(40)

# get all lemmas and create a dictionary to map to numbers
dict_lemmas = {lemma: index for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
# get all relations and create a dictionary to map to numbers
dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}

# construct a tree with a list of edges and a list of nodes
whole_tree = Tree()
root_node = Node(0, 0, None, None)  # add root
Tree.add_node(whole_tree, root_node)
for name, group in trees_df_filtered.groupby('sent_name'):
    row_ids = trees_df_filtered.index[trees_df_filtered.sent_name == name].to_list()
    # temporary dictionary for remapping indices
    temp_dict = {key: row_ids[ind] for ind, key in enumerate(group.id.to_list())}
    temp_dict[0] = 0
    for _, row in group.iterrows():
        new_id = temp_dict.get(row['id'])
        new_node = Node(new_id, dict_lemmas.get(row['lemma']), row['form'], row['sent_name'])
        Tree.add_node(whole_tree, new_node)
        new_edge = Edge(temp_dict.get(row['head']), new_id, dict_rel.get(row['deprel']))
        Tree.add_edge(whole_tree, new_edge)

# partition nodes by height
Tree.calculate_heights(whole_tree)

heights_dictionary = {Tree.get_node(whole_tree, node_id): height for node_id, height in enumerate(whole_tree.heights)}

grouped_heights = defaultdict(list)
for key, value in heights_dictionary.items():
    grouped_heights[value].append(key)
grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

# compute subtree repeats
reps = 0
count = len(dict_lemmas.keys())
r_classes= [[] for _ in range(len(whole_tree.nodes))]
k = ["" for x in range(len(whole_tree.nodes))]  # identifiers of subtrees
k_2 = ["" for x in range(len(whole_tree.nodes))]  # identifiers of edges of subtrees
for nodes in grouped_heights:
    # construct a string of numbers for each node v and its children
    s = {}  # key: node id, value: str(lemma id + ids of subtrees)
    w = {key: "" for key in list(map(lambda x: x.id, nodes[1]))}  # key: node_id, value: str(weights of edges from current node to leaves)
    n_children = 0
    for v in nodes[1]:
        children_v = Tree.get_children(whole_tree, v)
        s[v.id] = str(v.lemma)
        if len(children_v) > 0:
            n_children += len(children_v)
            w[v.id] += str(k_2[v.id])
            for child_id in children_v:
                s[v.id] += str(k[child_id])
        edge_to_curr = Tree.get_edge(whole_tree, v.id)
        if edge_to_curr is not None:
            k_2[edge_to_curr.node_from] += str(edge_to_curr.weight) + str(v.lemma)
    # remap numbers from [1, |alphabet| + |T|) to [1, H[i] + #of children for each node]
    # needed for radix sort - to keep strings shorter
    remapped_nodes = remap_s(s)  # key: v.id, value: int array of remapped value
    remapped_edges = remap_s(w)
    # sort inside each string
    sorted_remapped_nodes = sort_strings_inside(remapped_nodes)  # {v.id: str value of mapped string}
    sorted_remapped_edges = sort_strings_inside(remapped_edges)  # {v.id: str value of mapped string}
    # upper_map_bound_n = len(nodes[1]) + n_children
    # upper_map_bound_e = n_children
    # lexicographically sort the mapped strings with radix sort
    sorted_strings = radix_sort(sorted_remapped_nodes)  # {v.id: str value of mapped string}
    sorted_edges = radix_sort(sorted_remapped_edges)  # {v.id: str value of mapped string}
    reps += 1
    # assign classes
    sorted_vertices_id = list(sorted_strings.keys())
    prev_vertex = sorted_vertices_id[0]
    k[prev_vertex] = reps + count
    r_classes[reps].append(prev_vertex)
    for ind in range(1, len(sorted_vertices_id)):
        vertex = sorted_vertices_id[ind]
        if sorted_strings[vertex] == sorted_strings[prev_vertex] and len(sorted_edges) > 0 and sorted_edges[vertex] == \
                sorted_edges[prev_vertex]:
            r_classes[reps].append(vertex)
        else:
            reps += 1
            r_classes[reps] = [vertex]
        k[vertex] = reps + count
        prev_vertex = vertex

trees_df_filtered.head()

# show classes once finished with debug
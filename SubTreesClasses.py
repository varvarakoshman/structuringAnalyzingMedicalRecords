import os
from collections import defaultdict
import pandas as pd

import Util
from Tree import Tree, Node, Edge
from Util import radix_sort, sort_strings_inside, remap_s
from itertools import combinations


def read_data():
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
    return trees_df_filtered


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel):
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
    return whole_tree


def compute_full_subtrees(whole_tree, count, grouped_heights):
    # compute subtree repeats
    reps = 0
    r_classes= [[] for _ in range(len(whole_tree.nodes))]
    k = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of subtrees
    k_2 = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of edges of subtrees
    for nodes in grouped_heights:
        # construct a string of numbers for each node v and its children
        s = {}  # key: node id, value: str(lemma id + ids of subtrees)
        w = {key: "" for key in list(map(lambda x: x.id, nodes[1]))}  # key: node_id, value: str(weights of edges from current node to children)
        for v in nodes[1]:
            children_v = Tree.get_children(whole_tree, v)
            s[v.id] = str(v.lemma)
            if len(children_v) > 0:
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
    return r_classes


def compute_part_subtrees(whole_tree, dict_lemmas, grouped_heights):
    # compute subtree repeats
    reps = 0
    count = len(dict_lemmas.keys())
    r_classes = [[] for _ in range(len(whole_tree.nodes))]
    k = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of subtrees
    k_2 = [[] for _ in range(len(whole_tree.nodes))]  # identifiers of edges of subtrees
    for nodes in grouped_heights:
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        filtered_groups = list(filter(lambda x: len(x[1]) > 1, list(grouped_lemmas.items())))
        for group in filtered_groups:
            curr_lemma = group[0]
            id_combinations_dict = {}
            for v_id in group[1]:
                edge_to_curr = Tree.get_edge(whole_tree, v_id)
                if edge_to_curr is not None:
                    k_2[edge_to_curr.node_from].append(str(edge_to_curr.weight) + str(curr_lemma))
                if nodes[0] != 0: # not applicable to leaves
                    all_combinations = [list(combinations(k_2[v_id], i)) for i in range(1, len(k_2[v_id]) + 1)] # do flat map thing
                    flat_mapped_combs = [i[0][0] for i in all_combinations]
                    id_combinations_dict[v_id] = all_combinations



def main():
    trees_df_filtered = read_data()
    # TEST - тест на первых 3х предложениях
    trees_df_filtered = trees_df_filtered.head(40)

    # get all lemmas and create a dictionary to map to numbers
    dict_lemmas = {lemma: index for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    # get all relations and create a dictionary to map to numbers
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}

    whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel)
    # partition nodes by height
    Tree.calculate_heights(whole_tree)

    heights_dictionary = {Tree.get_node(whole_tree, node_id): height for node_id, height in
                          enumerate(whole_tree.heights)}
    grouped_heights = defaultdict(list)
    for key, value in heights_dictionary.items():
        grouped_heights[value].append(key)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # classes for full repeats
    # classes_full = compute_full_subtrees(whole_tree, len(dict_lemmas.keys()), grouped_heights)
    # classes_part = compute_part_subtrees(whole_tree, len(dict_lemmas.keys()), grouped_heights)

    # TEST
    test_tree = Util.get_test_tree()
    dict_lemmas_test_size = len(set(map(lambda x: x.lemma, test_tree.nodes)))
    Tree.calculate_heights(test_tree)
    heights_dictionary_tst = {Tree.get_node(test_tree, node_id): height for node_id, height in
                          enumerate(test_tree.heights)}
    grouped_heights_tst = defaultdict(list)
    for key, value in heights_dictionary_tst.items():
        grouped_heights_tst[value].append(key)
    grouped_heights_test = sorted(grouped_heights_tst.items(), key=lambda x: x[0])
    compute_part_subtrees(test_tree, dict_lemmas_test_size, grouped_heights_test)

    trees_df_filtered.head()


if __name__ == '__main__':
    main()
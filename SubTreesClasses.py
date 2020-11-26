import os
import time
from collections import defaultdict
from itertools import combinations, product

import pandas as pd
from gensim.models import Word2Vec

from Tree import Tree, Node, Edge
from Util import radix_sort, sort_strings_inside, remap_s, new_test


def read_data():
    DATA_PATH = r'medicalTextTrees/parus_results'
    files = os.listdir(DATA_PATH)
    trees_df = pd.DataFrame(columns=['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])

    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        name = file.split('.')[0]
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t',
                                  names=['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])
            if this_df['id'].duplicated().any():
                start_of_subtree_df = list(this_df.groupby(this_df.id).get_group(1).index)
                boundaries = start_of_subtree_df + [max(list(this_df.index)) + 1]
                list_of_dfs = [this_df.iloc[boundaries[n]:boundaries[n + 1]] for n in range(len(boundaries) - 1)]
                local_counter = 1
                for df in list_of_dfs:
                    df['sent_name'] = name + '_' + str(local_counter)
                    trees_df = pd.concat([trees_df, df], ignore_index=True)
                    local_counter += 1
            else:
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


def train_word2vec(trees_df_filtered, lemmas):
    lemma_sent_df = trees_df_filtered[['lemma', 'sent_name']]
    lemma_sent_dict = {}
    for name, group in lemma_sent_df.groupby('sent_name'):
        lemma_sent_dict[name] = []
        for _, row in group.iterrows():
            lemma_sent_dict[name].append(row['lemma'])
    model = Word2Vec(list(lemma_sent_dict.values()), min_count=1)
    similar_dict = {}
    for lemma in lemmas:
        similar_dict[lemma] = model.most_similar(lemma)
    embeddings = [model[i] for i in lemmas]
    # vocab = list(model.wv.vocab)
    # X = model[vocab]
    # tsne = TSNE(n_components=2)
    # coordinates = tsne.fit_transform(X)
    # df = pd.DataFrame(coordinates, index=vocab, columns=['x', 'y'])
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(df['x'], df['y'])
    # ax.set_xlim(right=200)
    # ax.set_ylim(top=200)
    # for word, pos in df.iterrows():
    #     ax.annotate(word, pos)
    # plt.show()


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel):
    # construct a tree with a list of edges and a list of nodes
    whole_tree = Tree()
    root_node = Node(0, 0)  # add root
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
    r_classes = [[] for _ in range(len(whole_tree.nodes))]
    k = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of subtrees
    k_2 = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of edges of subtrees
    for nodes in grouped_heights:
        # construct a string of numbers for each node v and its children
        s = {}  # key: node id, value: str(lemma id + ids of subtrees)
        w = {key: "" for key in list(
            map(lambda x: x.id, nodes[1]))}  # key: node_id, value: str(weights of edges from current node to children)
        for v in nodes[1]:
            children_v = Tree.get_children(whole_tree, v.id)
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
            if sorted_strings[vertex] == sorted_strings[prev_vertex] and len(sorted_edges) > 0 and sorted_edges[
                vertex] == \
                    sorted_edges[prev_vertex]:
                r_classes[reps].append(vertex)
            else:
                reps += 1
                r_classes[reps] = [vertex]
            k[vertex] = reps + count
            prev_vertex = vertex
    return r_classes


def add_children_to_parents(k_2, filtered_groups, whole_tree, curr_height, old_node_new_nodes):
    all_parents = set()
    for k, v in filtered_groups.items():
        for v_id in list(v):
            edge_to_curr = Tree.get_edge(whole_tree, v_id)
            Tree.get_node(whole_tree, v_id).is_included = True
            if edge_to_curr is not None:
                parent = edge_to_curr[0].node_from
                if max(whole_tree.heights[parent]) > curr_height:
                    all_parents.add(parent)
                if v_id in old_node_new_nodes.keys():
                    lemmas_to_visit = old_node_new_nodes[v_id]
                else:
                    lemmas_to_visit = [k]
                for lemma in lemmas_to_visit:
                    label_for_child = str(edge_to_curr[0].weight) + str(lemma)
                    if parent not in k_2.keys():
                        k_2[parent] = {label_for_child}
                    else:
                        k_2[parent].add(label_for_child)
    return all_parents


def add_additional_children_to_parents(k_2, whole_tree, all_parents):
    additional_child_nodes = {}
    for parent in all_parents:
        for child_id in Tree.get_children(whole_tree, parent):
            edge_to_curr = Tree.get_edge(whole_tree, child_id)[0]
            child_node = Tree.get_node(whole_tree, child_id)
            if not child_node.is_included:
                label_for_child = str(edge_to_curr.weight) + str(child_node.lemma)
                k_2[parent].add(label_for_child)
                child_node.is_included = True
                if parent not in additional_child_nodes.keys():
                    additional_child_nodes[label_for_child] = child_id
                else:
                    additional_child_nodes[label_for_child].append(child_id)
    return additional_child_nodes


# update a label of a subtree with edge weight + a lemma of the last node
def insert_children_labels_to_parents(k_2, grouped_lemmas, whole_tree):
    for k, v in grouped_lemmas.items():
        for v_id in list(v):
            edge_to_curr = Tree.get_edge(whole_tree, v_id)
            if edge_to_curr is not None:
                label_for_child = str(edge_to_curr.weight) + str(k)
                if edge_to_curr.node_from not in k_2.keys():
                    k_2[edge_to_curr.node_from] = [label_for_child]
                else:
                    k_2[edge_to_curr.node_from].append(label_for_child)


def insert_nodeid_label_dict(lemma_nodeid_dict, whole_tree, v_id, curr_lemma):
    edge_to_curr = Tree.get_edge(whole_tree, v_id)
    if edge_to_curr is not None:
        label_for_child = str(edge_to_curr.weight) + str(curr_lemma)
        tup_id_sent = tuple([v_id, whole_tree.get_node(v_id).sent_name])  # ????????????
        if label_for_child not in lemma_nodeid_dict.keys():
            lemma_nodeid_dict[label_for_child] = [tup_id_sent]
        else:
            lemma_nodeid_dict[label_for_child].append(tup_id_sent)


def produce_combinations(k_2, v_id, str_sequence_help, str_sequence_help_reversed, equal_nodes, equal_nodes_mapping):
    if len(equal_nodes) > 0:
        list_for_combinations = []
        prepared_k_2 = set()
        included_labels = []
        for child_tree in k_2[v_id]:
            if child_tree in [item for sublist in list(equal_nodes.values()) for item in sublist] or child_tree in equal_nodes.keys():
                if child_tree in equal_nodes_mapping.keys():
                    actual_label = equal_nodes_mapping[child_tree]
                else:
                    actual_label = child_tree
                if actual_label not in included_labels:
                    list_for_combinations.append(equal_nodes[actual_label])
                    included_labels.append(actual_label)
            else:
                prepared_k_2.add(child_tree)
        combinations_repeated = list(product(*(list_for_combinations)))
        # test1 = ['10', '11', '12', '13']
        # test2 = ['14', '15']
        # combinations_repeated = list(product(*([list_for_combinations[0], test1, test2])))
        all_combinations = []
        if len(prepared_k_2) > 0:
            for l in combinations_repeated:
                merged = list(l) + list(prepared_k_2)
                all_combinations.extend(list(combinations(merged, i)) for i in range(1, len(merged) + 1))
        else:
            for l in combinations_repeated:
                all_combinations.extend(list(combinations(list(l), i)) for i in range(1, len(list(l)) + 1))
    else:
        list_for_combinations = k_2[v_id]
        all_combinations = [list(combinations(list_for_combinations, i)) for i in range(1, len(list_for_combinations) + 1)]
    all_combinations_str_joined = set()
    for comb in all_combinations:
        for tup in comb:
            combs = [str(item) for item in sorted(list(map(int, list(tup))))]
            joined_label = ''.join(sorted(combs))
            if joined_label not in str_sequence_help.keys():
                str_sequence_help[joined_label] = combs
                str_sequence_help_reversed[tuple(combs)] = joined_label
            all_combinations_str_joined.add(joined_label)
    return all_combinations_str_joined


def get_nodeid_repeats(filtered_combination_ids, str_sequence_help):
    dict_nodeid_comb = {}
    for k, v in filtered_combination_ids.items():
        for v_i in v:
            if v_i in dict_nodeid_comb.keys():
                dict_nodeid_comb[v_i].append(str_sequence_help.get(k))
            else:
                dict_nodeid_comb[v_i] = [str_sequence_help.get(k)]
    return dict_nodeid_comb


def get_unique_subtrees_mapped(dict_nodeid_comb, lemma_count, unique_subtrees_mapped_global):
    unique_subtrees = set(tuple(x) for x in [sublist for list in list(dict_nodeid_comb.values()) for sublist in list])
    unique_subtrees_mapped = {}
    existing_combinations = unique_subtrees_mapped_global.keys()
    for subtree in unique_subtrees:
        # if subtree not in existing_combinations:
        if subtree not in existing_combinations:
            lemma_count += 1
            unique_subtrees_mapped[subtree] = lemma_count
    return unique_subtrees_mapped, lemma_count


def classify_existing_node(curr_node, unique_subtrees_mapped, classes_subtreeid_nodes, node_subtrees):
    subtree_new_label = unique_subtrees_mapped.get(tuple(node_subtrees[0]))
    curr_node.lemma = subtree_new_label
    if curr_node.lemma not in classes_subtreeid_nodes.keys():
        classes_subtreeid_nodes[curr_node.lemma] = [curr_node.id]
    else:
        classes_subtreeid_nodes[curr_node.lemma].append(curr_node.id)


def remove_old_node(whole_tree, curr_node, grouped_heights, curr_height, k_2, lemma_nodeid_dict):
    edge_to_curr = Tree.get_edge(whole_tree, curr_node.id)
    node_from = edge_to_curr.node_from
    try:
        whole_tree.edges.remove(edge_to_curr)
        whole_tree.nodes.remove(curr_node)
    except ValueError as e:
        print(e)
        pass
    if curr_node in grouped_heights[curr_height][1]:
        grouped_heights[curr_height][1].remove(curr_node)
    del k_2[node_from]

    old_label = str(edge_to_curr.weight) + str(curr_node.lemma)
    old_node_id = curr_node.id
    nodes_with_old_label = lemma_nodeid_dict.get(old_label)
    return edge_to_curr, node_from, old_label, old_node_id, nodes_with_old_label


#
# def create_new_node_new(total_nodes_count, edge_to_curr, subtree, unique_subtrees_mapped, lemma_nodeid_dict):
#     new_id = total_nodes_count
#     subtree_new_label = unique_subtrees_mapped.get(tuple(subtree))
#     new_label_for_child = str(edge_to_curr.weight) + str(subtree_new_label)
#     if new_label_for_child not in lemma_nodeid_dict.keys():
#         lemma_nodeid_dict[new_label_for_child] = [new_id]
#     else:
#         lemma_nodeid_dict[new_label_for_child].append(new_id)
#     return new_id, subtree_new_label


def create_new_node(total_nodes_count, whole_tree, node_from, edge_to_curr, subtree, curr_node,
                    k_2, unique_subtrees_mapped, grouped_heights, curr_height, old_node_id,
                    nodes_with_old_label, lemma_nodeid_dict):
    new_id = total_nodes_count
    Tree.add_edge(whole_tree, Edge(node_from, new_id, edge_to_curr.weight))
    subtree_new_label = unique_subtrees_mapped.get(tuple(subtree))
    new_node = Node(new_id, subtree_new_label, curr_node.form, curr_node.sent_name)
    Tree.add_node(whole_tree, new_node)
    grouped_heights[curr_height][1].append(new_node)
    new_label_for_child = str(edge_to_curr.weight) + str(subtree_new_label)
    if node_from not in k_2.keys():
        k_2[node_from] = [new_label_for_child]
    else:
        k_2[node_from].append(new_label_for_child)
    curr_sent = list(filter(lambda x: x[0] == old_node_id, nodes_with_old_label))[0][1]  # current sentence
    new_tup = tuple([new_node.id, curr_sent])
    if new_label_for_child not in lemma_nodeid_dict.keys():
        lemma_nodeid_dict[new_label_for_child] = [new_tup]
    else:
        lemma_nodeid_dict[new_label_for_child].append(new_tup)
    return new_id, subtree_new_label


def create_and_remove_edges_to_children(labl, lemma_nodeid_dict, curr_node, whole_tree, new_id):
    ids_same_label = lemma_nodeid_dict.get(labl)
    if ids_same_label is not None:
        curr = list(filter(lambda x: x[1] == curr_node.sent_name, ids_same_label))
        if len(curr) > 0:
            id_this_sent = curr[0][0]  # always only 1
            # except IndexError:
            #     omg = {}
            old_edge = Tree.get_edge(whole_tree, id_this_sent)
            if old_edge is not None:
                whole_tree.add_edge(Edge(new_id, id_this_sent, old_edge.weight))
                try:
                    whole_tree.edges.remove(old_edge)
                except ValueError as e:
                    Tree.remove_edge(whole_tree, old_edge.node_to)


def compute_part_new_new(whole_tree, lemma_count, grouped_heights):
    classes_subtreeid_nodes = {}
    classes_subtreeid_nodes_list = {}
    # unique_subtrees_mapped_global_node_ids = {}
    unique_subtrees_mapped_global_subtree_lemma = {}
    old_node_new_nodes = {}
    equal_nodes_mapping = {}
    k_2 = {}  # identifiers of edges of subtrees
    lemma_nodeid_dict = {}
    for nodes in grouped_heights:
        curr_height = nodes[0]
        if curr_height == 1:
            djksf = {}
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        all_parents = add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes)
        additional_child_nodes = add_additional_children_to_parents(k_2, whole_tree, all_parents)
        for additional_child, child_id in additional_child_nodes.items():
            if additional_child not in lemma_nodeid_dict.keys():
                lemma_nodeid_dict[additional_child] = {child_id}
            else:
                lemma_nodeid_dict[additional_child].add(child_id)
        for lemma, ids in grouped_lemmas.items():
            for v_id in ids:
                edge_to_curr = Tree.get_edge(whole_tree, v_id)
                if edge_to_curr is not None:
                    label_for_child = str(edge_to_curr[0].weight) + str(lemma)
                    if label_for_child not in lemma_nodeid_dict.keys():
                        lemma_nodeid_dict[label_for_child] = {v_id}
                    else:
                        lemma_nodeid_dict[label_for_child].add(v_id)
        filtered_groups = {k: v for k, v in grouped_lemmas.items() if len(v) > 1}

        if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
            for lemma, ids in filtered_groups.items():
                combination_ids = {}
                str_sequence_help = {}
                str_sequence_help_reversed = {}

                # generate combinations
                for v_id in ids:
                    equal_nodes = {}
                    # only for duplicating nodes
                    children = list(filter(lambda child: child not in whole_tree.created, Tree.get_children(whole_tree, v_id)))
                    for child in children:
                        if child in old_node_new_nodes.keys():
                            edge_to_child = Tree.get_edge(whole_tree, child)[0]
                            child_node = Tree.get_node(whole_tree, child)
                            w = str(edge_to_child.weight)
                            actual_label = w + str(child_node.lemma)
                            if actual_label not in equal_nodes.keys():
                                merge = []
                                for l in old_node_new_nodes[child]:
                                    new_label = w + str(l)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                                equal_nodes[actual_label] = merge
                            else:
                                merge = []
                                for l in old_node_new_nodes[child]:
                                    new_label = w + str(l)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                                equal_nodes[actual_label].extend(merge)
                    all_combinations_str_joined = produce_combinations(k_2, v_id, str_sequence_help,
                                                                       str_sequence_help_reversed, equal_nodes,
                                                                       equal_nodes_mapping)
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]

                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                for tree_label, node_list in filtered_combination_ids.items():
                    if tree_label not in unique_subtrees_mapped_global_subtree_lemma.keys():
                        unique_subtrees_mapped_global_subtree_lemma[tree_label] = lemma_count
                        lemma_count += 1
                # 16: [['107', '919'], ['208', '919'], ['919'], ['107'], ['208']]
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help)
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    existing_node = Tree.get_node(whole_tree, node_id)
                    edge_to_curr = Tree.get_edge(whole_tree, node_id)[0]
                    children = Tree.get_children(whole_tree, node_id)
                    for subtree in node_subtrees:
                        subtree_text = str_sequence_help_reversed.get(tuple(subtree))
                        subtree_new_label = unique_subtrees_mapped_global_subtree_lemma.get(subtree_text)

                        # add new node with a new lemma
                        new_node = Tree.copy_node_details(whole_tree, existing_node)
                        new_node.lemma = subtree_new_label
                        Tree.add_node_to_dict(whole_tree, new_node)

                        # add new node to node aliases
                        if node_id not in old_node_new_nodes.keys():
                            old_node_new_nodes[node_id] = [new_node.lemma]
                        else:
                            old_node_new_nodes[node_id].append(new_node.lemma)

                        # add an edge to it
                        edge = Edge(edge_to_curr.node_from, new_node.id, edge_to_curr.weight)
                        Tree.add_edge_to_dict(whole_tree, edge)

                        # add new node to parent
                        parent_subtree_text = str(edge_to_curr.weight) + str(subtree_new_label)
                        if parent_subtree_text not in lemma_nodeid_dict.keys():
                            lemma_nodeid_dict[parent_subtree_text] = {new_node.id}
                        else:
                            lemma_nodeid_dict[parent_subtree_text].add(new_node.id)

                        subtree_children = []
                        for subtree_node in subtree:
                            intersection = set(lemma_nodeid_dict[subtree_node]) & set(children)
                            if len(intersection) != 0:
                                target_child = list(intersection)[0]
                            else:
                                target_child = list(set(lemma_nodeid_dict[equal_nodes_mapping[subtree_node]]) & set(children))[0]
                            subtree_children.append(target_child)

                        # add edges to subtree's children from new node
                        Tree.add_new_edges(whole_tree, new_node.id, subtree_children)

                        # assign class
                        if subtree_new_label not in classes_subtreeid_nodes.keys():
                            classes_subtreeid_nodes[subtree_new_label] = [new_node.id]
                        else:
                            classes_subtreeid_nodes[subtree_new_label].append(new_node.id)

                        if subtree_new_label not in classes_subtreeid_nodes_list.keys():
                            classes_subtreeid_nodes_list[subtree_new_label] = subtree_children
                        else:
                            classes_subtreeid_nodes_list[subtree_new_label].extend(subtree_children)
                        subtree_deep_children = []
                        for subtree_lemma in list(map(lambda x: Tree.get_node(whole_tree, x).lemma, subtree_children)):
                            if subtree_lemma in classes_subtreeid_nodes_list.keys():
                                try:
                                    subtree_deep_children.extend(classes_subtreeid_nodes_list[subtree_lemma])
                                except TypeError as e:
                                    hjvhjgh =[]
                        classes_subtreeid_nodes_list[subtree_new_label].extend(subtree_deep_children)
                        classes_subtreeid_nodes_list[subtree_new_label].append(new_node.id)

                    # remove old node and edges to/from it
                    Tree.add_inactive(whole_tree, node_id)
    return classes_subtreeid_nodes


def compute_part_new(whole_tree, lemma_count, grouped_heights):
    classes_subtreeid_nodes = {}
    unique_subtrees_mapped_global_node_ids = {}
    unique_subtrees_mapped_global_subtree_lemma = {}
    old_node_new_nodes = {}
    k_2 = {}  # identifiers of edges of subtrees
    lemma_nodeid_dict = {}
    equal_nodes_mapping = {}
    for nodes in grouped_heights:
        curr_height = nodes[0]
        if curr_height == 2:
            djksf = {}
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        all_parents = add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes)
        additional_child_nodes = add_additional_children_to_parents(k_2, whole_tree, all_parents)
        for additional_child, child_id in additional_child_nodes.items():
            if additional_child not in lemma_nodeid_dict.keys():
                lemma_nodeid_dict[additional_child] = {child_id}
            else:
                lemma_nodeid_dict[additional_child].add(child_id)
        for lemma, ids in grouped_lemmas.items():
            for v_id in ids:
                edge_to_curr = Tree.get_edge(whole_tree, v_id)
                if edge_to_curr is not None:
                    label_for_child = str(edge_to_curr.weight) + str(lemma)
                    if label_for_child not in lemma_nodeid_dict.keys():
                        lemma_nodeid_dict[label_for_child] = {v_id}
                    else:
                        lemma_nodeid_dict[label_for_child].add(v_id)
        filtered_groups = {k: v for k, v in grouped_lemmas.items() if len(v) > 1}
        for lemma, ids in filtered_groups.items():
            combination_ids = {}
            str_sequence_help = {}
            str_sequence_help_reversed = {}
            for v_id in ids:
                if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
                    equal_nodes = {}
                    for child in Tree.get_children(whole_tree, v_id):
                        if child in old_node_new_nodes.keys():
                            edge_to_child = Tree.get_edge(whole_tree, child)
                            child_node = Tree.get_node(whole_tree, child)
                            w = str(edge_to_child.weight)
                            actual_label = w + str(child_node.lemma)
                            if actual_label not in equal_nodes.keys():
                                merge = []
                                for l in old_node_new_nodes[child]:
                                    new_label = w + str(l)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                                equal_nodes[actual_label] = merge
                            else:
                                merge = []
                                for l in old_node_new_nodes[child]:
                                    new_label = w + str(l)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                                equal_nodes[actual_label].extend(merge)
                    all_combinations_str_joined = produce_combinations(k_2, v_id, str_sequence_help,
                                                                       str_sequence_help_reversed, equal_nodes, equal_nodes_mapping)
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]
            if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                for tree_label, node_list in filtered_combination_ids.items():
                    if tree_label not in unique_subtrees_mapped_global_node_ids:
                        unique_subtrees_mapped_global_node_ids[tree_label] = node_list
                        unique_subtrees_mapped_global_subtree_lemma[tree_label] = lemma_count
                        for child_id in node_list:
                            if child_id not in old_node_new_nodes.keys():
                                old_node_new_nodes[child_id] = [lemma_count]
                            else:
                                old_node_new_nodes[child_id].append(lemma_count)
                        lemma_count += 1
                    else:
                        for node in node_list:
                            unique_subtrees_mapped_global_node_ids[tree_label].append(node)
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help)
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    for subtree in node_subtrees:
                        subtree_text = str_sequence_help_reversed.get(tuple(subtree))
                        subtree_new_label = unique_subtrees_mapped_global_subtree_lemma.get(subtree_text)
                        if subtree_text not in lemma_nodeid_dict.keys():
                            lemma_nodeid_dict[subtree_text] = {node_id}
                        else:
                            lemma_nodeid_dict[subtree_text].add(node_id)
                        children = Tree.get_children(whole_tree, node_id)
                        subtree_children = []
                        for subtree_node in subtree:
                            intersection = set(lemma_nodeid_dict[subtree_node]) & set(children)
                            if len(intersection) != 0:
                                target_child = list(intersection)[0]
                            else:
                                target_child = list(set(lemma_nodeid_dict[equal_nodes_mapping[subtree_node]]) & set(children))[0]
                            subtree_children.append(target_child)
                        new_entry = tuple([node_id, subtree_children])
                        if subtree_new_label not in classes_subtreeid_nodes.keys():
                            classes_subtreeid_nodes[subtree_new_label] = [new_entry]
                        else:
                            classes_subtreeid_nodes[subtree_new_label].append(new_entry)
    return classes_subtreeid_nodes


# TODO: get rid of height grouping
def compute_part_subtrees(whole_tree, lemma_count, grouped_heights):
    # compute subtree repeats
    classes_subtreeid_nodes = {}
    total_nodes_count = len(whole_tree.nodes)
    k_2 = {}  # identifiers of edges of subtrees
    # dictionary {lemma: [(node id with it, sentence),...,]}
    lemma_nodeid_dict = {}

    for nodes in grouped_heights:
        curr_height = nodes[0]
        print(curr_height)
        start = time.time()
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        # group node ids of current height by lemmas
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        # remember labels of subtrees
        add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height)
        # leave for processing only nodes with repeating lemmas on current height
        filtered_groups = list(filter(lambda x: len(x[1]) > 1, list(grouped_lemmas.items())))

        # for each group of nodes with the same lemma check which have the same subtrees
        for group in filtered_groups:
            curr_lemma = group[0]
            combination_ids = {}
            # same as lemma_nodeid_dict, but for current lemma
            str_sequence_help = {}
            # find all possible subtrees combinations and corresponding node ids
            for v_id in group[1]:
                # insert node with current lemma in lemma_nodeid_dict
                insert_nodeid_label_dict(lemma_nodeid_dict, whole_tree, v_id, curr_lemma)
                if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
                    # generate possible combinations of child nodes as possible subtree repeats
                    all_combinations_str_joined = produce_combinations(k_2, v_id, str_sequence_help)
                    # mapping nodes ids to combination
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]

            # if 1 vertex has n (more than 1) subtree repeats, it is replaced with n new nodes
            old_labels_to_remove = []  # remember which nodes were replaced and no longer needed

            if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
                # from attained combinations leave only repeating
                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                # do a reversed mapping to have a list of repeating subtrees for each node id
                # concretely, get a dictionary {node id: [[repeating subtree-1 nodes],..., [repeating subtree-N nodes]]}
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help)
                # get unique subtrees for this height and assign them labels
                unique_subtrees_mapped, lemma_count = get_unique_subtrees_mapped(dict_nodeid_comb, lemma_count)

                # for each node iterate over its subtree repeats, assign them new labels and modify tree accordingly
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    curr_node = Tree.get_node(whole_tree, node_id)
                    if len(node_subtrees) == 1:  # no need to create new nodes if there is only 1 subtree
                        # assign existing node a new label(lemma) as an identifier of a subtree
                        classify_existing_node(curr_node, unique_subtrees_mapped, classes_subtreeid_nodes,
                                               node_subtrees)
                    else:
                        # otherwise remove old vertex and edge to it and create new nodes (and edges)
                        edge_to_curr, node_from, old_label, old_node_id, nodes_with_old_label = \
                            remove_old_node(whole_tree, curr_node, grouped_heights, curr_height, k_2, lemma_nodeid_dict)

                        # mark to remove old node
                        old_labels_to_remove.append(old_label)

                        # iterate over repeated subtrees and create for each a new identifier (which includes current)
                        for subtree in node_subtrees:
                            # for every subtree create a new node
                            total_nodes_count += 1
                            new_id, subtree_new_label = create_new_node(total_nodes_count, whole_tree, node_from,
                                                                        edge_to_curr, subtree, curr_node,
                                                                        k_2, unique_subtrees_mapped, grouped_heights,
                                                                        curr_height, old_node_id,
                                                                        nodes_with_old_label, lemma_nodeid_dict)
                            # for each node in a repeat subtree
                            # for all its existing children remove previous edges and create new
                            for labl in subtree:
                                create_and_remove_edges_to_children(labl, lemma_nodeid_dict, curr_node, whole_tree,
                                                                    new_id)

                            # assign a class to a subtree
                            if subtree_new_label not in classes_subtreeid_nodes.keys():
                                classes_subtreeid_nodes[subtree_new_label] = [new_id]
                            else:
                                classes_subtreeid_nodes[subtree_new_label].append(new_id)
            # remove replaced nodes
            for to_remove in set(old_labels_to_remove):
                del lemma_nodeid_dict[to_remove]
        print(time.time() - start)
    return classes_subtreeid_nodes


def main():
    # trees_df_filtered = read_data()
    # # TEST - тест на первых 3х предложениях
    # trees_df_filtered = trees_df_filtered.head(48)  # 341 - all? 48 - 3 # 3884 # 5015
    # # trees_df_filtered = trees_df_filtered[trees_df_filtered.sent_name == '48554_5']
    #
    # # get all lemmas and create a dictionary to map to numbers
    # dict_lemmas = {lemma: index for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    # # get all relations and create a dictionary to map to numbers
    # dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    # train_word2vec(trees_df_filtered, dict_lemmas)
    #
    start = time.time()
    # whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel)
    # print('Time on constructing the tree: ' + str(time.time() - start))
    whole_tree = new_test()
    Tree.set_help_dict(whole_tree)
    # partition nodes by height
    start = time.time()
    Tree.calculate_heights(whole_tree)
    print('Time on calculating all heights: ' + str(time.time() - start))

    heights_dictionary = {Tree.get_node(whole_tree, node_id): heights for node_id, heights in
                          whole_tree.heights.items()}
    grouped_heights = defaultdict(list)
    for node, heights in heights_dictionary.items():
        for height in heights:
            grouped_heights[height].append(node)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # heights_dictionary = {Tree.get_node(whole_tree, node_id): height for node_id, height in
    #                       enumerate(whole_tree.heights)}
    # grouped_heights = defaultdict(list)
    # for key, value in heights_dictionary.items():
    #     grouped_heights[value].append(key)
    # grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # classes for full repeats
    # start = time.time()
    # classes_full = compute_full_subtrees(whole_tree, len(dict_lemmas.keys()), grouped_heights)
    # print('Time on calculating full repeats: ' + str(time.time() - start))
    # for index, listt in enumerate(classes_full):
    #     vertex_seq = {}
    #     for vertex in listt:
    #         vertex_seq[vertex] = Tree.simple_dfs(whole_tree, vertex)
    #     # if len(vertex_seq.items()) > 0:
    #     filename = 'results_full/results_%s.txt' % (str(index))
    #     with open(filename, 'w') as filehandle:
    #         for key, value in vertex_seq.items():
    #             filehandle.write("%s: %s\n" % (key, value))

    dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes)))

    # classes for partial repeats
    start = time.time()
    classes_part = compute_part_new_new(whole_tree, dict_lemmas_size, grouped_heights)
    # classes_part = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)
    print('Time on calculating partial repeats: ' + str(time.time() - start))
    for k, v in classes_part.items():
        vertex_seq = {}
        for vertex in v:
            vertex_seq[vertex] = Tree.simple_dfs(whole_tree, vertex)
        if len(vertex_seq.items()) > 0:
            filename = 'results_part/results_%s.txt' % (str(k))
            with open(filename, 'w') as filehandle:
                for key, value in vertex_seq.items():
                    filehandle.write("%s: %s\n" % (key, value))

    # TEST
    # test_tree = Util.get_test_tree()
    # dict_lemmas_test_size = max(set(map(lambda x: x.lemma, test_tree.nodes)))
    # Tree.calculate_heights(test_tree)
    # heights_dictionary_tst = {Tree.get_node(test_tree, node_id): height for node_id, height in
    #                       enumerate(test_tree.heights)}
    # grouped_heights_tst = defaultdict(list)
    # for key, value in heights_dictionary_tst.items():
    #     grouped_heights_tst[value].append(key)
    # grouped_heights_test = sorted(grouped_heights_tst.items(), key=lambda x: x[0])
    # compute_part_subtrees(test_tree, dict_lemmas_test_size, grouped_heights_test)

    # trees_df_filtered.head()


if __name__ == '__main__':
    main()

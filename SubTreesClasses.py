import time
from collections import defaultdict
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import read_data, replace_time_constructions
from Tree import Tree, Node, Edge
from Constants import *
from Util import create_needed_directories, merge_in_file, write_in_file
from W2Vprocessing import load_trained_word2vec, train_word2vec


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel):
    # construct a tree with a list of edges and a list of nodes
    whole_tree = Tree()
    root_node = Node(0, 0)  # add root
    whole_tree.add_node(root_node)
    new_id_count = len(trees_df_filtered) + 1
    similar_lemmas_dict = {}
    global_similar_mapping = {}
    for name, group in trees_df_filtered.groupby('sent_name'):
        row_ids = trees_df_filtered.index[trees_df_filtered.sent_name == name].to_list()
        # temporary dictionary for remapping indices
        temp_dict = {key: row_ids[ind] for ind, key in enumerate(group.id.to_list())}
        temp_dict[0] = 0
        children_dict = {}
        edge_to_weight = {}
        for _, row in group.iterrows():
            # parameters for main node
            curr_lemmas = dict_lemmas.get(row['lemma'])
            main_id = temp_dict.get(row['id'])
            from_id = temp_dict.get(row['head'])
            weight = dict_rel.get(row['deprel'])
            sent = row['sent_name']
            form = row['form']
            # create main node
            whole_tree.create_new_node(main_id, curr_lemmas[0], form, sent, weight, from_id)
            children = [main_id]
            edge_to_weight[main_id] = weight
            global_similar_mapping[main_id] = main_id
            # if lemma has additional values add additional nodes
            if len(curr_lemmas) > 1:
                for i in range(1, len(curr_lemmas)):
                    whole_tree.create_new_node(new_id_count, curr_lemmas[i], form, sent, weight, from_id)
                    edge_to_weight[new_id_count] = weight
                    if main_id not in similar_lemmas_dict.keys():
                        similar_lemmas_dict[main_id] = [new_id_count]
                    else:
                        similar_lemmas_dict[main_id].append(new_id_count)
                    global_similar_mapping[new_id_count] = main_id
                    children.append(new_id_count)
                    new_id_count += 1
            if from_id not in children_dict.keys():
                children_dict[from_id] = children
            else:
                children_dict[from_id].extend(children)
        # if parent has additional values add additional edges
        for from_id, children in children_dict.items():
            if from_id in similar_lemmas_dict.keys():
                similar_ids = similar_lemmas_dict[from_id]
                for similar_id in similar_ids:
                    for child_id in children:
                        whole_tree.add_edge(Edge(similar_id, child_id, edge_to_weight[child_id]))
    whole_tree.additional_nodes = set([sublist for list in similar_lemmas_dict.values() for sublist in list])
    whole_tree.similar_lemmas = similar_lemmas_dict
    whole_tree.global_similar_mapping = global_similar_mapping
    return whole_tree


def add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes):
    all_parents = set()
    for k, v in grouped_lemmas.items():
        for v_id in list(v):
            edge_to_curr = whole_tree.get_edge(v_id)
            whole_tree.get_node(v_id).is_included = True
            if edge_to_curr is not None:
                parent = edge_to_curr[0].node_from
                if parent not in whole_tree.heights.keys():  # same as in whole_tree.additional
                    parent = whole_tree.global_similar_mapping[parent]
                if max(whole_tree.heights[parent]) > curr_height:
                    all_parents.add(parent)
                if v_id in old_node_new_nodes.keys():
                    lemmas_to_visit = old_node_new_nodes[v_id]
                else:
                    lemmas_to_visit = [k]
                if v_id in whole_tree.similar_lemmas.keys():
                    for node_id in whole_tree.similar_lemmas[v_id]:
                        lemmas_to_visit.append(whole_tree.get_node(node_id).lemma)
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
        for child_id in whole_tree.get_children(parent):
            edge_to_curr = whole_tree.get_edge(child_id)[0]
            child_node = whole_tree.get_node(child_id)
            if not child_node.is_included:
                label_for_child = str(edge_to_curr.weight) + str(child_node.lemma)
                k_2[parent].add(label_for_child)
                child_node.is_included = True
                if parent not in additional_child_nodes.keys():
                    additional_child_nodes[label_for_child] = child_id
                else:
                    additional_child_nodes[label_for_child].append(child_id)
    return additional_child_nodes


def produce_combinations(k_2, v_id, str_sequence_help, str_sequence_help_reversed, equal_nodes, equal_nodes_mapping, stat_dict):
    if v_id == 1194:
        rrrr = []
    if len(equal_nodes) > 0:
        list_for_combinations = []
        prepared_k_2 = set()
        included_labels = []
        for child_tree in k_2[v_id]:
            if child_tree in [item for sublist in list(equal_nodes.values()) for item in
                              sublist] or child_tree in equal_nodes.keys():
                if child_tree in equal_nodes_mapping.keys():
                    actual_label = equal_nodes_mapping[child_tree]
                else:
                    actual_label = child_tree
                if actual_label not in included_labels:
                    if actual_label in equal_nodes.keys():
                        list_for_combinations.append(equal_nodes[actual_label])
                    else:
                        list_for_combinations.append(equal_nodes[child_tree])
                    included_labels.append(actual_label)
            else:
                prepared_k_2.add(child_tree)
        combinations_repeated = list(product(*(list_for_combinations)))
        all_combinations = []
        for l in combinations_repeated:
            if len(prepared_k_2) > 0:
                merged = list(l) + list(prepared_k_2)
                all_combinations.extend(list(combinations(merged, i)) for i in range(1, len(merged) + 1))
            else:
                all_combinations.extend(list(combinations(list(l), i)) for i in range(1, len(list(l)) + 1))
    else:
        list_for_combinations = k_2[v_id]
        all_combinations = [list(combinations(list_for_combinations, i)) for i in
                            range(1, len(list_for_combinations) + 1)]
    return get_strings_from_combinations(all_combinations, v_id, str_sequence_help, str_sequence_help_reversed, stat_dict)


def get_strings_from_combinations(all_combinations, v_id, str_sequence_help, str_sequence_help_reversed, stat_dict):
    all_combinations_str_joined = set()
    new_local_duplicates = {}
    for comb in all_combinations:
        for tup in comb:
            combs = [str(item) for item in sorted(tup)]
            joined_label = EMPTY_STR.join(combs)
            l = len(joined_label)
            if l in stat_dict.keys():
                stat_dict[l] = stat_dict[l] + 1
            else:
                stat_dict[l] = 1
            if joined_label not in str_sequence_help.keys():
                str_sequence_help[joined_label] = [combs.copy()]
                str_sequence_help[joined_label].append(combs)
            if joined_label in str_sequence_help.keys():
                new_local_duplicates[v_id] = combs
            all_combinations_str_joined.add(joined_label)
            str_sequence_help_reversed[tuple(combs)] = joined_label
    return all_combinations_str_joined, new_local_duplicates


def get_nodeid_repeats(filtered_combination_ids, str_sequence_help, duplicate_combs):
    dict_nodeid_comb = {}
    for k, v in filtered_combination_ids.items():
        for v_i in v:
            if len(str_sequence_help.get(k)) > 1:
                subtree_comb = duplicate_combs[v_i]
            else:
                subtree_comb = str_sequence_help.get(k)[0]
            if v_i in dict_nodeid_comb.keys():
                dict_nodeid_comb[v_i].append(subtree_comb)
            else:
                dict_nodeid_comb[v_i] = [subtree_comb]
    return dict_nodeid_comb


def extend_equal_nodes_mapping(w, item_list, function, equal_nodes_mapping, actual_label):
    merge = []
    for item in item_list:
        new_label = w + function(item)
        merge.append(new_label)
        equal_nodes_mapping[new_label] = actual_label
    return merge


def collect_equal_nodes(whole_tree, v_id, old_node_new_nodes, equal_nodes_mapping):
    equal_nodes = {}
    function_lemma_getter = lambda node_id: str(whole_tree.get_node(node_id).lemma)
    function_identity = lambda lemma: str(lemma)
    # only for duplicating nodes
    children = whole_tree.get_children(v_id) - whole_tree.created - whole_tree.additional_nodes
    for child in children:
        if child in old_node_new_nodes.keys() or child in whole_tree.similar_lemmas.keys():
            edge_to_child = whole_tree.get_edge(child)[0]
            child_node = whole_tree.get_node(child)
            w = str(edge_to_child.weight)
            actual_label = w + str(child_node.lemma)
            merge = []
            if child in old_node_new_nodes.keys():
                merge = extend_equal_nodes_mapping(w, old_node_new_nodes[child], function_identity, equal_nodes_mapping,
                                                   actual_label)
            if child in whole_tree.similar_lemmas.keys():
                merge = extend_equal_nodes_mapping(w, whole_tree.similar_lemmas[child], function_lemma_getter,
                                                   equal_nodes_mapping, actual_label)
                merge.insert(0, actual_label)
            if actual_label not in equal_nodes.keys():
                equal_nodes[actual_label] = merge
            else:
                equal_nodes[actual_label].extend(merge)
    return equal_nodes


def find_subtree_children(whole_tree, children, subtree, lemma_nodeid_dict, equal_nodes_mapping):
    subtree_children = []
    children_nodes = {}
    for subtree_node in subtree:
        if subtree_node not in lemma_nodeid_dict.keys() or subtree_node in equal_nodes_mapping.keys():
            if subtree_node in equal_nodes_mapping.keys():
                target_child_list = list(set(lemma_nodeid_dict[equal_nodes_mapping[subtree_node]]) & children)
                if len(target_child_list) > 0:
                    target_child = target_child_list[0]
                else:
                    target_child = whole_tree.get_target_from_children(children_nodes, children, subtree_node)
            else:
                target_child = whole_tree.get_target_from_children(children_nodes, children, subtree_node)
        else:
            intersection = set(lemma_nodeid_dict[subtree_node]) & children
            if len(intersection) == 0:
                target_child = whole_tree.get_target_from_children(children_nodes, children, subtree_node)
            else:
                target_child = list(intersection)[0]
        subtree_children.append(target_child)
    return subtree_children


def find_deep_subtree_children(whole_tree, subtree_children, classes_subtreeid_nodes_list):
    subtree_deep_children = set()
    for subtree_lemma in list(
            map(lambda x: whole_tree.get_node(x).lemma, subtree_children)):
        if subtree_lemma in classes_subtreeid_nodes_list.keys():
            subtree_deep_children.update(classes_subtreeid_nodes_list[subtree_lemma])
    subtree_deep_children.update(subtree_children)
    return subtree_deep_children


def insert_node_in_tree(whole_tree, existing_node, id_count, subtree_new_label, subtree_label_sent,
                        lemma_nodeid_dict, old_node_new_nodes, edge_to_curr, node_id):
    # add a new node with a new lemma
    new_node = Tree.copy_node_details(existing_node, id_count)
    new_node.lemma = subtree_new_label
    whole_tree.add_node_to_dict(new_node)
    whole_tree.global_similar_mapping[new_node.id] = new_node.id
    if subtree_new_label not in subtree_label_sent.keys():
        subtree_label_sent[subtree_new_label] = [existing_node.sent_name]
    else:
        subtree_label_sent[subtree_new_label].append(existing_node.sent_name)
    # add new node to node aliases
    if node_id not in old_node_new_nodes.keys():
        old_node_new_nodes[node_id] = [new_node.lemma]
    else:
        old_node_new_nodes[node_id].append(new_node.lemma)
    # add an edge to the new node
    edge = Edge(edge_to_curr.node_from, new_node.id, edge_to_curr.weight)
    whole_tree.add_edge_to_dict(edge)
    # add new node to a parent
    parent_subtree_text = str(edge_to_curr.weight) + str(subtree_new_label)
    if parent_subtree_text not in lemma_nodeid_dict.keys():
        lemma_nodeid_dict[parent_subtree_text] = {new_node.id}
    else:
        lemma_nodeid_dict[parent_subtree_text].add(new_node.id)
    return new_node


def add_additional_child_nodes(additional_child_nodes, lemma_nodeid_dict):
    for additional_child, child_id in additional_child_nodes.items():
        if additional_child not in lemma_nodeid_dict.keys():
            lemma_nodeid_dict[additional_child] = {child_id}
        else:
            lemma_nodeid_dict[additional_child].add(child_id)


def add_grouped_lemmas_to_dict(whole_tree, grouped_lemmas, lemma_nodeid_dict):
    for lemma, ids in grouped_lemmas.items():
        for v_id in ids:
            edge_to_curr = whole_tree.get_edge(v_id)
            if edge_to_curr is not None:
                label_for_child = str(edge_to_curr[0].weight) + str(lemma)
                if label_for_child not in lemma_nodeid_dict.keys():
                    lemma_nodeid_dict[label_for_child] = {v_id}
                else:
                    lemma_nodeid_dict[label_for_child].add(v_id)


def add_new_subtree_label(unique_subtrees_mapped_global_subtree_lemma, lemma, lemma_count, tree_label):
    str_tree_label = str(lemma) + tree_label
    if str_tree_label not in unique_subtrees_mapped_global_subtree_lemma.keys():
        unique_subtrees_mapped_global_subtree_lemma[str_tree_label] = lemma_count
        lemma_count += 1
    return lemma_count


def compute_part_subtrees(whole_tree, lemma_count, grouped_heights):
    classes_subtreeid_nodes = {}
    classes_subtreeid_nodes_list = {}
    unique_subtrees_mapped_global_subtree_lemma = {}
    old_node_new_nodes = {}
    equal_nodes_mapping = {}
    subtree_label_sent = {}
    k_2 = {}  # identifiers of edges of subtrees
    lemma_nodeid_dict = {}
    stat_dict = {}
    saved_combinations = []
    id_count = sorted([node.id for node in whole_tree.nodes], reverse=True)[0] + 1
    for nodes in grouped_heights:
        curr_height = nodes[0]
        print(curr_height)
        start = time.time()
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        all_parents = add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes)
        additional_child_nodes = add_additional_children_to_parents(k_2, whole_tree, all_parents)
        add_additional_child_nodes(additional_child_nodes, lemma_nodeid_dict)
        add_grouped_lemmas_to_dict(whole_tree, grouped_lemmas, lemma_nodeid_dict)

        if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
            filtered_groups = {k: v for k, v in grouped_lemmas.items() if len(v) > 1}
            for lemma, ids in filtered_groups.items():
                combination_ids = {}
                str_sequence_help = {}
                str_sequence_help_reversed = {}
                duplicate_combs = {}  # for several cases
                # generate combinations
                for v_id in ids:
                    equal_nodes = collect_equal_nodes(whole_tree, v_id, old_node_new_nodes, equal_nodes_mapping)
                    all_combinations_str_joined, new_local_duplicates = produce_combinations(k_2, v_id,
                                                                                             str_sequence_help,
                                                                                             str_sequence_help_reversed,
                                                                                             equal_nodes,
                                                                                             equal_nodes_mapping, stat_dict)
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]
                    for id, comb_list in new_local_duplicates.items():
                        duplicate_combs[id] = comb_list
                        lemma_count = add_new_subtree_label(unique_subtrees_mapped_global_subtree_lemma, lemma,
                                                            lemma_count,
                                                            EMPTY_STR.join(comb_list))
                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                for tree_label, node_list in filtered_combination_ids.items():
                    lemma_count = add_new_subtree_label(unique_subtrees_mapped_global_subtree_lemma, lemma, lemma_count,
                                                        tree_label)
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help, duplicate_combs)
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    existing_node = whole_tree.get_node(node_id)
                    edge_to_curr = whole_tree.get_edge(node_id)[0]
                    children = whole_tree.get_children(node_id)
                    for subtree in node_subtrees:
                        subtree_text = str_sequence_help_reversed.get(tuple(subtree))
                        subtree_new_label = unique_subtrees_mapped_global_subtree_lemma.get(str(lemma) + subtree_text)
                        if subtree_new_label not in subtree_label_sent.keys() or existing_node.sent_name not in \
                                subtree_label_sent[subtree_new_label]:
                            # create a new node for a subtree
                            new_node = insert_node_in_tree(whole_tree, existing_node, id_count, subtree_new_label,
                                                           subtree_label_sent,
                                                           lemma_nodeid_dict, old_node_new_nodes, edge_to_curr, node_id)
                            id_count += 1
                            subtree_children = find_subtree_children(whole_tree, children, subtree, lemma_nodeid_dict,
                                                                     equal_nodes_mapping)
                            if len(subtree_children) > 0:
                                general_comb = EMPTY_STR.join(sorted(
                                    [str(whole_tree.global_similar_mapping[child_id]) for child_id in
                                     subtree_children]))
                                if general_comb not in saved_combinations:
                                    # add edges to subtree's children from new node
                                    Tree.add_new_edges(whole_tree, new_node.id, subtree_children)
                                    # assign class
                                    if subtree_new_label not in classes_subtreeid_nodes.keys():
                                        classes_subtreeid_nodes[subtree_new_label] = [new_node.id]
                                    else:
                                        classes_subtreeid_nodes[subtree_new_label].append(new_node.id)
                                    subtree_deep_children = find_deep_subtree_children(whole_tree, subtree_children,
                                                                                       classes_subtreeid_nodes_list)
                                    only_active = subtree_deep_children - whole_tree.inactive
                                    if len(only_active) == 0:
                                        only_active.update(subtree_children)
                                    if subtree_new_label not in classes_subtreeid_nodes_list.keys():
                                        classes_subtreeid_nodes_list[subtree_new_label] = only_active
                                    else:
                                        classes_subtreeid_nodes_list[subtree_new_label].update(only_active)
                                    classes_subtreeid_nodes_list[subtree_new_label].add(new_node.id)
                                    saved_combinations.append(general_comb)
                            # remove old node and edges to/from it
                            whole_tree.add_inactive(node_id)
        print(time.time() - start)
    classes_subtreeid_nodes = {k: v for k, v in classes_subtreeid_nodes.items() if
                               len(v) > 1}  # TODO: why do len=1 entries even appear here??

    res2 = dict(sorted(stat_dict.items(), key=lambda x: x[0]))
    alphab = list(res2.values())
    frequencies = list(res2.keys())

    pos = np.arange(len(frequencies))
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(frequencies)
    ax.set_xlabel('Длина метки')
    ax.set_ylabel('Число вершин с такой меткой')
    plt.bar(pos, alphab, width=0.8, color='b', align='center')
    plt.title('Число классов с равным числом повторов')
    plt.show()
    return classes_subtreeid_nodes, classes_subtreeid_nodes_list


def main():
    # merge_in_file()
    # create_needed_directories()
    # sort_the_data()
    # pick_new_sentences()
    # draw_histogram()
    start = time.time()
    trees_full_df, trees_df_filtered, long_df = read_data()
    replace_time_constructions(trees_df_filtered)
    replace_time_constructions(trees_full_df)
    replace_time_constructions(long_df)
    print('Time on reading the data: ' + str(time.time() - start))
    part_of_speech_node_id = dict(trees_full_df[['lemma', 'upostag']].groupby(['lemma', 'upostag']).groups.keys())

    # get all lemmas and create a dictionary to map to numbers
    dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    dict_lemmas_full = {lemma: [index] for index, lemma in
                        enumerate(dict.fromkeys(trees_full_df['lemma'].to_list()), 1)}
    dict_lemmas_rev = {index[0]: lemma for lemma, index in dict_lemmas_full.items()}
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    dict_rel_rev = {v: k for k, v in dict_rel.items()}

    # dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(long_df['lemma'].to_list()), 1)}
    # dict_form_lemma = dict(zip(long_df['form'].to_list(), long_df['lemma'].to_list()))
    # dict_lemmas_full = {lemma: [index] for index, lemma in
    #                     enumerate(dict.fromkeys(trees_full_df['lemma'].to_list()), 1)}
    # dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(long_df['deprel'].to_list()))}
    # dict_rel_rev = {v: k for k, v in dict_rel.items()}

    if RUN_WITH_W2V:
        start = time.time()
        if LOAD_TRAINED:
            load_trained_word2vec(dict_lemmas_full, dict_lemmas, part_of_speech_node_id)
        else:
            train_word2vec(trees_full_df)
            load_trained_word2vec(dict_lemmas_full, dict_lemmas, part_of_speech_node_id)
        print('Time on word2vec: ' + str(time.time() - start))

    start = time.time()
    long_df = long_df[:1223]
    whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel)
    # whole_tree = construct_tree(long_df, dict_lemmas, dict_rel)
    # write_tree_in_table(whole_tree)
    print('Time on constructing the tree: ' + str(time.time() - start))

    whole_tree.set_help_dict()
    # partition nodes by height
    start = time.time()
    whole_tree.calculate_heights()
    print('Time on calculating all heights: ' + str(time.time() - start))

    heights_dictionary = {whole_tree.get_node(node_id): heights for node_id, heights in
                          whole_tree.heights.items()}
    grouped_heights = defaultdict(list)
    for node_1, heights in heights_dictionary.items():
        for height in heights:
            grouped_heights[height].append(node_1)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])
    dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes)))

    # classes for partial repeats
    start = time.time()
    classes_part, classes_part_list = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)
    # write_tree_in_table(whole_tree)
    print('Time on calculating partial repeats: ' + str(time.time() - start))
    write_in_file(classes_part, classes_part_list, whole_tree)
    merge_in_file()
    gg = []

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


if __name__ == '__main__':
    main()

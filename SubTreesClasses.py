import time
from collections import defaultdict
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import read_data, replace_time_constructions
from Tree import Tree, Node, Edge
from Util import merge_in_file, filter_classes, write_classes_in_txt, filter_meaningless_classes, get_all_words, \
    label_classes, \
    squash_classes, \
    label_data_with_wiki, get_all_wikidata_entities, construct_db_tree
from Visualisation import draw_histogram
from W2Vprocessing import load_trained_word2vec, train_node2vec, train_node2vec_db
from const.Constants import *


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel, remapped_sent):
    # construct a tree with a list of edges and a list of nodes
    whole_tree = Tree()
    root_node = Node(0, 0)  # add root
    whole_tree.add_node(root_node)
    new_id_count = len(trees_df_filtered) + 1
    similar_lemmas_dict = {}
    global_similar_mapping = {}
    dict_lemmas_rev = {}
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
            sent = remapped_sent[row['sent_name']]
            pos_tag = row['upostag']
            pos_extended = row['feats']
            form = row['form']
            # create main node
            curr_lemma = curr_lemmas[0]
            whole_tree.create_new_node(main_id, curr_lemma, form, sent, pos_tag, pos_extended, weight, from_id)
            children = [main_id]
            edge_to_weight[main_id] = weight
            global_similar_mapping[main_id] = main_id
            # if lemma has additional values add additional nodes
            if len(curr_lemmas) > 1:
                for i in range(1, len(curr_lemmas)):
                    if curr_lemma not in whole_tree.dict_lemmas.keys():
                        whole_tree.dict_lemmas[curr_lemma] = {curr_lemmas[i]}
                    else:
                        whole_tree.dict_lemmas[curr_lemma].add(curr_lemmas[i])
                    if curr_lemmas[i] not in dict_lemmas_rev.keys():
                        dict_lemmas_rev[curr_lemmas[i]] = {curr_lemma}
                    else:
                        dict_lemmas_rev[curr_lemmas[i]].add(curr_lemma)
                    whole_tree.create_new_node(new_id_count, curr_lemmas[i], form, sent, pos_tag, pos_extended, weight, from_id)
                    edge_to_weight[new_id_count] = weight
                    if main_id not in similar_lemmas_dict.keys():
                        similar_lemmas_dict[main_id] = [new_id_count]
                    else:
                        similar_lemmas_dict[main_id].append(new_id_count)
                    # whole_tree.heights[new_id_count] = whole_tree.heights[main_id]
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
    whole_tree.dict_lemmas_rev = list({k: v for k, v in dict_lemmas_rev.items() if len(v) > 1}.values())
    return whole_tree


def set_node_depth(whole_tree, grouped_heights):
    for height, nodes in grouped_heights:
        if height != 0:
            for node in nodes:
                num_deep_children = sum([whole_tree.get_node(child_id).num_deep_children + 1 for child_id in whole_tree.get_children(node.id)])
                node.num_deep_children = num_deep_children


def add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes):
    all_parents = set()
    for lem, v_ids in grouped_lemmas.items():
        for v_id in list(v_ids):
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
                    lemmas_to_visit = [lem]
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


def get_comb_last_and_children(equal_nodes, equal_nodes_mapping, children_trees):
    list_for_combinations = set()
    prepared_k_2 = set()
    included_labels = []
    for child_tree in children_trees:
        if child_tree in [item for sublist in list(equal_nodes.values()) for item in
                          sublist] or child_tree in equal_nodes.keys():
            if child_tree in equal_nodes_mapping.keys():
                actual_label = equal_nodes_mapping[child_tree]
            else:
                actual_label = child_tree
            if actual_label not in included_labels:
                if actual_label in equal_nodes.keys():
                    list_for_combinations.add(tuple(equal_nodes[actual_label]))
                else:
                    list_for_combinations.add(tuple(equal_nodes[child_tree]))
                included_labels.append(actual_label)
        else:
            prepared_k_2.add(child_tree)
    return list_for_combinations, prepared_k_2


def compute_restricted_combinations(whole_tree, node_children_by_depth_dict):
    list_for_combinations_filtered = {}
    all_depths = set([key for dict in node_children_by_depth_dict for key in dict.keys() if key < WORD_LIMIT])
    for depth in all_depths:
        word_range = WORD_LIMIT - depth
        for dict in node_children_by_depth_dict:
            relevant_subtrees = [str(whole_tree.get_edge(child_id)[0].weight) + str(whole_tree.get_node(child_id).lemma)
                                 for depthh, child_ids in dict.items() if depthh < word_range for child_id in child_ids]
            if depth not in list_for_combinations_filtered.keys():
                list_for_combinations_filtered[depth] = [tuple(relevant_subtrees)]
            else:
                list_for_combinations_filtered[depth].append(tuple(relevant_subtrees))
    return list_for_combinations_filtered


def produce_combinations(whole_tree, k_2, v_id, str_sequence_help, equal_nodes, equal_nodes_mapping):
    subtree_v_id_dict = {str(whole_tree.get_edge(child_id)[0].weight) + str(whole_tree.get_node(child_id).lemma): child_id for child_id in whole_tree.get_children(v_id)}
    if len(equal_nodes) > 0:
        list_for_combinations, prepared_k_2 = get_comb_last_and_children(equal_nodes, equal_nodes_mapping, k_2[v_id])
        node_children_by_depth_dict = []  # for each child node stores a dict {depth of alias node: [alias node label]}
        for comb_list in list(list_for_combinations):
            limit_children_dict_local = {}
            for subtree_label in comb_list:
                node_id = subtree_v_id_dict[subtree_label]
                num_children = whole_tree.get_node(node_id).num_deep_children
                if num_children not in limit_children_dict_local.keys():
                    limit_children_dict_local[num_children] = [node_id]
                else:
                    limit_children_dict_local[num_children].append(node_id)
            node_children_by_depth_dict.append(limit_children_dict_local)
        list_for_combinations_filtered = compute_restricted_combinations(whole_tree, node_children_by_depth_dict)
        combinations_repeated = []
        for value in list_for_combinations_filtered.values():
            combinations_repeated.extend(list(product(*(value))))
        # combinations_repeated = list(product(*(list_for_combinations)))
        all_combinations = []
        for l in combinations_repeated:
            if len(prepared_k_2) > 0:
                merged = list(l) + list(prepared_k_2)
                upper_bound_init = min(len(merged), WORD_LIMIT)
                upper_bound = get_upper_bound(whole_tree, upper_bound_init, merged, subtree_v_id_dict)
                all_combinations.extend(list(combinations(merged, i)) for i in range(1, upper_bound + 1))
            else:
                upper_bound_init = min(len(list(l)), WORD_LIMIT)
                upper_bound = get_upper_bound(whole_tree, upper_bound_init, list(l), subtree_v_id_dict)
                all_combinations.extend(list(combinations(list(l), i)) for i in range(1, upper_bound + 1))
    else:
        list_for_combinations = k_2[v_id]
        upper_bound_init = min(len(list_for_combinations), WORD_LIMIT)
        upper_bound = get_upper_bound(whole_tree, upper_bound_init, list_for_combinations, subtree_v_id_dict)
        all_combinations = [list(combinations(list_for_combinations, i)) for i in
                            range(1, upper_bound + 1)]
    all_combinations_flat = set([comb for comb_list in all_combinations for comb in comb_list])
    all_combinations_filtered = [tup for tup in all_combinations_flat if sum([whole_tree.get_node(subtree_v_id_dict[subtree]).num_deep_children for subtree in tup]) < WORD_LIMIT]
    return get_strings_from_combinations(all_combinations_filtered, str_sequence_help)


def get_upper_bound(whole_tree, upper_bound_init, list_subtrees, subtree_v_id_dict):
    upper_bound = upper_bound_init
    count = upper_bound_init
    all_depths = [whole_tree.get_node(subtree_v_id_dict[subtree]).num_deep_children for subtree in list_subtrees]
    if sum(all_depths) > WORD_LIMIT:
        upper_bound -= 1
    partial_sums = []
    for index in range(len(all_depths)):
        partial_sums.append(sum([depth for ind, depth in enumerate(all_depths) if ind != index]))
    if not all(partial_sum < WORD_LIMIT for partial_sum in partial_sums):
        upper_bound -= 1
    return upper_bound


def get_strings_from_combinations(all_combinations, str_sequence_help):
    all_combinations_labels = set()
    for comb in all_combinations:
        # for tup in comb:
        # combs = sorted(tup)
        combs = sorted(comb)
        hashcode_label = hash(tuple(combs))
        all_combinations_labels.add(hashcode_label)
        if hashcode_label not in str_sequence_help.keys() and len(set(combs)) == len(combs):  # 2nd condition is to filter out duplicates
            str_sequence_help[hashcode_label] = [combs.copy()]
    return all_combinations_labels


def get_nodeid_repeats(filtered_combination_ids, str_sequence_help, duplicate_combs):
    dict_nodeid_comb = {}
    for subtree_label in str_sequence_help.keys():
        if subtree_label in filtered_combination_ids.keys():
            for v_i in filtered_combination_ids[subtree_label]:
                if len(str_sequence_help.get(subtree_label)) > 1:
                    subtree_comb = duplicate_combs[v_i]
                else:
                    subtree_comb = str_sequence_help.get(subtree_label)[0]
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
        child_node = whole_tree.get_node(child)
        if child in old_node_new_nodes.keys() or child in whole_tree.similar_lemmas.keys():
            edge_to_child = whole_tree.get_edge(child)[0]
            w = str(edge_to_child.weight)
            actual_label = w + str(child_node.lemma)
            merge = []
            if child in old_node_new_nodes.keys():
                merge = extend_equal_nodes_mapping(w, old_node_new_nodes[child], function_identity, equal_nodes_mapping,
                                                   actual_label)
            if child in whole_tree.similar_lemmas.keys():
                merge_additional = extend_equal_nodes_mapping(w, whole_tree.similar_lemmas[child],
                                                              function_lemma_getter,
                                                              equal_nodes_mapping, actual_label)
                if len(merge) != 0:
                    merge.extend(merge_additional)
                else:
                    merge = merge_additional.copy()
            merge.insert(0, actual_label)
            if actual_label not in equal_nodes.keys():
                equal_nodes[actual_label] = merge
            else:
                equal_nodes[actual_label].extend(merge)
    return equal_nodes


def find_subtree_children(whole_tree, classes_similar_mapping, children, subtree, lemma_nodeid_dict):  # , equal_nodes_mapping)
    subtree_children = []
    for subtree_node in subtree:
        if subtree_node not in lemma_nodeid_dict.keys():  # or subtree_node in equal_nodes_mapping.keys():
            target_child = [whole_tree.get_target_from_children(children, subtree_node)]
        else:
            equal_set = set(lemma_nodeid_dict[subtree_node])
            intersection = equal_set & children
            if len(intersection) != 0:
                target_child = list(intersection)
            else:
                children_nodes = whole_tree.get_children_nodes(children)
                if subtree_node in children_nodes.keys():
                    target_child = [children_nodes[subtree_node]]
                else:
                    children_nodes_2 = whole_tree.get_children_nodes(equal_set)
                    target_child = [children_nodes_2[subtree_node]]
        for target_ch in target_child: # there CAN be multiple target children
            subtree_children.append(target_ch)
    target_child_filtered = []
    nonrepeating_main_nodes = []
    for tc in subtree_children:
        init_node_id = classes_similar_mapping[tc]
        if init_node_id not in nonrepeating_main_nodes:
            nonrepeating_main_nodes.append(init_node_id)
            target_child_filtered.append(tc)
    subtree_children = target_child_filtered
    if len(subtree_children) > len(subtree):
        lemma_ids_dict_local = {}
        for subtree_child in subtree_children:
            lemma = whole_tree.get_node(subtree_child).lemma
            if lemma not in lemma_ids_dict_local.keys():
                lemma_ids_dict_local[lemma] = [subtree_child]
            else:
                lemma_ids_dict_local[lemma].append(subtree_child)
        valid_ids = list({k: v for k, v in lemma_ids_dict_local.items() if len(v) == 1}.values())
        valid_ids_flatmapped = [valid_id for valid_list in valid_ids for valid_id in valid_list]
        invalid_entries = {k: v for k, v in lemma_ids_dict_local.items() if len(v) > 1}
        for _, ids in invalid_entries.items():
            valid_id = set(ids) - whole_tree.additional_nodes
            if len(valid_id) > 0:
                valid_ids_flatmapped.append(valid_id.pop())
        subtree_children = valid_ids_flatmapped
    return subtree_children


def find_deep_subtree_children(whole_tree, subtree_children, classes_subtreeid_nodes_list):
    subtree_deep_children = set()
    for subtree_lemma in list(map(lambda x: whole_tree.get_node(x).lemma, subtree_children)):
        if subtree_lemma in classes_subtreeid_nodes_list.keys():
            subtree_deep_children.update(classes_subtreeid_nodes_list[subtree_lemma])
    subtree_deep_children.update(subtree_children)
    return subtree_deep_children


def insert_node_in_tree(whole_tree, existing_node, id_count, subtree_new_label, lemma_nodeid_dict, old_node_new_nodes,
                     edge_to_curr, node_id, curr_height, classes_similar_mapping, num_deep_children):
    # add a new node with a new lemma
    new_node = Tree.copy_node_details(existing_node, id_count)
    new_node.num_deep_children = num_deep_children
    new_node.lemma = subtree_new_label
    whole_tree.add_node_to_dict(new_node)
    whole_tree.global_similar_mapping[new_node.id] = new_node.id
    # whole_tree.global_similar_mapping[new_node.id] = existing_node.id
    classes_similar_mapping[new_node.id] = existing_node.id
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
    whole_tree.heights[new_node.id] = [curr_height]
    return new_node


def add_additional_child_nodes(additional_child_nodes, lemma_nodeid_dict):
    for additional_child_lemma, child_id in additional_child_nodes.items():
        if additional_child_lemma not in lemma_nodeid_dict.keys():
            lemma_nodeid_dict[additional_child_lemma] = {child_id}
        else:
            lemma_nodeid_dict[additional_child_lemma].add(child_id)


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


def add_new_subtree_label(unique_subtrees_mapped_global_subtree_lemma, lemma, lemma_count, tree_label,
                          str_sequence_help):
    new_hash = hash((str(lemma),) + tuple(str_sequence_help[tree_label][0]))
    if new_hash not in unique_subtrees_mapped_global_subtree_lemma.keys():
        unique_subtrees_mapped_global_subtree_lemma[new_hash] = lemma_count
        lemma_count += 1
    return lemma_count


def check_if_has_with_no_children(whole_tree, subtree_children):
    for child in subtree_children:
        if len(whole_tree.get_children(child)) == 0:
            return True
    return False


def compute_part_subtrees(whole_tree, lemma_count, grouped_heights):
    init_labels = {}
    classes_similar_mapping = whole_tree.global_similar_mapping.copy()  # dict for linking initial subtrees with its similar copies
    classes_subtreeid_nodes = {}  # dict for storing found subtrees
    classes_subtreeid_nodes_list = {}  # dict for storing nodes included in each repeat
    unique_subtrees_mapped_global_subtree_lemma = {}  # global dict for storing {hash of repeat: lemma_id} pairs
    old_node_new_nodes = {}  # dict for storing {old node id: [subtree's new label]} pairs
    equal_nodes_mapping = {}  # dict for storing {new subtree label: actual label (mapped)}
    k_2 = {}  # dict for storing children labels for nodes: {parent node id: [children label (edge_to.weight + lemma)]}
    lemma_nodeid_dict = {}  # stores a set of node_ids for each lemma, needed for searching of target children of a subtree
    # additional dicts needed to track number of unique lemmas in a sentence for not adding a duplicate node
    subtree_sent_dict = {}
    subtree_node_id_dict = {}
    subtree_node_id_children = {}  # 1st level children (node ids) for a root (node id) of a subtree
    nonrepeating_subtrees = {}
    classes_subtreeid_nodes_heights = {}
    subtree_hash_sent_dict = {}
    id_count = sorted([node.id for node in whole_tree.nodes], reverse=True)[0] + 1 # id increment for new nodes
    for nodes in grouped_heights:
        curr_height = nodes[0]
        print(curr_height)
        start = time.time()
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for id, lemma in id_lemma_dict.items():
            grouped_lemmas[lemma].append(id)
        all_parents = add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes)
        additional_child_nodes = add_additional_children_to_parents(k_2, whole_tree, all_parents)
        add_additional_child_nodes(additional_child_nodes, lemma_nodeid_dict)
        add_grouped_lemmas_to_dict(whole_tree, grouped_lemmas, lemma_nodeid_dict)
        if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
            filtered_groups = {k: v for k, v in grouped_lemmas.items() if len(v) > 1}
            unique_groups = {k: v for k, v in grouped_lemmas.items() if len(v) == 1}
            unique_groups_keys_set = set(unique_groups.keys())
            unique_groups_keys_set_len = len(unique_groups_keys_set)
            additional_groups = {}
            for repeat_group_set in whole_tree.dict_lemmas_rev.copy():
                if len(repeat_group_set) != 0 and len(unique_groups_keys_set - repeat_group_set) == unique_groups_keys_set_len - len(repeat_group_set):
                    target_list = [unique_groups[repeat_lemma][0] for repeat_lemma in repeat_group_set]
                    additional_groups[repeat_group_set.pop()] = target_list
            filtered_groups_extended = {**filtered_groups, **additional_groups}
            for lemma, ids in filtered_groups_extended.items():
                combination_ids = {}
                str_sequence_help = {}
                duplicate_combs = {}  # for several cases
                # generate combinations
                for v_id in ids:
                    equal_nodes = collect_equal_nodes(whole_tree, v_id, old_node_new_nodes, equal_nodes_mapping)
                    all_combinations_str_joined = produce_combinations(whole_tree, k_2, v_id, str_sequence_help, equal_nodes, equal_nodes_mapping)
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]
                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                for subtree_hash, v_ids in filtered_combination_ids.items():
                    for v_id in v_ids:
                        if v_id in whole_tree.created:
                            sent = whole_tree.get_node(v_id).sent_name
                            if subtree_hash not in subtree_hash_sent_dict.keys():
                                subtree_hash_sent_dict[subtree_hash] = {sent}
                            else:
                                subtree_hash_sent_dict[subtree_hash].add(sent)
                for tree_label, node_list in filtered_combination_ids.items():
                    if tree_label in str_sequence_help.keys():
                        lemma_count = add_new_subtree_label(unique_subtrees_mapped_global_subtree_lemma, lemma, lemma_count,
                                                            tree_label, str_sequence_help)
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help, duplicate_combs)
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    existing_node = whole_tree.get_node(node_id)
                    curr_sent = whole_tree.get_node(node_id).sent_name
                    edge_to_curr = whole_tree.get_edge(node_id)[0]
                    children = whole_tree.get_children(node_id)
                    for subtree in node_subtrees:
                        old_hash = hash(tuple(subtree))
                        new_hash = hash((str(lemma),) + tuple(str_sequence_help[old_hash][0]))
                        subtree_new_label = unique_subtrees_mapped_global_subtree_lemma[new_hash]
                        subtree_children = find_subtree_children(whole_tree, classes_similar_mapping, children, subtree,
                                                                 lemma_nodeid_dict)
                        num_deep_children = sum([whole_tree.get_node(child_id).num_deep_children + 1 for child_id in subtree_children])
                        if num_deep_children < WORD_LIMIT:
                            # check a node only for nonrepeating case: if subtree hasn't been seen yet or when it has been seen before, but in other sentences
                            # this condition is the main one
                            main_condition = subtree_new_label not in subtree_sent_dict.keys() or existing_node.sent_name not in subtree_sent_dict[subtree_new_label]
                            help_condition = False
                            if not main_condition:
                                target_children_hashs = []
                                if subtree_new_label in subtree_node_id_dict.keys():
                                    for n_id in subtree_node_id_dict[subtree_new_label]:
                                        if n_id in subtree_node_id_children.keys():
                                            for child_id in subtree_node_id_children[n_id]:
                                                target_children_hashs.append(child_id)
                                help_condition = hash(tuple(subtree_children)) not in target_children_hashs
                            if main_condition or help_condition:
                                if len(subtree_children) > 0:
                                    if old_hash in subtree_hash_sent_dict.keys():
                                        sentences = subtree_hash_sent_dict[old_hash]
                                    else:
                                        sentences = set()
                                    has_with_no_children = False
                                    if curr_height != 1:
                                        has_with_no_children = check_if_has_with_no_children(whole_tree, subtree_children)
                                    deep_subtrees = [subtree_child for subtree_child in subtree_children if
                                                     whole_tree.get_node(
                                                         subtree_child).lemma in classes_subtreeid_nodes.keys()]
                                    init_subtree_mapped = []
                                    for subtree_child in subtree_children:
                                        if subtree_child not in deep_subtrees:
                                            init_subtree_mapped.append(classes_similar_mapping[subtree_child])
                                        else:
                                            init_subtree_mapped.append(subtree_child)

                                    if subtree_new_label not in subtree_sent_dict.keys():
                                        subtree_sent_dict[subtree_new_label] = [existing_node.sent_name]
                                    else:
                                        subtree_sent_dict[subtree_new_label].append(existing_node.sent_name)

                                    subtree_mapped = [whole_tree.global_similar_mapping[subtree_child] for subtree_child in
                                                      subtree_children]
                                    initial_label = tuple(sorted(init_subtree_mapped))  # label of a subtree without mapping
                                    mapped_label = tuple(sorted(subtree_mapped))  # label of a subtree where each w2v-node is mapped to the original
                                    has_deep_subtrees = len(deep_subtrees) > 0
                                    #
                                    if not has_with_no_children or (
                                            has_with_no_children and initial_label in nonrepeating_subtrees.keys() and len(sentences) != 0 and len(sentences - nonrepeating_subtrees[initial_label]) != 0) \
                                            or initial_label == mapped_label:
                                        if (mapped_label not in nonrepeating_subtrees.keys() and not has_deep_subtrees) or \
                                                (mapped_label in nonrepeating_subtrees.keys() and len(sentences - nonrepeating_subtrees[mapped_label]) != 0) \
                                                or has_deep_subtrees and \
                                                (initial_label not in nonrepeating_subtrees.keys()
                                                 or initial_label in nonrepeating_subtrees.keys() and len(sentences) != 0 and (len(sentences - nonrepeating_subtrees[initial_label]) != 0)):
                                            subtree_deep_children = find_deep_subtree_children(whole_tree, subtree_children,
                                                                                               classes_subtreeid_nodes_list)
                                            only_active = set()
                                            for node in subtree_deep_children:
                                                if whole_tree.get_node(node).sent_name == curr_sent:
                                                    only_active.add(node)
                                            nonrepeating = []
                                            only_active_filtered = []
                                            if len(only_active) != 0:
                                                active_heights = {active_id: whole_tree.heights[active_id] for active_id in only_active}
                                                max_height = max(active_heights.values())
                                                ids_with_max_height = {k: v for k, v in active_heights.items() if v == max_height}
                                                for root_subtree_id, height in ids_with_max_height.items():
                                                    if node_id not in list(map(lambda x: x.node_from, whole_tree.get_edge(root_subtree_id))):
                                                        children_to_remove = whole_tree.dfs_subtree(root_subtree_id, only_active)
                                                        only_active = only_active - set(children_to_remove)
                                            for active in only_active:
                                                if classes_similar_mapping[active] not in nonrepeating:
                                                    only_active_filtered.append(active)
                                                    nonrepeating.append(classes_similar_mapping[active])
                                            initial_label_deep = tuple(sorted([classes_similar_mapping[subtree_child] for subtree_child in only_active_filtered]))
                                            if not has_deep_subtrees or (
                                                    has_deep_subtrees and initial_label_deep not in nonrepeating_subtrees.keys()):
                                                # create a new node for a repeating subtree
                                                new_node = insert_node_in_tree(whole_tree, existing_node, id_count,
                                                                               subtree_new_label,
                                                                               lemma_nodeid_dict, old_node_new_nodes,
                                                                               edge_to_curr, node_id,
                                                                               curr_height, classes_similar_mapping,
                                                                               num_deep_children)
                                                id_count += 1

                                                if subtree_new_label not in subtree_node_id_dict.keys():
                                                    subtree_node_id_dict[subtree_new_label] = [new_node.id]
                                                else:
                                                    subtree_node_id_dict[subtree_new_label].append(new_node.id)
                                                if new_node.id not in subtree_node_id_children.keys():
                                                    subtree_node_id_children[new_node.id] = [hash(tuple(subtree_children))]
                                                else:
                                                    subtree_node_id_children[new_node.id].append(
                                                        hash(tuple(subtree_children)))

                                                if has_deep_subtrees or curr_height == 1:
                                                    nonrepeating_subtrees[initial_label_deep] = sentences
                                                    if initial_label not in nonrepeating_subtrees.keys():
                                                        nonrepeating_subtrees[initial_label] = sentences
                                                elif mapped_label not in nonrepeating_subtrees.keys():
                                                    nonrepeating_subtrees[mapped_label] = sentences

                                                Tree.add_new_edges(whole_tree, new_node.id, subtree_children)
                                                # assign class
                                                if subtree_new_label not in classes_subtreeid_nodes.keys():
                                                    classes_subtreeid_nodes[subtree_new_label] = [new_node.id]
                                                else:
                                                    classes_subtreeid_nodes[subtree_new_label].append(new_node.id)
                                                # add subtree children to this repeat, this is needed for dfs when all repeats are found
                                                if len(only_active_filtered) == 0:
                                                    only_active_filtered = subtree_children
                                                if subtree_new_label not in classes_subtreeid_nodes_list.keys():
                                                    classes_subtreeid_nodes_list[subtree_new_label] = set(
                                                        only_active_filtered)
                                                else:
                                                    classes_subtreeid_nodes_list[subtree_new_label].update(
                                                        only_active_filtered)
                                                if subtree_new_label not in classes_subtreeid_nodes_heights.keys():
                                                    classes_subtreeid_nodes_heights[subtree_new_label] = curr_height
                                                classes_subtreeid_nodes_list[subtree_new_label].add(new_node.id)
                                                init_labels[new_node.id] = initial_label
        print(time.time() - start)
    classes_subtreeid_nodes_heights = dict(sorted(classes_subtreeid_nodes_heights.items(), key=lambda item: item[0]))
    classes_subtreeid_nodes_sorted = {k: classes_subtreeid_nodes[k] for k in classes_subtreeid_nodes_heights.keys()}
    classes_subtreeid_nodes_sorted = {k: v for k, v in classes_subtreeid_nodes_sorted.items() if
                               len(v) > 1}  # TODO: why do len=1 entries even appear here??
    return classes_subtreeid_nodes_sorted, classes_subtreeid_nodes_list


def read_data_to_df():
    trees_df_filtered = read_data()
    replace_time_constructions(trees_df_filtered)
    return trees_df_filtered


def calculate_repeats_helper(whole_tree):
    heights_dictionary = {whole_tree.get_node(node_id): heights for node_id, heights in
                          whole_tree.heights.items()}
    grouped_heights = defaultdict(list)
    for node_1, heights in heights_dictionary.items():
        for height in heights:
            grouped_heights[height].append(node_1)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])
    dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes))) + 1
    for node in whole_tree.nodes:
        if node.id not in whole_tree.heights.keys():
            whole_tree.heights[node.id] = whole_tree.heights[whole_tree.global_similar_mapping[node.id]]
    set_node_depth(whole_tree, grouped_heights)
    classes_part, classes_part_list = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)
    return classes_part, classes_part_list


def postprocess_and_label(classes_part, classes_part_list, whole_tree, dict_lemmas_full, trees_df_filtered, remapped_sent):
    dict_form_lemma_str = dict(zip(trees_df_filtered['form'].to_list(), trees_df_filtered['lemma'].to_list()))
    dict_form_lemma_int = {k: dict_lemmas_full[v][0] for k, v in dict_form_lemma_str.items()}
    remapped_sent_rev = {index: sent_name for sent_name, index in remapped_sent.items()}

    classes_wordgroups_filtered, classes_sents_filtered = filter_classes(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_form_lemma_str)
    meaningful_classes, meaningless_classes = filter_meaningless_classes(classes_wordgroups_filtered,
                                                                         dict_form_lemma_str)
    classes_words, classes_count_passive_verbs = get_all_words(meaningful_classes, dict_form_lemma_str)
    class_labels = label_classes(classes_words, classes_count_passive_verbs)
    meaningful_classes_filtered = dict(sorted(meaningful_classes.items(), key=lambda x: len(x[1]), reverse=True))
    dict_lemmas_full_edit = {v[0]: set(v) for k, v in dict_lemmas_full.items()}
    for lemma, sim_lemmas in dict_lemmas_full_edit.items():
        new_temp = dict_lemmas_full_edit[lemma].copy()
        for sim_lemma in sim_lemmas:
            new_temp.update(dict_lemmas_full_edit[sim_lemma])
        dict_lemmas_full_edit[lemma] = new_temp.copy()
    dict_lemmas_full_extended_2 = {k: tuple(sorted(list(v))) for k, v in dict_lemmas_full_edit.items()}
    res = defaultdict(list)
    for key, val in sorted(dict_lemmas_full_extended_2.items()):
        res[val].append(key)
    new_labels = {v: k for k, v in enumerate(res.keys())}
    dict_lemmas_full_new_labels = {k: new_labels[v] for k, v in dict_lemmas_full_extended_2.items()}
    meaningful_classes_filtered_squashed, new_classes_mapping = squash_classes(whole_tree, meaningful_classes_filtered, dict_lemmas_full_new_labels, dict_form_lemma_int)
    meaningful_classes_filtered_squashed_sort = dict(sorted(meaningful_classes_filtered_squashed.items(), key=lambda x: len(x[1]), reverse=True))
    class_id_labels, new_class_id_label = label_data_with_wiki(meaningful_classes_filtered_squashed_sort, dict_form_lemma_str, class_labels, new_classes_mapping)
    return meaningful_classes_filtered_squashed_sort, class_id_labels, new_classes_mapping, new_class_id_label


def group_classes_by_labels(class_id_labels, class_labels):
    class_id_labels_full = class_id_labels.copy()
    # class_id_labels_full = {}
    label_classes = {}
    for class_id, labels in class_id_labels.items():
        # if labels is not None:
        #     class_id_labels_full[class_id] = labels
        for label in labels:
            if label not in label_classes.keys():
                label_classes[label] = [class_id]
            else:
                label_classes[label].append(class_id)
    class_labels_filtered = {k: v for k, v in class_labels.items() if len(v) > 0}
    for class_id, labels in class_labels_filtered.items():
        labels_fixed = [label.lower() for label in labels]
        if class_id not in class_id_labels_full.keys():
            class_id_labels_full[class_id] = labels
        else:
            unique = class_id_labels_full[class_id].union(labels_fixed)
            class_id_labels_full[class_id] = unique
            # if unique is not None:
            #     class_id_labels_full[class_id] = unique
            # else:
            #     omg = []
        for label in labels_fixed:
            # label_fixed = label.lower()
            if label not in label_classes.keys():
                label_classes[label] = [class_id]
            else:
                label_classes[label].append(class_id)
    num_classes_labeled = len(set(list(class_id_labels.keys())).union(set(list(class_labels_filtered.keys()))))
    label_classes_sorted = dict(sorted(label_classes.items(), key=lambda x: len(x[1])))
    return label_classes_sorted, num_classes_labeled, class_id_labels_full


def prepare_results(meaningful_classes, classes_sents_filtered):
    results_dict = {}
    result_sent_dict = {}
    for class_id, node_seq_list in meaningful_classes.items():
        for repeat_count, node_seq in enumerate(node_seq_list):
            joined_res_str = SPACE.join(list(map(lambda node: node.form, node_seq)))
            sent = classes_sents_filtered[class_id][repeat_count]
            if class_id not in results_dict.keys():
                results_dict[class_id] = [joined_res_str]
            else:
                results_dict[class_id].append(joined_res_str)
            if class_id not in result_sent_dict.keys():
                result_sent_dict[class_id] = [sent]
            else:
                result_sent_dict[class_id].append(sent)
    return results_dict, result_sent_dict


def annotate_data():
    # READ DATA
    start = time.time()
    trees_df_filtered = read_data_to_df()
    num_sentences = len(set(trees_df_filtered['sent_name'].to_list()))
    reading_time = (time.time() - start) / 60

    # LOAD VECTOR SPACE
    start = time.time()
    part_of_speech_node_id = dict(trees_df_filtered[['lemma', 'upostag']].groupby(['lemma', 'upostag']).groups.keys())
    dict_lemmas_full = {lemma: [index] for index, lemma in
                        enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    remapped_sent = {sent_name: index for index, sent_name in enumerate(dict.fromkeys(trees_df_filtered['sent_name'].to_list()), 1)}
    load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained_node2vec.model")  # node2vec
    w2v_time = (time.time() - start) / 60

    # CONSTRUCTING A TREE
    start = time.time()
    whole_tree = construct_tree(trees_df_filtered, dict_lemmas_full, dict_rel, remapped_sent)
    whole_tree.set_help_dict()
    whole_tree.calculate_heights()
    construct_tree_time = (time.time() - start) / 60

    # COMPUTE REPEATS
    start = time.time()
    classes_part, classes_part_list = calculate_repeats_helper(whole_tree)
    algo_time = (time.time() - start) / 60

    # POSTPROCESSING AND ANNOTATION
    start = time.time()
    meaningful_classes_filtered_squashed_sort, class_id_labels, new_classes_mapping, new_class_id_label = postprocess_and_label(classes_part, classes_part_list, whole_tree, dict_lemmas_full, trees_df_filtered, remapped_sent)
    postprocess_label_time = (time.time() - start) / 60

    # GROUP CLASSES BY LABELS
    label_classes_sorted, num_classes_labeled, class_id_labels_full = group_classes_by_labels(class_id_labels, new_class_id_label)
    remapped_sent_rev = {index: sent_name for sent_name, index in remapped_sent.items()}
    classes_sents_filtered = {k: list(map(lambda x: remapped_sent_rev[x[0].sent_name], v)) for k, v in meaningful_classes_filtered_squashed_sort.items()}
    results_dict, result_sent_dict = prepare_results(meaningful_classes_filtered_squashed_sort, classes_sents_filtered)

    overall_time = [reading_time, w2v_time, construct_tree_time, algo_time, postprocess_label_time]

    # NOT NEEDED FOR RUNTIME
    # dict_rel_rev = {v: k for k, v in dict_rel.items()}
    # write_classes_in_txt(whole_tree, meaningful_classes_filtered_squashed_sort, classes_sents_filtered,
    #                      new_classes_mapping, dict_rel_rev, RESULT_PATH, class_id_labels_full)
    # merge_in_file(RESULT_PATH, MERGED_PATH)
    return overall_time, label_classes_sorted, results_dict, num_sentences, num_classes_labeled, result_sent_dict, class_id_labels_full


def plot_repeat_len(results_dict):
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    count_dict = {}
    class_id_len = {k: len(v[0].split(SPACE)) for k, v in results_dict.items()}
    for class_id, leng in class_id_len.items():
        if leng not in count_dict.keys():
            count_dict[leng] = 1
        else:
            count_dict[leng] += 1
    alphab = list(count_dict.values())
    frequencies = list(count_dict.keys())
    pos1 = np.arange(len(frequencies))
    ax1 = plt.axes()
    ax1.set_xticks(pos1)
    ax1.set_xticklabels(frequencies)
    ax1.set_xlabel('Число слов в повторе')
    ax1.set_ylabel('Число классов')
    plt.bar(pos1, alphab, width=0.4, color='b')
    plt.title('Число классов с равным числом слов в повторе')
    plt.show()

def main():
    annotate_data()
    # whole_tree_plain = construct_db_tree()
    # label_data_with_wiki()
    # merge_in_file()
    # create_needed_directories()
    # sort_the_data()
    # pick_new_sentences()

    # draw_histogram()

    # lemmas_list = get_joint_lemmas('medicalTextTrees/all_lemmas_n2v.txt', 'medicalTextTrees/all_lemmas_w2v.txt')
    # dict1 = sort_already_logged('medicalTextTrees/merged_long.txt')
    # dict2 = sort_already_logged('medicalTextTrees/merged_wtf2.txt')
    # write_sorted_res_in_file(dict1, 'medicalTextTrees/merged_long_sorted.txt')
    # write_sorted_res_in_file(dict2, 'medicalTextTrees/merged_wtf2_sorted.txt')
    start = time.time()
    # trees_df_filtered, test_df = read_data() # TEST
    trees_df_filtered = read_data()
    # trees_df_filtered = trees_df_filtered[:1998]
    replace_time_constructions(trees_df_filtered)
    # replace_time_constructions(trees_full_df)
    # replace_time_constructions(long_df)
    print('Time on reading the data: ' + str(time.time() - start))
    part_of_speech_node_id = dict(trees_df_filtered[['lemma', 'upostag']].groupby(['lemma', 'upostag']).groups.keys())
    #
    # # get all lemmas and create a dictionary to map to numbers
    # dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    dict_lemmas_full = {lemma: [index] for index, lemma in
                        enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    # dict_lemmas_test = {lemma: [index] for index, lemma in
    #                     enumerate(dict.fromkeys(test_df['lemma'].to_list()), 1)}
    dict_lemmas_rev = {index[0]: lemma for lemma, index in dict_lemmas_full.items()}
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    dict_rel_rev = {v: k for k, v in dict_rel.items()}
    remapped_sent = {sent_name: index for index, sent_name in enumerate(dict.fromkeys(trees_df_filtered['sent_name'].to_list()), 1)}
    remapped_sent_rev = {index: sent_name for sent_name, index in remapped_sent.items()}

    # dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(long_df['lemma'].to_list()), 1)}
    dict_form_lemma_str = dict(zip(trees_df_filtered['form'].to_list(), trees_df_filtered['lemma'].to_list()))
    dict_form_lemma_int = {k: dict_lemmas_full[v][0] for k, v in dict_form_lemma_str.items()}
    # dict_lemmas_full = {lemma: [index] for index, lemma in
    #                     enumerate(dict.fromkeys(trees_full_df['lemma'].to_list()), 1)}
    # dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(long_df['deprel'].to_list()))}
    # dict_rel_rev = {v: k for k, v in dict_rel.items()}
    # visualize_embeddings(dict_lemmas_full, "trained_node2vec.model", "trained.model")
    if RUN_WITH_W2V:
        start = time.time()
        if LOAD_TRAINED:
            # load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id)
            load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained_node2vec.model")  # node2vec
            # load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained_node2vec")  # node2vec
            # load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained.model") # word2vec
        else:
            # train_word2vec(trees_df_filtered)
            whole_tree_plain = construct_tree(trees_df_filtered, dict_lemmas_full, dict_rel, remapped_sent) # graph is needed for node2vec
            whole_tree_plain.set_help_dict()
            # db_tree_edges = construct_db_tree(get_all_wikidata_entities())
            train_node2vec(whole_tree_plain, dict_lemmas_rev)
            # load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id) #dict_lemmas,
            # load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained_node2vec.model")
            load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, "trained_final.model")
        print('Time on word2vec: ' + str(time.time() - start))

    start = time.time()
    # long_df = long_df[:1223]
    # whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel)
    # whole_tree = construct_tree(test_df, dict_lemmas_test, dict_rel, remapped_sent) # TEST
    whole_tree = construct_tree(trees_df_filtered, dict_lemmas_full, dict_rel, remapped_sent)
    # whole_tree = new_test()
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
    dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes))) + 1
    for node in whole_tree.nodes:
        if node.id not in whole_tree.heights.keys():
            whole_tree.heights[node.id] = whole_tree.heights[whole_tree.global_similar_mapping[node.id]]
    set_node_depth(whole_tree, grouped_heights)
    start = time.time()
    classes_part, classes_part_list = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)
    # write_tree_in_table(whole_tree)
    print('Time on calculating partial repeats: ' + str(time.time() - start))
    # old
    # start = time.time()   # write_in_file_old(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_rel_rev)
    # merge_in_file(RESULT_PATH_OLD, MERGED_PATH_OLD)
    # print('Time on writing data old: ' + str(time.time() - start))
    # new
    start = time.time()
    classes_wordgroups_filtered, classes_sents_filtered = filter_classes(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_form_lemma_str)
    meaningful_classes, meaningless_classes = filter_meaningless_classes(classes_wordgroups_filtered, dict_form_lemma_str)
    classes_words, classes_count_passive_verbs = get_all_words(meaningful_classes, dict_form_lemma_str)
    class_labels = label_classes(classes_words, classes_count_passive_verbs)
    meaningful_classes_filtered = dict(sorted(meaningful_classes.items(), key=lambda x: len(x[1]), reverse=True))

    dict_lemmas_full_edit = {v[0]: set(v) for k, v in dict_lemmas_full.items()}
    for lemma, sim_lemmas in dict_lemmas_full_edit.items():
        new_temp = dict_lemmas_full_edit[lemma].copy()
        for sim_lemma in sim_lemmas:
            new_temp.update(dict_lemmas_full_edit[sim_lemma])
        dict_lemmas_full_edit[lemma] = new_temp.copy()
            # if lemma not in dict_lemmas_full_extended.keys():
            #     dict_lemmas_full_extended[lemma] = dict_lemmas_full_edit[sim_lemma].copy()
            # else:
            #     dict_lemmas_full_extended[lemma].update(dict_lemmas_full_edit[sim_lemma])
    dict_lemmas_full_extended_2 = {k: tuple(sorted(list(v))) for k, v in dict_lemmas_full_edit.items()}
    res = defaultdict(list)
    for key, val in sorted(dict_lemmas_full_extended_2.items()):
        res[val].append(key)
    new_labels = {v: k for k, v in enumerate(res.keys())}
    dict_lemmas_full_new_labels = {k: new_labels[v] for k, v in dict_lemmas_full_extended_2.items()}
    # dict_lemmas_similar = {sim_lemma_id: sim_lemma_ids[0] for _, sim_lemma_ids in dict_lemmas_full.items() for sim_lemma_id in sim_lemma_ids}
    meaningful_classes_filtered_squashed, new_classes_mapping = squash_classes(whole_tree, meaningful_classes_filtered, dict_lemmas_full_new_labels, dict_form_lemma_int)
    # # group classes by assigned labels
    # grouped_classes_by_label = defaultdict(list)
    # for key, values in sorted(class_labels.items()):
    #     for value in values:
    #         grouped_classes_by_label[value].append(key)
    meaningful_classes_filtered_squashed_sort = dict(sorted(meaningful_classes_filtered_squashed.items(), key=lambda x: len(x[1]), reverse=True))
    class_id_labels = label_data_with_wiki(meaningful_classes_filtered_squashed_sort, dict_form_lemma_str, class_labels, new_classes_mapping)

    # grouped_heights_2 = {}
    # for v_id, v_id_heights in whole_tree.heights.items():
    #     for v_id_height in v_id_heights:
    #         if v_id_height not in grouped_heights_2.keys():
    #             grouped_heights_2[v_id_height] = [v_id]
    #         else:
    #             grouped_heights_2[v_id_height].append(v_id)
    # deep_children = {}
    # for node in whole_tree.nodes:
    #     if node.num_deep_children not in deep_children.keys():
    #         deep_children[node.num_deep_children] = [tuple([node.id, node.form])]
    #     else:
    #         deep_children[node.num_deep_children].append(tuple([node.id, node.form]))
    classes_sents_filtered = {k:list(map(lambda x: remapped_sent_rev[x[0].sent_name], v)) for k, v in meaningful_classes_filtered_squashed_sort.items()}
    if WRITE_IN_FILES:
        # meaningful
        write_classes_in_txt(whole_tree, meaningful_classes_filtered_squashed_sort, classes_sents_filtered, new_classes_mapping, dict_rel_rev, class_labels, RESULT_PATH, class_id_labels)
        merge_in_file(RESULT_PATH, MERGED_PATH)
        # meaningless
        # write_classes_in_txt(whole_tree, meaningless_classes, classes_sents_filtered, new_classes_mapping, dict_rel_rev, {}, RESULT_PATH_FILTERED)
        # merge_in_file(RESULT_PATH_FILTERED, MERGED_PATH_FILTERED)
    print('Time on writing data new: ' + str(time.time() - start))
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Constants import *
from Tree import Tree, Node, Edge


def create_needed_directories():
    if not os.path.exists(CLASSIFIED_DATA_PATH):
        os.makedirs(CLASSIFIED_DATA_PATH)
    if not os.path.exists(LONG_DATA_PATH):
        os.makedirs(LONG_DATA_PATH)
    if not os.path.exists(STABLE_DATA_PATH):
        os.makedirs(STABLE_DATA_PATH)
    if not os.path.exists(MANY_ROOTS_DATA_PATH):
        os.makedirs(MANY_ROOTS_DATA_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)


def write_tree_in_table(whole_tree):
    with open(EXCEL_EDGES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_1, open(
            EXCEL_NODES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_2:
        writer_1 = csv.writer(csv_file_1, delimiter=',')
        writer_2 = csv.writer(csv_file_2, delimiter=',')
        writer_1.writerow(['Source', 'Target', 'Weight'])
        writer_2.writerow(['Id', 'Label'])
        for edge in whole_tree.edges:
            writer_1.writerow([edge.node_from, edge.node_to, edge.weight + 1])
        for node in whole_tree.nodes:
            writer_2.writerow([node.id, (node.lemma, node.form, node.sent_name)])


def read_med_data():
    df_disease = pd.read_excel("medicalTextTrees/support_med_data.xlsx", sheet_name='Болезнь')
    diseases = [x.lower() for x in list(df_disease.Name)]
    df_symp = pd.read_excel("medicalTextTrees/support_med_data.xlsx", sheet_name='Симптом')
    symptoms = [x.lower() for x in df_symp.Name.to_list()]
    df_doc = pd.read_excel("medicalTextTrees/support_med_data.xlsx", sheet_name='Врач')
    docs = [x.lower() for x in df_doc.Name.to_list()]
    df_drug = pd.read_excel("medicalTextTrees/support_med_data.xlsx", sheet_name='Лекарство')
    drugs = [x.lower() for x in df_drug.Name.to_list()]
    return diseases, symptoms, docs, drugs


def label_classes(classes_words, dict_form_lemma):
    diseases, symptoms, docs, drugs = read_med_data()
    dis = 'Болезнь'
    sym = 'Симптом'
    doc = 'Врач'
    drg = 'Лекарство'
    labels = {}
    for k, v in classes_words.items():
        freqs = dict(zip([dis, sym, doc, drg], [0, 0, 0, 0]))
        for v_i in v:
            lemma = dict_form_lemma[v_i]
            if lemma in diseases:
                freqs[dis] += 1
            elif lemma in symptoms:
                freqs[sym] += 1
            elif lemma in docs:
                freqs[doc] += 1
            elif lemma in drugs:
                freqs[drg] += 1
        sorted_d = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
        if max(sorted_d.values()) > 0:
            possible_label = next(iter(sorted_d))
        else:
            possible_label = ""
        labels[k] = possible_label
    return labels


# POST-PROCESSING
def merge_in_file():
    files = os.listdir(RESULT_PATH)
    files_sorted = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
    writer = open(MERGED_PATH, 'w', encoding='utf-8')
    try:
        for file in files_sorted:
            full_dir = os.path.join(RESULT_PATH, file)
            try:
                with open(full_dir, encoding='utf-8') as reader:
                    class_entries = reader.readlines()
            finally:
                reader.close()
            for entry in class_entries:
                writer.write(entry)
            writer.write('\n')
    finally:
        writer.close()


# POST-PROCESSING
def write_in_file(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_rel_rev):
    count = 0
    sent_unique_class_content = {}
    classes_wordgroups = {}
    classes_sents = {}
    surviving_strings_classes = {}
    for class_id, root_ids in classes_part.items():
        vertex_seq = {}
        # vertex_sub_seq = {}
        count += 1
        temp_sent_set = set()
        vertex_strings_dict = {}
        vertex_seq_ids = {}
        # vertex_sub_seq_ids = {}
        for vertex_id in root_ids:
            sent_id = whole_tree.get_node(vertex_id).sent_name
            temp_sent_set.add(sent_id)
            # sent_name = remapped_sent_rev[sent_id]
            # curr_height = max(whole_tree.heights[vertex])
            vertex_seq[vertex_id]= whole_tree.simple_dfs(vertex_id, classes_part_list[class_id])
        if len(vertex_seq.items()) > 0 and len(vertex_seq[list(vertex_seq)[0]]) > 1:
            for v_id, entries in vertex_seq.items():
                sent_name = remapped_sent_rev[entries[0][3]]
                for entry in entries:
                    node = whole_tree.get_node(entry[0])
                    if node.res_class is None:
                        node.res_class = count
                joined_res_str = SPACE.join(SPACE.join(list(map(lambda list_entry: '(' + str(dict_rel_rev[list_entry[4]]) + ') ' + str(list_entry[2]), entries))).split(SPACE)[1:])
                if class_id not in classes_wordgroups.keys():
                    classes_wordgroups[class_id] = [joined_res_str]
                    classes_sents[class_id] = [sent_name]
                else:
                    classes_wordgroups[class_id].append(joined_res_str)
                    classes_sents[class_id].append(sent_name)
                vertex_strings_dict[v_id] = joined_res_str
                vertex_seq_ids[v_id] = list(map(lambda list_entry: str(list_entry[4]) + str(list_entry[2]), entries))
                # sub_seq_ids = []
                # for sub_entries in vertex_sub_seq[v_id]:
                #     # joined_sub_res_str = SPACE.join(SPACE.join(list(map(lambda list_entry: '(' + str(dict_rel_rev[list_entry[4]]) + ') ' + str(list_entry[2]), sub_entries))).split(SPACE)[1:])
                #     ids_sub_extracted = list(map(lambda list_entry: str(list_entry[4]) + str(list_entry[2]), sub_entries))
                #     sub_seq_ids.append(ids_sub_extracted)
                # vertex_sub_seq_ids[v_id] = sub_seq_ids
        sent_label = tuple(temp_sent_set)
        if sent_label not in sent_unique_class_content.keys():
            sent_unique_class_content[sent_label] = set([tuple(i) for i in list(vertex_seq_ids.values())])
        else:
            already_checked = set()
            for v_id, new_array in vertex_seq_ids.items():
                new_array_label = tuple(new_array)
                if new_array_label not in already_checked:
                    completely_new = new_array_label not in sent_unique_class_content[sent_label]
                    if completely_new:
                        filter_condition = False
                        is_new_nested_in_existing = False
                        # extend_condition = False
                        ids_to_filter = set()
                        # sub_seq_ids = vertex_sub_seq_ids[v_id]
                        for existing_array in sent_unique_class_content[sent_label]:
                            is_existing_nested_in_new = all(entr in new_array for entr in existing_array) and existing_array != new_array
                            is_new_nested_in_existing = is_new_nested_in_existing or all(entr in existing_array for entr in new_array)
                            # is_sub_new_nested_in_existing = any(all(entr in existing_array for entr in sub_seq) for sub_seq in sub_seq_ids)
                            filter_condition = filter_condition or (is_existing_nested_in_new)# or is_sub_new_nested_in_existing)
                            # extend_condition = extend_condition or (is_existing_nested_in_new or is_sub_new_nested_in_existing)
                            if is_existing_nested_in_new:# or is_sub_new_nested_in_existing:
                                ids_to_filter.add(existing_array)
                        if filter_condition:
                            sent_unique_class_content[sent_label] = sent_unique_class_content[sent_label] - ids_to_filter
                        if not is_new_nested_in_existing:
                            sent_unique_class_content[sent_label].add(new_array_label)
                    already_checked.add(new_array_label)
                    surviving_strings_classes[new_array_label] = class_id
    classes_ids_filtered = [surviving_strings_classes[sent_group] for sent_groups in sent_unique_class_content.values() for sent_group in list(sent_groups)]
    classes_wordgroups_filtered = {k: v for k, v in classes_wordgroups.items() if k in classes_ids_filtered}
    classes_sents_filtered = {k: v for k, v in classes_sents.items() if k in classes_ids_filtered}
    if MERGE_IN_FILE:
        write_classes_in_txt(classes_wordgroups_filtered, classes_sents_filtered)
    return classes_wordgroups_filtered


def write_classes_in_txt(classes_wordgroups_filtered, classes_sents_filtered):
    count = 1
    for class_id, str_repeats in classes_wordgroups_filtered.items():
        filename = RESULT_PATH + '/%s.txt' % (str(count))
        try:
            with open(filename, 'w', encoding='utf-8') as filehandle:
                for repeat_count, str_repeat in enumerate(str_repeats):
                    filehandle.write("sent=%s: %s\n" % (classes_sents_filtered[class_id][repeat_count], str_repeat))
        finally:
            filehandle.close()
        count += 1

    # filename = RESULT_PATH + '/%s.txt' % (str(count))
    # try:
    #     with open(filename, 'w', encoding='utf-8') as filehandle:
    #         for _, value in vertex_seq.items():
    #             for val in value:
    #                 node = whole_tree.get_node(val[0])
    #                 if node.res_class is None:
    #                     node.res_class = count
    #             joined_res_str = SPACE.join(SPACE.join(list(map(lambda list_entry: '(' + str(dict_rel_rev[list_entry[4]]) + ') ' + str(list_entry[2]), value))).split(' ')[1:])
    #             temp_strings_set.add(joined_res_str)
    #             filehandle.write("len=%d h=%d sent=%s %s: %s\n" % (len(value), curr_height, value[0][1], remapped_sent_rev[value[0][3]], joined_res_str))
    # finally:
    #     filehandle.close()


# for vertex in v:
#     filename = RESULT_PATH + '/%s.txt' % (str(count))
#     curr_height = max(whole_tree.heights[vertex])
#     try:
#         with open(filename, 'w', encoding='utf-8') as filehandle:
#             filehandle.write("len=%d h=%d sent=%s %s: %s\n" % (len(entries), curr_height, entries[0][1], remapped_sent_rev[entries[0][3]], joined_res_str))
#     finally:
#         filehandle.close()

# def write_tree_in_table(whole_tree, dict_rel_rev, labels):
#     source_1 = 'medicalTextTrees/gephi_edges_import_word2vec.csv'
#     source_2 = 'medicalTextTrees/gephi_nodes_import_word2vec.csv'
#     with open(source_1, "w", newline='', encoding='utf-8') as csv_file_1, open(
#             source_2, "w", newline='', encoding='utf-8') as csv_file_2:
#         writer_1 = csv.writer(csv_file_1, delimiter=',')
#         writer_2 = csv.writer(csv_file_2, delimiter=',')
#         writer_1.writerow(['Source', 'Target', 'Weight'])
#         writer_2.writerow(['Id', 'Label', 'Class', 'Class_Label'])
#         included_nodes = list(filter(lambda x: x.res_class is not None, whole_tree.nodes))
#         included_ids = [node.id for node in included_nodes]
#         for edge in whole_tree.edges:
#             if edge.node_to in included_ids and edge.node_from in included_ids:
#                 writer_1.writerow([edge.node_from, edge.node_to, dict_rel_rev[edge.weight]])
#         for node in included_nodes:
#             writer_2.writerow(
#                 [node.id, (node.lemma, node.form, node.sent_name), node.res_class, labels[node.res_class]])


def draw_histogram():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('title', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    path = MERGED_PATH
    try:
        with open(path, encoding='utf-8') as reader:
            class_entries = reader.readlines()
    finally:
        reader.close()
    class_count = 1
    local_c = 0
    curr_len = 0
    group_len = {}
    str_len = {}
    for line in class_entries:
        if line == NEW_LINE:
            group_len[class_count] = local_c
            str_len[class_count] = curr_len
            local_c = 0
            class_count += 1
        else:
            words = [w for w in line.split(':')[1].split(NEW_LINE)[0].split(" ") if w != EMPTY_STR]
            curr_len = len(words) // 2 + 1
            local_c += 1
    res = {}
    for key, val in sorted(group_len.items()):
        if val not in res.keys():
            res[val] = 1
        else:
            res[val] += 1
    res_len = {}
    for key, val in sorted(str_len.items()):
        if val not in res_len.keys():
            res_len[val] = 1
        else:
            res_len[val] += 1

    res2 = dict(sorted(res.items(), key=lambda x: x[0]))
    alphab = list(res2.values())
    frequencies = list(res2.keys())

    pos = np.arange(len(frequencies))
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(frequencies)
    ax.set_xlabel('Размер класса')
    ax.set_ylabel('Число классов')
    plt.xticks(rotation=45)
    plt.bar(pos, alphab, width=0.9, color='b')
    plt.title('Число классов с равным числом повторов')
    plt.show()

    res_len = dict(sorted(res_len.items(), key=lambda x: x[0]))
    alphab = list(res_len.values())
    frequencies = list(res_len.keys())

    pos1 = np.arange(len(frequencies))
    ax1 = plt.axes()
    ax1.set_xticks(pos1)
    ax1.set_xticklabels(frequencies)
    ax1.set_xlabel('Объем класса')
    ax1.set_ylabel('Число классов')
    plt.bar(pos1, alphab, width=0.4, color='b')
    plt.title('Число классов с равным числом слов в повторе')
    plt.show()


def get_test_tree():
    test_tree = Tree()
    root_node = Node(0, 0, None, None)
    # add root
    Tree.add_node(test_tree, root_node)
    # add test nodes
    Tree.add_node(test_tree, Node(1, 18, None, 1))
    Tree.add_node(test_tree, Node(2, 20, None, 1))
    Tree.add_node(test_tree, Node(3, 3, None, 1))
    Tree.add_node(test_tree, Node(4, 19, None, 1))
    Tree.add_node(test_tree, Node(5, 8, None, 1))
    Tree.add_node(test_tree, Node(6, 18, None, 2))
    Tree.add_node(test_tree, Node(7, 20, None, 2))
    Tree.add_node(test_tree, Node(8, 3, None, 2))
    Tree.add_node(test_tree, Node(9, 7, None, 2))
    Tree.add_node(test_tree, Node(10, 19, None, 2))
    Tree.add_node(test_tree, Node(11, 18, None, 3))
    Tree.add_node(test_tree, Node(12, 20, None, 3))
    Tree.add_node(test_tree, Node(13, 7, None, 3))
    Tree.add_node(test_tree, Node(14, 19, None, 3))
    Tree.add_node(test_tree, Node(15, 8, None, 3))
    # add test edges
    Tree.add_edge(test_tree, Edge(0, 1, 0))
    Tree.add_edge(test_tree, Edge(0, 6, 0))
    Tree.add_edge(test_tree, Edge(0, 11, 0))
    Tree.add_edge(test_tree, Edge(1, 2, 1))
    Tree.add_edge(test_tree, Edge(6, 7, 1))
    Tree.add_edge(test_tree, Edge(11, 12, 1))
    Tree.add_edge(test_tree, Edge(2, 3, 4))
    Tree.add_edge(test_tree, Edge(7, 8, 4))
    Tree.add_edge(test_tree, Edge(2, 4, 9))
    Tree.add_edge(test_tree, Edge(7, 10, 9))
    Tree.add_edge(test_tree, Edge(12, 14, 9))
    Tree.add_edge(test_tree, Edge(7, 9, 10))
    Tree.add_edge(test_tree, Edge(12, 13, 10))
    Tree.add_edge(test_tree, Edge(2, 5, 20))
    Tree.add_edge(test_tree, Edge(12, 15, 20))
    return test_tree


def new_test():
    test_tree = Tree()
    root_node = Node(0, 0)
    # add root
    Tree.add_node(test_tree, root_node)
    # add test nodes
    Tree.add_node(test_tree, Node(1, lemma=18, sent_name=1, form=18))
    Tree.add_node(test_tree, Node(2, lemma=20, sent_name=1, form=20))
    Tree.add_node(test_tree, Node(3, lemma=3, sent_name=1, form=3))
    Tree.add_node(test_tree, Node(4, lemma=19, sent_name=1, form=19))
    Tree.add_node(test_tree, Node(5, lemma=5, sent_name=1, form=5))
    Tree.add_node(test_tree, Node(6, lemma=6, sent_name=1, form=6))
    Tree.add_node(test_tree, Node(7, lemma=8, sent_name=1, form=8))
    Tree.add_node(test_tree, Node(8, lemma=18, sent_name=2, form=18))
    Tree.add_node(test_tree, Node(9, lemma=20, sent_name=2, form=20))
    Tree.add_node(test_tree, Node(10, lemma=3, sent_name=2, form=3))
    Tree.add_node(test_tree, Node(11, lemma=4, sent_name=2, form=4))
    Tree.add_node(test_tree, Node(12, lemma=7, sent_name=2, form=7))
    Tree.add_node(test_tree, Node(13, lemma=19, sent_name=2, form=19))
    Tree.add_node(test_tree, Node(14, lemma=2, sent_name=2, form=2))
    Tree.add_node(test_tree, Node(15, lemma=18, sent_name=3, form=18))
    Tree.add_node(test_tree, Node(16, lemma=20, sent_name=3, form=20))
    Tree.add_node(test_tree, Node(17, lemma=7, sent_name=3, form=7))
    Tree.add_node(test_tree, Node(18, lemma=19, sent_name=3, form=19))
    Tree.add_node(test_tree, Node(19, lemma=8, sent_name=3, form=8))
    Tree.add_node(test_tree, Node(22, lemma=14, sent_name=3, form=14))
    # add test edges
    Tree.add_edge(test_tree, Edge(0, 1, 0))
    Tree.add_edge(test_tree, Edge(0, 8, 0))
    Tree.add_edge(test_tree, Edge(0, 15, 0))
    Tree.add_edge(test_tree, Edge(1, 2, 1))
    Tree.add_edge(test_tree, Edge(2, 3, 4))
    Tree.add_edge(test_tree, Edge(2, 4, 9))
    Tree.add_edge(test_tree, Edge(2, 7, 20))
    Tree.add_edge(test_tree, Edge(4, 5, 1))
    Tree.add_edge(test_tree, Edge(5, 6, 1))
    Tree.add_edge(test_tree, Edge(8, 9, 1))
    Tree.add_edge(test_tree, Edge(9, 10, 4))
    Tree.add_edge(test_tree, Edge(9, 12, 10))
    Tree.add_edge(test_tree, Edge(9, 13, 9))
    Tree.add_edge(test_tree, Edge(10, 11, 1))
    Tree.add_edge(test_tree, Edge(15, 16, 1))
    Tree.add_edge(test_tree, Edge(16, 17, 10))
    Tree.add_edge(test_tree, Edge(16, 18, 9))
    Tree.add_edge(test_tree, Edge(16, 19, 20))
    Tree.add_edge(test_tree, Edge(8, 14, 2))
    Tree.add_edge(test_tree, Edge(16, 22, 4))

    # additional
    Tree.add_node(test_tree, Node(20, lemma=14, sent_name=2, form=14))
    Tree.add_node(test_tree, Node(21, lemma=9, sent_name=2, form=9))
    Tree.add_node(test_tree, Node(23, lemma=21, sent_name=2, form=21))

    Tree.add_edge(test_tree, Edge(9, 20, 4))
    Tree.add_edge(test_tree, Edge(9, 21, 4))

    test_tree.additional_nodes = {20, 21}
    test_tree.similar_lemmas = {10: [20, 21]}

    test_tree.global_similar_mapping = {20: 10, 21: 10, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                                        11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 22: 22}
    return test_tree

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
    files = sorted(os.listdir(RESULT_PATH))
    writer = open(MERGED_PATH, 'w', encoding='utf-8')
    try:
        for file in files:
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
def write_in_file(classes_part, classes_part_list, whole_tree, remapped_sent_rev):
    count = 0
    classes_words = {}
    for k, v in classes_part.items():
        vertex_seq = {}
        count += 1
        for vertex in v:
            curr_height = max(whole_tree.heights[vertex])
            vertex_seq[vertex] = whole_tree.simple_dfs(vertex, classes_part_list[k])
            if len(vertex_seq.items()) > 0 and len(vertex_seq[list(vertex_seq)[0]]) > 1:
                filename = RESULT_PATH + '/results_%s.txt' % (str(k))
                try:
                    with open(filename, 'w', encoding='utf-8') as filehandle:
                        # target_indices = {v[0][3]: k for k, v in vertex_seq.items()}.values()
                        # vertex_seq_filtered = {k: v for k, v in vertex_seq.items() if k in target_indices}
                        # better print for testing
                        # for key, value in vertex_seq_filtered.items():
                        #     filehandle.write("%s: %s\n" % (key, value))
                        for _, value in vertex_seq.items():
                            for val in value:
                                node = whole_tree.get_node(val[0])
                                if node.res_class is None:
                                    node.res_class = count
                            words = list(map(lambda list_entry: list_entry[2], value))
                            if count not in classes_words.keys():
                                classes_words[count] = words
                            filehandle.write("%d %s %s: %s\n" % (curr_height, value[0][1], remapped_sent_rev[value[0][3]], SPACE.join(list(map(lambda list_entry: str(list_entry[2]), value)))))
                finally:
                    filehandle.close()
    return classes_words


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
    path = 'data/merged_extended.txt'
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
            curr_len = len(words)
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
    plt.bar(pos, alphab, width=0.8, color='b', align='center')
    plt.title('Число классов с равным числом повторов')
    plt.show()

    alphab = list(res_len.values())
    frequencies = list(res_len.keys())

    pos = np.arange(len(frequencies))
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(frequencies)
    ax.set_xlabel('Объем класса')
    ax.set_ylabel('Число классов')
    plt.bar(pos, alphab, width=0.4, color='b', align='center')
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

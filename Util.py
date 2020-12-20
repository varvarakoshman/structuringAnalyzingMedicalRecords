import os
import shutil

import pandas as pd

from Constants import *
from Tree import Tree, Node, Edge
import csv


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

# PRE-PROCESSING
# method splits input data in 3 datasets:
# 1) stable (ready to run an algorithm),
# 2) very long-read (sentences are too long and need to be split in 2 parts),
# 3) many-rooted (case for compound sentences and incorrect parser's results (Ex: 5 roots in a sentence of length 10)
# and copies files in corresponding directories
# fix 2) and 3) manually and then run the main algorithm, which will walk through these directories and add all files.
def sort_the_data():
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            this_df = this_df[this_df.deprel != 'PUNC']
        if this_df.groupby(this_df.deprel).get_group('ROOT').shape[0] > 1:
            shutil.copy(full_dir, MANY_ROOTS_DATA_PATH)
        elif this_df.shape[0] > 21:
            name_split = file.split(DOT)
            shutil.copy(os.path.join(ORIGINAL_DATA_PATH, DOT.join([name_split[0], name_split[1]])), LONG_DATA_PATH)
        else:
            shutil.copy(full_dir, STABLE_DATA_PATH)


# POST-PROCESSING
def merge_in_file():
    files = sorted(os.listdir(DATA_PATH))
    writer = open(MERGED_PATH, 'w', encoding='utf-8')
    try:
        for file in files:
            full_dir = os.path.join(DATA_PATH, file)
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
    Tree.add_node(test_tree, Node(1, lemma=18))
    Tree.add_node(test_tree, Node(2, lemma=20))
    Tree.add_node(test_tree, Node(3, lemma=3))
    Tree.add_node(test_tree, Node(4, lemma=19))
    Tree.add_node(test_tree, Node(5, lemma=5))
    Tree.add_node(test_tree, Node(6, lemma=6))
    Tree.add_node(test_tree, Node(7, lemma=8))
    Tree.add_node(test_tree, Node(8, lemma=18))
    Tree.add_node(test_tree, Node(9, lemma=20))
    Tree.add_node(test_tree, Node(10, lemma=3))
    Tree.add_node(test_tree, Node(11, lemma=4))
    Tree.add_node(test_tree, Node(12, lemma=7))
    Tree.add_node(test_tree, Node(13, lemma=19))
    Tree.add_node(test_tree, Node(14, lemma=2))
    Tree.add_node(test_tree, Node(15, lemma=18))
    Tree.add_node(test_tree, Node(16, lemma=20))
    Tree.add_node(test_tree, Node(17, lemma=7))
    Tree.add_node(test_tree, Node(18, lemma=19))
    Tree.add_node(test_tree, Node(19, lemma=8))
    Tree.add_node(test_tree, Node(22, lemma=14))
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
    Tree.add_node(test_tree, Node(20, lemma=14))
    Tree.add_node(test_tree, Node(21, lemma=9))

    Tree.add_edge(test_tree, Edge(9, 20, 4))
    Tree.add_edge(test_tree, Edge(9, 21, 4))

    test_tree.additional_nodes = {20, 21}
    test_tree.similar_lemmas = {10: [20, 21]}
    return test_tree

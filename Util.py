import os
from collections import defaultdict
import numpy as np

# input: map_dict - {v.id: str value of mapped string}
# returns: sorted_res - {v.id: str value of mapped string} in sorted order
import pandas as pd

from Tree import Tree, Node, Edge
import shutil
from Constants import *


def radix_sort(map_dict):
    if len(map_dict) == 0:
        return map_dict
    map_dict_rev = defaultdict(list)  # reversed: {str value of mapped string: [v.id]}
    for key, val in map_dict.items():
        map_dict_rev[val].append(key)
    n_digits = max(list(map(lambda x: len(x), map_dict_rev.keys())))  # max N of digits
    numbers = list(map_dict_rev.keys())
    b = numbers.copy()  # array for intermediate results_part_new
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


# input: remapped_dict - dictionary {v.id: int array of mapped string}
# returns: sorted_res - dictionary {v.id: str value of mapped string}
def sort_strings_inside(remapped_dict):
    sorted_res = {}
    for item in remapped_dict.items():
        sorted_array = quick_sort(item[1])
        sorted_res[item[0]] = "".join([str(digit) for digit in sorted_array])
    return sorted_res


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
        if this_df.groupby(this_df.deprel).get_group('ROOT').shape[0] > 1:
            shutil.copy(full_dir, MANY_ROOTS_DATA_PATH)
        elif this_df.shape[0] > 21:
            shutil.copy(full_dir, LONG_DATA_PATH)
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
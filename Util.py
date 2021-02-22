import os
from collections import defaultdict

import numpy as np

# input: map_dict - {v.id: str value of mapped string}
# returns: sorted_res - {v.id: str value of mapped string} in sorted order
from Constants import EMPTY_STR, RESULT_PATH, SPACE, MERGED_PATH
from Tree import Tree, Node, Edge


def radix_sort(map_dict):
    if len(map_dict) == 0:
        return map_dict
    map_dict_rev = defaultdict(list)  # reversed: {str value of mapped string: [v.id]}
    for key, val in map_dict.items():
        map_dict_rev[val].append(key)
    n_digits = max(list(map(lambda x: len(x), map_dict_rev.keys())))  # max N of digits
    numbers = list(map_dict_rev.keys())
    b = numbers.copy()  # array for intermediate results_part
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


# input: remapped_dict - dictionary {v.id: int array of mapped string}
# returns: sorted_res - dictionary {v.id: str value of mapped string}
def sort_strings_inside(remapped_dict):
    sorted_res = {}
    for v_id, strings in remapped_dict.items():
        sorted_array = sorted(strings)
        sorted_res[v_id] = EMPTY_STR.join([label for label in sorted_array])
    return sorted_res


def remap_s(str_dict):
    m = 0
    remapped = {}  # key : node id, value: mapped string
    already_seen = {}
    for v_id, string_labels in str_dict.items():
        for string_label in string_labels:
            if string_label in already_seen.keys():
                mapping = already_seen[string_label]
            else:
                m = m + 1
                already_seen[string_label] = m
                mapping = m
            if v_id not in remapped.keys():
                remapped[v_id] = [str(mapping)]
            else:
                remapped[v_id].append(str(mapping))
    # remapped = {item[0]: np.array([int(item[1][p]) for p in range(len(item[1]))]) for item in remapped.items()}
    return remapped


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
def write_in_file(classes_part, whole_tree, dict_rel_rev):
    count = 0
    classes_words = {}
    for k, v in classes_part.items():
        vertex_seq = {}
        count += 1
        for vertex in v:
            curr_height = whole_tree.heights[vertex]
            vertex_seq[vertex] = whole_tree.simple_dfs(vertex)
            if len(vertex_seq.items()) > 0 and len(vertex_seq[list(vertex_seq)[0]]) > 1:
                filename = RESULT_PATH + '/%s.txt' % (str(count))
                try:
                    with open(filename, 'w', encoding='utf-8') as filehandle:
                        for _, value in vertex_seq.items():
                            words = list(map(lambda list_entry: list_entry[2], value))
                            if count not in classes_words.keys():
                                classes_words[count] = words
                            filehandle.write("len=%d h=%d sent=%s %s: %s\n" % (len(value), curr_height, value[0][1], value[0][3],
                                SPACE.join(SPACE.join(list(map(lambda list_entry: '(' + str(dict_rel_rev[list_entry[4]]) + ') ' + str(list_entry[2]), value))).split(' ')[1:])))
                finally:
                    filehandle.close()
    return classes_words


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
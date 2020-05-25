from collections import defaultdict
import numpy as np


# input: map_dict - {v.id: str value of mapped string}
# returns: sorted_res - {v.id: str value of mapped string} in sorted order
from Tree import Tree, Node, Edge


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
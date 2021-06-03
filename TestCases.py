from SubTreesClasses import bfs_deep_similar
from Tree import Node, Tree, Edge


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


def test():
    dict_lemmas_full_edit = {
        0: [1, 2, 3],
        1: [4, 5, 6],
        2: [13, 4, 14],
        3: [15],
        4: [1, 7],
        5: [1, 7],
        6: [2, 8],
        7: [4, 5, 9],
        8: [6, 12],
        9: [10, 11],
        10: [],
        11: [],
        12: [13, 8],
        13: [],
        14: [5, 6],
        15: [16, 7],
        16: [17, 15],
        17: [18],
        18: []
    }
    deep_children = bfs_deep_similar(0, dict_lemmas_full_edit)
    print(deep_children)
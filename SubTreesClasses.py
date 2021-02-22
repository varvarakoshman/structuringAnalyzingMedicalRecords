import time
from collections import defaultdict

from Preprocessing import read_data
from Tree import Tree, Node, Edge
from Util import radix_sort, sort_strings_inside, remap_s, write_in_file, merge_in_file


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel):
    # construct a tree with a list of edges and a list of nodes
    whole_tree = Tree()
    # root_node = Node(0, 0, None, None)  # add root
    root_node = Node(0, 0)  # add root
    whole_tree.add_node(root_node)
    for name, group in trees_df_filtered.groupby('sent_name'):
        row_ids = trees_df_filtered.index[trees_df_filtered.sent_name == name].to_list()
        # temporary dictionary for remapping indices
        temp_dict = {key: row_ids[ind] for ind, key in enumerate(group.id.to_list())}
        temp_dict[0] = 0
        for _, row in group.iterrows():
            new_id = temp_dict.get(row['id'])
            new_node = Node(new_id, dict_lemmas.get(row['lemma']), row['form'], row['sent_name'])
            whole_tree.add_node(new_node)
            new_edge = Edge(temp_dict.get(row['head']), new_id, dict_rel.get(row['deprel']))
            whole_tree.add_edge(new_edge)
    return whole_tree


def compute_full_subtrees(whole_tree, count, grouped_heights):
    # compute subtree repeats
    reps = 1
    r_classes = {}
    v_id_children = {}
    for curr_height, nodes in grouped_heights:
        print(curr_height)
        start = time.time()
        # construct a string of numbers for each node v and its children
        s = {}  # {node id : str(lemma id + ids of subtrees)}
        for v in nodes:
            children_v_ids = whole_tree.get_children(v.id)
            if len(children_v_ids) > 0:
                children_lemmas = tuple([whole_tree.get_node(child_id).lemma for child_id in children_v_ids])
                v_id_children[v.id] = children_lemmas
                s[v.id] = [str(v.lemma)]
                for child_id in children_v_ids:
                    edge_to_curr_child = whole_tree.get_edge(child_id)
                    child_node = whole_tree.get_node(child_id)
                    s[v.id].append(str(edge_to_curr_child.weight) + str(child_node.lemma))
        if curr_height != 0:
            # remap numbers from [1, |alphabet| + |T|) to [1, H[i] + #of children for each node]
            # needed for radix sort - to keep strings shorter
            remapped_nodes = remap_s(s)  # key: v.id, value: int array of remapped value
            # sort inside each string
            sorted_remapped_nodes = sort_strings_inside(remapped_nodes)  # {v.id: str value of mapped string}
            # lexicographically sort the mapped strings with radix sort
            sorted_strings = radix_sort(sorted_remapped_nodes)  # {v.id: str value of mapped string}
            # assign classes
            sorted_vertices_ids = list(sorted_strings.keys())
            prev_vertex = sorted_vertices_ids[0]
            whole_tree.get_node(prev_vertex).lemma = reps + count
            r_classes[reps] = [prev_vertex]
            for ind in range(1, len(sorted_vertices_ids)):
                vertex = sorted_vertices_ids[ind]
                if sorted_strings[vertex] == sorted_strings[prev_vertex] and v_id_children[vertex] == v_id_children[prev_vertex]:
                    r_classes[reps].append(vertex)
                else:
                    reps += 1
                    r_classes[reps] = [vertex]
                whole_tree.get_node(vertex).lemma = reps + count
                prev_vertex = vertex
            reps += 1
        print(time.time() - start)
    r_classes_repeated = {k: v for k, v in r_classes.items() if len(v) > 1}
    return r_classes_repeated


def main():
    trees_full_df, trees_df_filtered, long_df = read_data()
    # trees_df_filtered = trees_df_filtered.head(40)

    # get all lemmas and create a dictionary to map to numbers
    dict_lemmas = {lemma: index for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    # get all relations and create a dictionary to map to numbers
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    dict_rel_rev = {v: k for k, v in dict_rel.items()}

    whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel)
    # partition nodes by height
    whole_tree.calculate_heights()

    heights_dictionary = {whole_tree.get_node(node_id): heights for node_id, heights in
                          whole_tree.heights.items()}
    grouped_heights = defaultdict(list)
    for key, value in heights_dictionary.items():
        grouped_heights[value].append(key)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # classes for full repeats
    classes_full = compute_full_subtrees(whole_tree, len(dict_lemmas.keys()), grouped_heights)
    write_in_file(classes_full, whole_tree, dict_rel_rev)
    merge_in_file()
    uuu = []
    # dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes)))
    # whole_tree.get_node(32).lemma = 20
    # classes_part = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)

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

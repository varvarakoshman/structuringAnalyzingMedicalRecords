from Tree import Tree, Edge
from Util import sort_strings_inside, radix_sort, remap_s


def compute_full_subtrees(whole_tree, count, grouped_heights):
    # compute subtree repeats
    reps = 0
    r_classes = [[] for _ in range(len(whole_tree.nodes))]
    k = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of subtrees
    k_2 = ["" for _ in range(len(whole_tree.nodes))]  # identifiers of edges of subtrees
    for nodes in grouped_heights:
        # construct a string of numbers for each node v and its children
        s = {}  # key: node id, value: str(lemma id + ids of subtrees)
        w = {key: "" for key in list(
            map(lambda x: x.id, nodes[1]))}  # key: node_id, value: str(weights of edges from current node to children)
        for v in nodes[1]:
            children_v = Tree.get_children(whole_tree, v.id)
            s[v.id] = str(v.lemma)
            if len(children_v) > 0:
                w[v.id] += str(k_2[v.id])
                for child_id in children_v:
                    s[v.id] += str(k[child_id])
            edge_to_curr = Tree.get_edge(whole_tree, v.id)
            if edge_to_curr is not None:
                k_2[edge_to_curr.node_from] += str(edge_to_curr.weight) + str(v.lemma)
        # remap numbers from [1, |alphabet| + |T|) to [1, H[i] + #of children for each node]
        # needed for radix sort - to keep strings shorter
        remapped_nodes = remap_s(s)  # key: v.id, value: int array of remapped value
        remapped_edges = remap_s(w)
        # sort inside each string
        sorted_remapped_nodes = sort_strings_inside(remapped_nodes)  # {v.id: str value of mapped string}
        sorted_remapped_edges = sort_strings_inside(remapped_edges)  # {v.id: str value of mapped string}
        # upper_map_bound_n = len(nodes[1]) + n_children
        # upper_map_bound_e = n_children
        # lexicographically sort the mapped strings with radix sort
        sorted_strings = radix_sort(sorted_remapped_nodes)  # {v.id: str value of mapped string}
        sorted_edges = radix_sort(sorted_remapped_edges)  # {v.id: str value of mapped string}
        reps += 1
        # assign classes
        sorted_vertices_id = list(sorted_strings.keys())
        prev_vertex = sorted_vertices_id[0]
        k[prev_vertex] = reps + count
        r_classes[reps].append(prev_vertex)
        for ind in range(1, len(sorted_vertices_id)):
            vertex = sorted_vertices_id[ind]
            if sorted_strings[vertex] == sorted_strings[prev_vertex] and len(sorted_edges) > 0 and sorted_edges[
                vertex] == \
                    sorted_edges[prev_vertex]:
                r_classes[reps].append(vertex)
            else:
                reps += 1
                r_classes[reps] = [vertex]
            k[vertex] = reps + count
            prev_vertex = vertex
    return r_classes
import csv
import json
import os
import re
import time
from collections import defaultdict
from itertools import combinations, product
from itertools import islice
import itertools
import operator
import pprint

import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as download_api
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from Tree import Tree, Node, Edge
from Util import new_test

EMPTY_STR = ''
pattern = re.compile('^#.+$')
REPLACED = 'REPLACED'


def read_data():
    DATA_PATH = r'medicalTextTrees/parus_results'
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    # trees_df = pd.DataFrame(columns=df_columns)
    full_df = []
    stable_df = []
    many_roots_df = []
    very_long_df = []
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        name = file.split('.')[0]
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            if this_df['id'].duplicated().any():
                start_of_subtree_df = list(this_df.groupby(this_df.id).get_group(1).index)
                boundaries = start_of_subtree_df + [max(list(this_df.index)) + 1]
                list_of_dfs = [this_df.iloc[boundaries[n]:boundaries[n + 1]] for n in range(len(boundaries) - 1)]
                local_counter = 1
                for df in list_of_dfs:
                    df['sent_name'] = name + '_' + str(local_counter)
                    full_df.append(df)
                    stable_df.append(df)
                    # trees_df = pd.concat([trees_df, df], ignore_index=True)
                    local_counter += 1
            else:
                this_df['sent_name'] = name
                if this_df.groupby(this_df.deprel).get_group('ROOT').shape[0] > 1:
                    many_roots_df.append(this_df)
                elif this_df.shape[0] > 21:
                    very_long_df.append(this_df)
                else:
                    stable_df.append(this_df)
                # trees_df = pd.concat([trees_df, this_df], ignore_index=True)
                full_df.append(this_df)
    trees_df = pd.concat(stable_df, axis=0, ignore_index=True)
    # delete useless data
    # trees_df = trees_df.drop(columns=['upostag', 'xpostag', 'feats'], axis=1)
    trees_df = trees_df.drop(columns=['xpostag', 'feats'], axis=1)
    # trees_df.drop(index=[11067], inplace=True)
    trees_df.loc[13742, 'deprel'] = 'разъяснит'

    # delete relations of type PUNC and reindex
    trees_df_filtered = trees_df[trees_df.deprel != 'PUNC']
    # trees_df_filtered = trees_df_filtered.loc[trees_df_filtered['sent_name'] != '37918_12']
    # trees_df_filtered = trees_df_filtered.loc[trees_df_filtered['sent_name'] != '38897_9'] # 1) very long sentence 2) 5 roots
    trees_df_filtered = trees_df_filtered.reset_index(drop=True)
    trees_df_filtered.index = trees_df_filtered.index + 1
    # trees_df_filtered.loc[12239, 'deprel'] = '1-компл'
    # trees_df_filtered.loc[12239, 'head'] = 2

    trees_long_df = pd.concat(very_long_df, axis=0, ignore_index=True)
    trees_roots_df = pd.concat(many_roots_df, axis=0, ignore_index=True)
    trees_long_df = trees_long_df[trees_long_df.deprel != 'PUNC']
    trees_roots_df = trees_roots_df[trees_roots_df.deprel != 'PUNC']
    trees_long_df = trees_long_df.reset_index(drop=True)
    trees_long_df.index = trees_long_df.index + 1
    trees_roots_df = trees_roots_df.reset_index(drop=True)
    trees_roots_df.index = trees_roots_df.index + 1

    trees_full_df = pd.concat(full_df, axis=0, ignore_index=True)
    trees_full_df = trees_full_df.reset_index(drop=True)
    trees_full_df.index = trees_full_df.index + 1
    trees_full_df.drop(columns=['upostag', 'xpostag', 'feats'], axis=1)
    trees_full_df = trees_full_df[trees_full_df.deprel != 'PUNC']

    replaced_numbers = [k for k, v in trees_full_df.lemma.str.contains('#').to_dict().items() if
                        v == True]  # едленно, вставить выше
    for num in replaced_numbers:
        trees_df_filtered.loc[num, 'upostag'] = 'Num'
        trees_full_df.loc[num, 'upostag'] = 'Num'

    target_sents = list({'55338_41', '58401_7', '32384_8', '31736_14', '48714_8', '54996_6'}) # TEST
    target_sents = list({'55338_41', '58401_7'})  # TEST
    trees_df_filtered = trees_df_filtered.loc[trees_df_filtered.sent_name.isin(target_sents)] # TEST

    # trees_full_df.loc[trees_full_df.index.isin(replaced_numbers)].assign(upostag = 'N')
    return trees_full_df, trees_df_filtered


def train_word2vec(trees_df_filtered, lemmas, dict_lemmas_3, part_of_speech_node_id):
    lemma_sent_df = trees_df_filtered[['lemma', 'sent_name']]
    lemma_sent_dict = {}
    for name, group in lemma_sent_df.groupby('sent_name'):
        lemma_sent_dict[name] = []
        for _, row in group.iterrows():
            lemma_sent_dict[name].append(row['lemma'])
    sentences = list(lemma_sent_dict.values())

    only_medical = Word2Vec(sentences, min_count=1)
    # X = only_medical[only_medical.wv.vocab]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # create a scatter plot of the projection
    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = list(only_medical.wv.vocab)
    # for i, word in enumerate(words):
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

    model_2 = Word2Vec(min_count=1)
    model_2.build_vocab(sentences)
    w2v_fpath = "additionalCorpus/all_norm-sz100-w10-cb0-it1-min100.w2v"

    # model.intersect_word2vec_format
    # similar_dict = {}
    # for lemma in lemmas:
    #     similar_dict[lemma] = model.most_similar(lemma)
    # embeddings = [model[i] for i in lemmas]

    # additional_model =
    # w2v = Word2Vec.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
    upper_bound = 50000
    high_cosine_dist = 0.8
    model = KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
    model_2.build_vocab([list(model.vocab.keys())[:upper_bound]], update=True)
    model_2.intersect_word2vec_format(w2v_fpath, binary=True, lockf=1.0, unicode_errors='ignore')
    model_2.train(sentences, total_examples=upper_bound, epochs=model_2.iter)
    similar_dict = {lemma: model_2.most_similar(lemma, topn=15) for lemma in lemmas if not pattern.match(lemma)}
    similar_lemmas_dict = {}
    for lemma, similar_lemmas in similar_dict.items():
        for similar_lemma, cosine_dist in similar_lemmas:
            if cosine_dist > high_cosine_dist and similar_lemma in lemmas.keys() and part_of_speech_node_id[similar_lemma] == part_of_speech_node_id[lemma]:
                if lemma not in similar_lemmas_dict.keys():
                    similar_lemmas_dict[lemma] = [similar_lemma]
                else:
                    similar_lemmas_dict[lemma].append(similar_lemma)
    all_values = [item for sublist in similar_lemmas_dict.values() for item in sublist]
    most_freq = set([i for i in all_values if all_values.count(i) > 11])
    similar_lemmas_dict_filtered = {}
    for k, v in similar_lemmas_dict.items():
        stable = set(v) - most_freq
        if 0 < len(stable) <= 10:
            similar_lemmas_dict_filtered[k] = stable

    pprint.pprint(similar_lemmas_dict_filtered)
    # similar_mapping = {}
    for lemma, similar_lemmas in similar_lemmas_dict_filtered.items():
        for similar_lemma in similar_lemmas:
            lemmas[lemma].append(lemmas[similar_lemma][0])
            # if lemmas[similar_lemma][0] not in similar_mapping.keys():
            #     similar_mapping[lemmas[similar_lemma][0]] = [lemma]
            # else:
            #     similar_mapping[lemmas[similar_lemma][0]].append(lemma)
    pprint.pprint(lemmas)

    for lemma, similar_lemmas in similar_lemmas_dict_filtered.items():
        for similar_lemma in similar_lemmas:
            if lemma in dict_lemmas_3.keys():
                dict_lemmas_3[lemma].append(lemmas[similar_lemma][0])

    # dict(sorted(similar_lemmas_dict_filtered.items(), key=lambda item: len(item[1]), reverse=True))
    # words_to_cluster = set([item[0] for sublist in similar_lemmas_dict.values() for item in sublist])
    # embeddings_to_cluster = [model_2[word] for word in words_to_cluster]
    # transformed_embeddings = PCA(n_components=2).fit_transform(embeddings_to_cluster)
    # pyplot.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1])
    # for i, similar_lemma in enumerate(words_to_cluster):
    #     pyplot.annotate(similar_lemma, xy=(transformed_embeddings[i, 0], transformed_embeddings[i, 1]))
    # pyplot.show()
    # DBSCAN(metric=cosine_distances).fit(
    #     [model_2[word] for word in set([item[0] for sublist in similar_lemmas_dict.values() for item in sublist])])
    # DBSCAN(metric=cosine_distances).fit(PCA(n_components=2).fit_transform(
    #     [[model_2[word]] for word in set([item[0] for sublist in similar_lemmas_dict.values() for item in sublist])]))
    # X = model_2[only_medical.wv.vocab]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # create a scatter plot of the projection
    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = list(only_medical.wv.vocab)
    # for i, similar_lemma in enumerate(words):
    #     pyplot.annotate(similar_lemma, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()

    # {k: v for k, v in sorted(
    #     #     {i: [item[0] for sublist in similar_lemmas_dict.values() for item in sublist].count(i) for i in
    #     #      [item[0] for sublist in similar_lemmas_dict.values() for item in sublist]}.items(), key=lambda item: item[1],
    #     #     reverse=True)}
    #

    # w2v.init_sims(replace=True)
    # russian_model = download_api.load('word2vec-ruscorpora-300')
    # KeyedVectors.load_word2vec_format
    # russian_model.build_vocab([list(russian_model.vocab.keys())], update=True)
    # # russian_model.build_vocab(list(lemma_sent_dict.values()), update=True)
    # russian_model.train(list(lemma_sent_dict.values()))
    # vocab = list(russian_model.wv.vocab)
    # vocab = list(model.wv.vocab)
    # X = model[vocab]
    # tsne = TSNE(n_components=2)
    # coordinates = tsne.fit_transform(X)
    # df = pd.DataFrame(coordinates, index=vocab, columns=['x', 'y'])
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(df['x'], df['y'])
    # ax.set_xlim(right=200)
    # ax.set_ylim(top=200)
    # for word, pos in df.iterrows():
    #     ax.annotate(word, pos)
    # plt.show()


def create_new_node(whole_tree, new_id, lemma, form, sent, weight, from_id):
    new_node = Node(new_id, lemma, form, sent)
    Tree.add_node(whole_tree, new_node)
    new_edge = Edge(from_id, new_id, weight)
    Tree.add_edge(whole_tree, new_edge)


def construct_tree(trees_df_filtered, dict_lemmas, dict_rel, dict_lemmas_rev):
    # construct a tree with a list of edges and a list of nodes
    whole_tree = Tree()
    root_node = Node(0, 0)  # add root
    Tree.add_node(whole_tree, root_node)
    # new_id_count = len(trees_df_filtered) + 1
    new_id_count = 1
    similar_lemmas_dict = {}
    global_similar_mapping = {}
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
            sent = row['sent_name']
            form = row['form']
            # create main node
            create_new_node(whole_tree, main_id, curr_lemmas[0], form, sent, weight, from_id)
            children = [main_id]
            edge_to_weight[main_id] = weight
            global_similar_mapping[main_id] = main_id
            # if lemma has additional values add additional nodes
            if len(curr_lemmas) > 1:
                for i in range(1, len(curr_lemmas)):
                    new_id_count += 1
                    while new_id_count in list(map(lambda x: x.id, whole_tree.nodes)):
                        new_id_count += 1
                    create_new_node(whole_tree, new_id_count, curr_lemmas[i], form, sent, weight, from_id)
                    edge_to_weight[new_id_count] = weight
                    if main_id not in similar_lemmas_dict.keys():
                        similar_lemmas_dict[main_id] = [new_id_count]
                    else:
                        similar_lemmas_dict[main_id].append(new_id_count)
                    global_similar_mapping[new_id_count] = main_id
                    children.append(new_id_count)
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
                        Tree.add_edge(whole_tree, Edge(similar_id, child_id, edge_to_weight[child_id]))
        #
        # for main_id, similar_ids in similar_lemmas_dict.items():
        #     if main_id in children_dict.keys():
        #         for child in children_dict[main_id]:
        #             for similar_id in similar_ids:
        #                 Tree.add_edge(whole_tree, Edge(similar_id, child, edge_to_weight[main_id]))
    whole_tree.additional_nodes = set([sublist for list in similar_lemmas_dict.values() for sublist in list])
    whole_tree.similar_lemmas = similar_lemmas_dict
    whole_tree.global_similar_mapping = global_similar_mapping
    return whole_tree


def add_children_to_parents(k_2, filtered_groups, whole_tree, curr_height, old_node_new_nodes):
    all_parents = set()
    for k, v in filtered_groups.items():
        for v_id in list(v):
            edge_to_curr = Tree.get_edge(whole_tree, v_id)
            Tree.get_node(whole_tree, v_id).is_included = True
            if edge_to_curr is not None:
                parent = edge_to_curr[0].node_from
                try:
                    if parent not in whole_tree.heights.keys(): # same as in whole_tree.additional
                        parent = whole_tree.global_similar_mapping[parent]
                    if max(whole_tree.heights[parent]) > curr_height:
                        all_parents.add(parent)
                except KeyError as ke:
                    dfkdf = []
                if v_id in old_node_new_nodes.keys():
                    lemmas_to_visit = old_node_new_nodes[v_id]
                else:
                    lemmas_to_visit = [k]
                if v_id in whole_tree.similar_lemmas.keys():
                    for node_id in whole_tree.similar_lemmas[v_id]:
                        lemmas_to_visit.append(Tree.get_node(whole_tree, node_id).lemma)
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
        for child_id in Tree.get_children(whole_tree, parent):# - whole_tree.additional_nodes:
            edge_to_curr = Tree.get_edge(whole_tree, child_id)[0]
            child_node = Tree.get_node(whole_tree, child_id)
            if not child_node.is_included:
                label_for_child = str(edge_to_curr.weight) + str(child_node.lemma)
                k_2[parent].add(label_for_child)
                child_node.is_included = True
                if parent not in additional_child_nodes.keys():
                    additional_child_nodes[label_for_child] = child_id
                else:
                    additional_child_nodes[label_for_child].append(child_id)
    return additional_child_nodes


# update a label of a subtree with edge weight + a lemma of the last node
def insert_children_labels_to_parents(k_2, grouped_lemmas, whole_tree):
    for k, v in grouped_lemmas.items():
        for v_id in list(v):
            edge_to_curr = Tree.get_edge(whole_tree, v_id)
            if edge_to_curr is not None:
                label_for_child = str(edge_to_curr.weight) + str(k)
                if edge_to_curr.node_from not in k_2.keys():
                    k_2[edge_to_curr.node_from] = [label_for_child]
                else:
                    k_2[edge_to_curr.node_from].append(label_for_child)


def insert_nodeid_label_dict(lemma_nodeid_dict, whole_tree, v_id, curr_lemma):
    edge_to_curr = Tree.get_edge(whole_tree, v_id)
    if edge_to_curr is not None:
        label_for_child = str(edge_to_curr.weight) + str(curr_lemma)
        tup_id_sent = tuple([v_id, whole_tree.get_node(v_id).sent_name])  # ????????????
        if label_for_child not in lemma_nodeid_dict.keys():
            lemma_nodeid_dict[label_for_child] = [tup_id_sent]
        else:
            lemma_nodeid_dict[label_for_child].append(tup_id_sent)


def produce_combinations(k_2, v_id, str_sequence_help, str_sequence_help_reversed, equal_nodes, equal_nodes_mapping):
    if len(equal_nodes) > 0:
        list_for_combinations = []
        prepared_k_2 = set()
        included_labels = []
        for child_tree in k_2[v_id]:
            if child_tree in [item for sublist in list(equal_nodes.values()) for item in
                              sublist] or child_tree in equal_nodes.keys():
                if child_tree in equal_nodes_mapping.keys():
                    actual_label = equal_nodes_mapping[child_tree]
                else:
                    actual_label = child_tree
                if actual_label not in included_labels:
                    if actual_label in equal_nodes.keys():
                        list_for_combinations.append(equal_nodes[actual_label])
                    else:
                        list_for_combinations.append(equal_nodes[child_tree])
                    included_labels.append(actual_label)
            else:
                prepared_k_2.add(child_tree)
        combinations_repeated = list(product(*(list_for_combinations)))
        all_combinations = []
        if len(prepared_k_2) > 0:
            for l in combinations_repeated:
                merged = list(l) + list(prepared_k_2)
                all_combinations.extend(list(combinations(merged, i)) for i in range(1, len(merged) + 1))
        else:
            for l in combinations_repeated:
                all_combinations.extend(list(combinations(list(l), i)) for i in range(1, len(list(l)) + 1))
    else:
        list_for_combinations = k_2[v_id]
        all_combinations = [list(combinations(list_for_combinations, i)) for i in
                            range(1, len(list_for_combinations) + 1)]
    all_combinations_str_joined = set()
    for comb in all_combinations:
        for tup in comb:
            combs = [str(item) for item in sorted(list(tup))]
            joined_label = EMPTY_STR.join(combs)
            if joined_label not in str_sequence_help.keys():
                str_sequence_help[joined_label] = combs
                str_sequence_help_reversed[tuple(combs)] = joined_label
            all_combinations_str_joined.add(joined_label)
    return all_combinations_str_joined


def get_nodeid_repeats(filtered_combination_ids, str_sequence_help):
    dict_nodeid_comb = {}
    for k, v in filtered_combination_ids.items():
        for v_i in v:
            if v_i in dict_nodeid_comb.keys():
                dict_nodeid_comb[v_i].append(str_sequence_help.get(k))
            else:
                dict_nodeid_comb[v_i] = [str_sequence_help.get(k)]
    return dict_nodeid_comb


def get_unique_subtrees_mapped(dict_nodeid_comb, lemma_count, unique_subtrees_mapped_global):
    unique_subtrees = set(tuple(x) for x in [sublist for list in list(dict_nodeid_comb.values()) for sublist in list])
    unique_subtrees_mapped = {}
    existing_combinations = unique_subtrees_mapped_global.keys()
    for subtree in unique_subtrees:
        # if subtree not in existing_combinations:
        if subtree not in existing_combinations:
            lemma_count += 1
            unique_subtrees_mapped[subtree] = lemma_count
    return unique_subtrees_mapped, lemma_count


def compute_part_new_new(whole_tree, lemma_count, grouped_heights):
    classes_subtreeid_nodes = {}
    classes_subtreeid_nodes_list = {}
    unique_subtrees_mapped_global_subtree_lemma = {}
    old_node_new_nodes = {}
    equal_nodes_mapping = {}
    subtree_label_sent = {}
    k_2 = {}  # identifiers of edges of subtrees
    lemma_nodeid_dict = {}
    saved_combinations = []
    for nodes in grouped_heights:
        curr_height = nodes[0]
        print(curr_height)
        start = time.time()
        id_lemma_dict = {node.id: node.lemma for node in nodes[1]}
        grouped_lemmas = defaultdict(list)
        for key, value in id_lemma_dict.items():
            grouped_lemmas[value].append(key)
        all_parents = add_children_to_parents(k_2, grouped_lemmas, whole_tree, curr_height, old_node_new_nodes)
        additional_child_nodes = add_additional_children_to_parents(k_2, whole_tree, all_parents)
        for additional_child, child_id in additional_child_nodes.items():
            if additional_child not in lemma_nodeid_dict.keys():
                lemma_nodeid_dict[additional_child] = {child_id}
            else:
                lemma_nodeid_dict[additional_child].add(child_id)
        for lemma, ids in grouped_lemmas.items():
            for v_id in ids:
                edge_to_curr = Tree.get_edge(whole_tree, v_id)
                if edge_to_curr is not None:
                    label_for_child = str(edge_to_curr[0].weight) + str(lemma)
                    if label_for_child not in lemma_nodeid_dict.keys():
                        lemma_nodeid_dict[label_for_child] = {v_id}
                    else:
                        lemma_nodeid_dict[label_for_child].add(v_id)

        if curr_height != 0:  # not applicable to leaves, leaves don't have subtrees
            filtered_groups = {k: v for k, v in grouped_lemmas.items() if len(v) > 1}
            for lemma, ids in filtered_groups.items():
                combination_ids = {}
                str_sequence_help = {}
                str_sequence_help_reversed = {}

                # generate combinations
                for v_id in ids:
                    equal_nodes = {}
                    # only for duplicating nodes
                    children = Tree.get_children(whole_tree, v_id) - whole_tree.created - whole_tree.additional_nodes
                    for child in children:
                        if child in old_node_new_nodes.keys() or child in whole_tree.similar_lemmas.keys():
                            edge_to_child = Tree.get_edge(whole_tree, child)[0]
                            child_node = Tree.get_node(whole_tree, child)
                            w = str(edge_to_child.weight)
                            actual_label = w + str(child_node.lemma)
                            merge = []
                            if child in old_node_new_nodes.keys():
                                for l in old_node_new_nodes[child]: # refactor this and below
                                    new_label = w + str(l)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                            else:
                                merge.append(actual_label)
                                for node_id in whole_tree.similar_lemmas[child]:
                                    new_label = w + str(Tree.get_node(whole_tree, node_id).lemma)
                                    merge.append(new_label)
                                    equal_nodes_mapping[new_label] = actual_label
                            if actual_label not in equal_nodes.keys():
                                equal_nodes[actual_label] = merge
                            else:
                                equal_nodes[actual_label].extend(merge)
                    all_combinations_str_joined = produce_combinations(k_2, v_id, str_sequence_help,
                                                                       str_sequence_help_reversed, equal_nodes,
                                                                       equal_nodes_mapping)
                    for label in all_combinations_str_joined:
                        if label in combination_ids.keys():
                            combination_ids[label].append(v_id)
                        else:
                            combination_ids[label] = [v_id]

                filtered_combination_ids = {k: v for k, v in combination_ids.items() if len(v) > 1}
                for tree_label, node_list in filtered_combination_ids.items():
                    str_tree_label = str(lemma) + tree_label
                    if str_tree_label not in unique_subtrees_mapped_global_subtree_lemma.keys():
                        unique_subtrees_mapped_global_subtree_lemma[str_tree_label] = lemma_count
                        lemma_count += 1
                # 16: [['107', '919'], ['208', '919'], ['919'], ['107'], ['208']]
                dict_nodeid_comb = get_nodeid_repeats(filtered_combination_ids, str_sequence_help)
                for node_id, node_subtrees in dict_nodeid_comb.items():
                    existing_node = Tree.get_node(whole_tree, node_id)
                    edge_to_curr = Tree.get_edge(whole_tree, node_id)[0]
                    children = Tree.get_children(whole_tree, node_id)
                    for subtree in node_subtrees:
                        subtree_text = str_sequence_help_reversed.get(tuple(subtree))
                        subtree_new_label = unique_subtrees_mapped_global_subtree_lemma.get(str(lemma) + subtree_text)

                        if subtree_new_label not in subtree_label_sent.keys() or existing_node.sent_name not in subtree_label_sent[subtree_new_label]:
                            # add new node with a new lemma
                            new_node = Tree.copy_node_details(whole_tree, existing_node)
                            new_node.lemma = subtree_new_label
                            Tree.add_node_to_dict(whole_tree, new_node)

                            if subtree_new_label not in subtree_label_sent.keys():
                                subtree_label_sent[subtree_new_label] = [existing_node.sent_name]
                            else:
                                subtree_label_sent[subtree_new_label].append(existing_node.sent_name)

                            # add new node to node aliases
                            if node_id not in old_node_new_nodes.keys():
                                old_node_new_nodes[node_id] = [new_node.lemma]
                            else:
                                old_node_new_nodes[node_id].append(new_node.lemma)

                            # add an edge to it
                            edge = Edge(edge_to_curr.node_from, new_node.id, edge_to_curr.weight)
                            Tree.add_edge_to_dict(whole_tree, edge)

                            # add new node to parent
                            parent_subtree_text = str(edge_to_curr.weight) + str(subtree_new_label)
                            if parent_subtree_text not in lemma_nodeid_dict.keys():
                                lemma_nodeid_dict[parent_subtree_text] = {new_node.id}
                            else:
                                lemma_nodeid_dict[parent_subtree_text].add(new_node.id)

                            subtree_children = []
                            children_nodes = {}
                            for subtree_node in subtree:
                                if subtree_node not in lemma_nodeid_dict.keys():
                                    if subtree_node in equal_nodes_mapping.keys():
                                        target_child = \
                                        list(set(lemma_nodeid_dict[equal_nodes_mapping[subtree_node]]) & children)[0]
                                    else:
                                        if len(children_nodes) == 0:
                                            children_nodes = Tree.get_children_nodes(whole_tree, children)
                                        target_child = children_nodes[subtree_node]
                                else:
                                    intersection = set(lemma_nodeid_dict[subtree_node]) & children
                                    if len(intersection) == 0:
                                        if len(children_nodes) == 0:
                                            children_nodes = Tree.get_children_nodes(whole_tree, children)
                                        try:
                                            target_child = children_nodes[subtree_node]
                                        except KeyError as ke:
                                            gh = []
                                    else:
                                        target_child = list(intersection)[0]
                                subtree_children.append(target_child)

                            if len(subtree_children) > 0:
                                general_comb = ''.join(sorted([str(whole_tree.global_similar_mapping[child_id]) for child_id in subtree_children]))
                                if general_comb not in saved_combinations:
                                    # add edges to subtree's children from new node
                                    Tree.add_new_edges(whole_tree, new_node.id, subtree_children)

                                    # assign class
                                    if subtree_new_label not in classes_subtreeid_nodes.keys():
                                        classes_subtreeid_nodes[subtree_new_label] = [new_node.id]
                                    else:
                                        classes_subtreeid_nodes[subtree_new_label].append(new_node.id)

                                    subtree_deep_children = set()
                                    for subtree_lemma in list(
                                            map(lambda x: Tree.get_node(whole_tree, x).lemma, subtree_children)):
                                        if subtree_lemma in classes_subtreeid_nodes_list.keys():
                                            subtree_deep_children.update(classes_subtreeid_nodes_list[subtree_lemma])
                                    subtree_deep_children.update(subtree_children)
                                    only_active = subtree_deep_children - whole_tree.inactive
                                    # only_active = (whole_tree.inactive.symmetric_difference(set_children))&set_children

                                    if subtree_new_label not in classes_subtreeid_nodes_list.keys():
                                        classes_subtreeid_nodes_list[subtree_new_label] = only_active
                                    else:
                                        classes_subtreeid_nodes_list[subtree_new_label].update(only_active)
                                    classes_subtreeid_nodes_list[subtree_new_label].add(new_node.id)

                                    saved_combinations.append(general_comb)

                    # remove old node and edges to/from it
                    Tree.add_inactive(whole_tree, node_id)
        print(time.time() - start)
    classes_subtreeid_nodes = {k: v for k, v in classes_subtreeid_nodes.items() if len(v) > 1} # TODO: why do len=1 entries even appear here??
    return classes_subtreeid_nodes, classes_subtreeid_nodes_list


def write_tree_in_table(whole_tree):
    source_1 = 'medicalTextTrees/gephi_edges_import_word2vec.csv'
    source_2 = 'medicalTextTrees/gephi_nodes_import_word2vec.csv'
    with open(source_1, "w", newline='', encoding='utf-8') as csv_file_1, open(
            source_2, "w", newline='', encoding='utf-8') as csv_file_2:
        writer_1 = csv.writer(csv_file_1, delimiter=',')
        writer_2 = csv.writer(csv_file_2, delimiter=',')
        writer_1.writerow(['Source', 'Target', 'Weight'])
        writer_2.writerow(['Id', 'Label'])
        for edge in whole_tree.edges:
            writer_1.writerow([edge.node_from, edge.node_to, edge.weight + 1])
        for node in whole_tree.nodes:
            writer_2.writerow([node.id, (node.lemma, node.form, node.sent_name)])


def main():
    start = time.time()
    trees_full_df, trees_df_filtered = read_data()
    test_3_sent = trees_df_filtered.head(12)
    print('Time on reading the data: ' + str(time.time() - start))
    # TEST - тест на первых 3х предложениях
    # trees_df_filtered = trees_df_filtered.head(7285)  # 341 - all? 48 - 3 # 3884 # 5015
    # trees_df_filtered = trees_df_filtered[trees_df_filtered.sent_name == '48554_5']
    #
    part_of_speech_node_id = dict(trees_full_df[['lemma', 'upostag']].groupby(['lemma', 'upostag']).groups.keys())

    # get all lemmas and create a dictionary to map to numbers
    dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    dict_lemmas_full = {lemma: [index] for index, lemma in
                        enumerate(dict.fromkeys(trees_full_df['lemma'].to_list()), 1)}
    dict_lemmas_rev = {index[0]: lemma for lemma, index in dict_lemmas_full.items()}
    dict_lemmas_3 = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(test_3_sent['lemma'].to_list()), 1)}
    #
    numbers = [item for item in list(dict_lemmas_full.keys()) if pattern.match(item)]
    numbers_one_lemma = dict_lemmas_full[numbers[0]]
    for num in numbers:
        dict_lemmas[num] = numbers_one_lemma
        dict_lemmas_full[num] = numbers_one_lemma
        if num in dict_lemmas_3.keys():
            dict_lemmas_3[num] = numbers_one_lemma
    # get all relations and create a dictionary to map to numbers
    # dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_full_df['deprel'].to_list()))}
    dict_rel = {rel: index for index, rel in enumerate(dict.fromkeys(trees_df_filtered['deprel'].to_list()))}
    train_word2vec(trees_full_df, dict_lemmas_full, dict_lemmas, part_of_speech_node_id)

    #
    start = time.time()
    # test_dict_lemmas = get_test_dict_lemmas()
    # new_test_dict = {k: test_dict_lemmas[i] for i, k in enumerate(dict_lemmas_3.keys())}
    whole_tree = construct_tree(trees_df_filtered, dict_lemmas, dict_rel, dict_lemmas_rev)
    # write_tree_in_table(whole_tree)

    print('Time on constructing the tree: ' + str(time.time() - start))
    # whole_tree = new_test()
    Tree.set_help_dict(whole_tree)
    # partition nodes by height
    start = time.time()
    Tree.calculate_heights(whole_tree)
    print('Time on calculating all heights: ' + str(time.time() - start))

    heights_dictionary = {Tree.get_node(whole_tree, node_id): heights for node_id, heights in
                          whole_tree.heights.items()}
    grouped_heights = defaultdict(list)
    for node_1, heights in heights_dictionary.items():
        for height in heights:
            grouped_heights[height].append(node_1)
    grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # heights_dictionary = {Tree.get_node(whole_tree, node_id): height for node_id, height in
    #                       enumerate(whole_tree.heights)}
    # grouped_heights = defaultdict(list)
    # for key, value in heights_dictionary.items():
    #     grouped_heights[value].append(key)
    # grouped_heights = sorted(grouped_heights.items(), key=lambda x: x[0])

    # classes for full repeats
    # start = time.time()
    # classes_full = compute_full_subtrees(whole_tree, len(dict_lemmas.keys()), grouped_heights)
    # print('Time on calculating full repeats: ' + str(time.time() - start))
    # for index, listt in enumerate(classes_full):
    #     vertex_seq = {}
    #     for vertex in listt:
    #         vertex_seq[vertex] = Tree.simple_dfs(whole_tree, vertex)
    #     # if len(vertex_seq.items()) > 0:
    #     filename = 'results_full/results_%s.txt' % (str(index))
    #     with open(filename, 'w') as filehandle:
    #         for key, value in vertex_seq.items():
    #             filehandle.write("%s: %s\n" % (key, value))

    dict_lemmas_size = max(set(map(lambda x: x.lemma, whole_tree.nodes)))

    # classes for partial repeats
    start = time.time()
    classes_part, classes_part_list = compute_part_new_new(whole_tree, dict_lemmas_size, grouped_heights)
    write_tree_in_table(whole_tree)
    # classes_part = compute_part_subtrees(whole_tree, dict_lemmas_size, grouped_heights)
    print('Time on calculating partial repeats: ' + str(time.time() - start))
    for k, v in classes_part.items():
        vertex_seq = {}
        for vertex in v:
            vertex_seq[vertex] = Tree.simple_dfs(whole_tree, vertex, classes_part_list[k])
        if len(vertex_seq.items()) > 0:
            filename = 'medicalTextTrees/results_part_new/results_%s.txt' % (str(k))
            try:
                with open(filename, 'w', encoding='utf-8') as filehandle:
                    for key, value in vertex_seq.items():
                        filehandle.write("%s: %s\n" % (key, value))
            finally:
                filehandle.close()


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

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
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
from matplotlib import pyplot
from sklearn.manifold import TSNE

from Preprocessing import replace_time_constructions, read_data, read_vidal
from const.Constants import EMPTY_STR, NEW_LINE, VIS_PATH_N2V, MERGED_PATH, SPACE


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


def visualise_classes():
    trees_full_df, trees_df_filtered = read_data()
    replace_time_constructions(trees_df_filtered)
    replace_time_constructions(trees_full_df)
    # part_of_speech_node_id = dict(trees_full_df[['lemma', 'upostag']].groupby(['lemma', 'upostag']).groups.keys())
    # dict_lemmas = {lemma: [index] for index, lemma in enumerate(dict.fromkeys(trees_df_filtered['lemma'].to_list()), 1)}
    # dict_lemmas_full = {lemma: [index] for index, lemma in
    #                     enumerate(dict.fromkeys(trees_full_df['lemma'].to_list()), 1)}
    form_lemma_dict = dict(zip(trees_df_filtered['form'].to_list(), trees_df_filtered['lemma'].to_list()))
    try:
        with open(VIS_PATH_N2V, encoding='utf-8') as reader:
            class_entries = reader.readlines()
    finally:
        reader.close()
    class_count = 1
    classes_numbered = {}
    all_words = set()
    for line in class_entries:
        if line == NEW_LINE:
            class_count += 1
        else:
            word = line.split(':')[1].split(NEW_LINE)[0]
            # if class_count not in classes_numbered.keys():
            #     try:
            #         classes_numbered[class_count].append(form_lemma_dict[word])
            #     except KeyError as ke:
            #         dfjkd = []
            # else:
            #     classes_numbered[class_count] = [form_lemma_dict[word]]
            words = word.split(" ")
            for w in words:
                if w != EMPTY_STR:
                    all_words.add(form_lemma_dict[w])
    medical_model = Word2Vec.load("trained.model")
    # words_to_cluster = set([item[0] for sublist in similar_lemmas_dict_filtered.values() for item in sublist])
    embeddings_to_cluster = [medical_model[word] for word in all_words]
    transformed_embeddings = TSNE(n_components=2, perplexity=8).fit_transform(embeddings_to_cluster)
    # words = list(only_medical.wv.vocab)
    for i, similar_lemma in enumerate(all_words):
        pyplot.annotate(similar_lemma, xy=(transformed_embeddings[i, 0], transformed_embeddings[i, 1]))
    pyplot.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1])
    pyplot.show()
    # transformed_embeddings = PCA(n_components=2).fit_transform(embeddings_to_cluster)


def draw_histogram():
    # alphab = [5071, 3384, 1361, 650, 347, 142]
    # frequencies = ['2', '3-5', '6-10', '11-20', '21-50', '51-2.6k']
    # pos = np.arange(len(frequencies))
    # ax = plt.axes()
    # ax.set_xticks(pos)
    # ax.set_xticklabels(frequencies)
    # ax.set_xlabel('Number of repeats in a group')
    # ax.set_ylabel('Number of groups')
    # plt.xticks()
    # plt.bar(pos, alphab, width=0.9, color='b')
    # plt.title('Number of groups with equal number of repeats')
    # plt.show()

    # import re
    #
    # MEDIUM_SIZE = 12
    # BIGGER_SIZE = 14
    #
    # # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # # plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # # plt.rc('title', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # path = MERGED_PATH
    # try:
    #     with open(path, encoding='utf-8') as reader:
    #         class_entries = reader.readlines()
    # finally:
    #     reader.close()
    # class_count = 1
    # local_c = 0
    # curr_len = 0
    # group_len = {}
    # str_len = {}
    # for line in class_entries:
    #     if not re.match('label*', line):
    #         if line == NEW_LINE:
    #             group_len[class_count] = local_c
    #             str_len[class_count] = curr_len
    #             local_c = 0
    #             class_count += 1
    #         else:
    #             words = [w for w in line.split(':')[1].split(NEW_LINE)[0].split(" ") if w != EMPTY_STR][1::3]
    #             curr_len = len(words)
    #             local_c += 1
    # res = {}
    # for key, val in sorted(group_len.items()):
    #     if val not in res.keys():
    #         res[val] = 1
    #     else:
    #         res[val] += 1
    #
    # res2 = dict(sorted(res.items(), key=lambda x: x[0]))
    # alphab = list(res2.values())
    # frequencies = list(res2.keys())
    #
    # pos = np.arange(len(frequencies))
    # ax = plt.axes()
    # ax.set_xticks(pos)
    # ax.set_xticklabels(frequencies)
    # ax.set_xlabel('Число повторов в классе', fontsize=12)
    # ax.set_ylabel('Число классов', fontsize=12)
    # plt.xticks()
    # plt.bar(pos, alphab, width=0.9, color='b')
    # plt.title('Число классов с равным числом повторов', fontsize=12)
    # for index, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if index % 10 != 0:
    #         label.set_visible(False)
    # plt.show()

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
        if not re.match('label*', line):
            if line == NEW_LINE:
                group_len[class_count] = local_c
                str_len[class_count] = curr_len
                local_c = 0
                class_count += 1
            else:
                words = [w for w in line.split(':')[1].split(NEW_LINE)[0].split(" ") if w != EMPTY_STR][1::3]
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
    ax.set_xlabel('Число повторов в классе')
    ax.set_ylabel('Число классов')
    plt.xticks()
    plt.bar(pos, alphab, width=0.9, color='b')
    plt.title('Число классов с равным числом повторов')
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 5 != 0:
            label.set_visible(False)
    plt.show()


def plot_labels_per_sentences(sent_labels):
    sent_labels = dict(sorted(sent_labels.items(), key=lambda entry: len(entry[1]), reverse=False))
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    count_dict = {}
    sent_num_labels = {k: len(v) for k, v in sent_labels.items()}
    for sent, num_labels in sent_num_labels.items():
        if num_labels not in count_dict.keys():
            count_dict[num_labels] = 1
        else:
            count_dict[num_labels] += 1
    numb_of_sentences_for_each_num_of_labels = list(count_dict.values())
    frequencies = list(count_dict.keys())
    pos1 = np.arange(len(frequencies))

    # hardcode to make a plot look nicer
    # frequencies = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12-17"] # change values accordingly
    # pos1 = np.arange(len(frequencies))
    # numb_of_sentences_for_each_num_of_labels = [826, 933, 841, 719, 551, 377, 219, 163, 104, 63, 28, 21]

    ax1 = plt.axes()
    ax1.grid(False)
    ax1.set_xticks(pos1)
    ax1.set_xticklabels(frequencies)
    ax1.set_xlabel('Number of labels', fontsize=12)
    ax1.set_ylabel('Number of sentences', fontsize=12)
    plt.title('Number of labels per sentence', fontsize=12)
    plt.bar(pos1, numb_of_sentences_for_each_num_of_labels, width=0.9, color='blue')
    plt.show()


def plot_repeat_len(results_dict):
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    count_dict = {}
    class_id_len = {k: len(v[0].split(SPACE)) for k, v in results_dict.items()}
    for class_id, leng in class_id_len.items():
        if leng not in count_dict.keys():
            count_dict[leng] = 1
        else:
            count_dict[leng] += 1
    alphab = list(count_dict.values())
    frequencies = list(count_dict.keys())
    pos1 = np.arange(len(frequencies))
    ax1 = plt.axes()
    ax1.set_xticks(pos1)
    ax1.set_xticklabels(frequencies)
    ax1.set_xlabel('Number of words in a repeat', fontsize=12)
    ax1.set_ylabel('Number of groups', fontsize=12)
    plt.bar(pos1, alphab, width=0.9, color='b')
    plt.title('Number of groups with equal number of words in a repeat', fontsize=12)
    plt.show()


def plot_top_30_labels(label_classes_sorted):
    label_classes_simple = {k: len(v) for k, v in label_classes_sorted.items()}
    top_30 = list(label_classes_simple.items())[-30:]
    labels = [item[0] for item in top_30]
    counts = [item[1] for item in top_30]
    joint_dict = dict(sorted({labels[i]: counts[i] for i in range(30)}.items(), key=lambda x: x[1], reverse=True))
    joint_df = pd.DataFrame({'label': list(joint_dict.keys()), 'count': list(joint_dict.values())})
    ax = sns.barplot(x='count', y='label',
                     data=joint_df,
                     palette="crest")
    ax.set(xlabel='count', ylabel='label')
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    read_vidal()

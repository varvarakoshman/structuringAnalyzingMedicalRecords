import os

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from Constants import MERGED_PATH, EMPTY_STR, VIS_PATH, NEW_LINE
from Preprocessing import replace_time_constructions, read_data, read_vidal

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
from W2Vprocessing import load_trained_word2vec


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
        with open(VIS_PATH, encoding='utf-8') as reader:
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


if __name__ == '__main__':
    read_vidal()

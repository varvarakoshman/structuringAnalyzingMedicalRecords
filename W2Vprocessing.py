from gensim.models import Word2Vec, KeyedVectors
from matplotlib import pyplot
from sklearn.manifold import TSNE

from Constants import *
import re
import pandas as pd
from stellargraph import StellarDiGraph
from stellargraph.data import BiasedRandomWalk

from Util import write_dict_in_file, label_lemmas

pattern = re.compile('^#.+$')


def train_word2vec(trees_df_filtered):
    lemma_sent_df = trees_df_filtered[['lemma', 'sent_name']]
    lemma_sent_dict = {}
    for name, group in lemma_sent_df.groupby('sent_name'):
        lemma_sent_dict[name] = []
        for _, row in group.iterrows():
            lemma_sent_dict[name].append(row['lemma'])
    sentences = list(lemma_sent_dict.values())

    medical_model = Word2Vec(min_count=1)
    medical_model.build_vocab(sentences)

    additional_model = KeyedVectors.load_word2vec_format(ADDITIONAL_CORPUS_PATH, binary=True, unicode_errors='ignore')
    medical_model.build_vocab([list(additional_model.vocab.keys())[:UPPER_BOUND_ADDITIONAL_DATA]], update=True)
    medical_model.intersect_word2vec_format(ADDITIONAL_CORPUS_PATH, binary=True, lockf=1.0, unicode_errors='ignore')
    medical_model.train(sentences, total_examples=medical_model.corpus_count, epochs=medical_model.iter)
    medical_model.save("trained.model")


def train_node2vec(whole_tree_plain, dict_lemmas_rev):
    walk_length = 10
    # filtered_edges = list(filter(lambda edge: edge.node_from != 0, whole_tree_plain.edges))
    dict_lemmas_rev[0] = 'root'
    sources = list(map(lambda edge: dict_lemmas_rev[whole_tree_plain.get_node(edge.node_from).lemma], whole_tree_plain.edges))
    targets = list(map(lambda edge: dict_lemmas_rev[whole_tree_plain.get_node(edge.node_to).lemma], whole_tree_plain.edges))
    weights = list(map(lambda edge: edge.weight, whole_tree_plain.edges))
    edges = pd.DataFrame({
        "source": sources,
        "target": targets,
        "weight": weights
    })
    stellar_graph = StellarDiGraph(edges=edges)
    random_walk = BiasedRandomWalk(stellar_graph)
    weighted_walks = random_walk.run(
        nodes=stellar_graph.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=5,  # number of random walks per root node
        p=3,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=7,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    print("Number of random walks: {}".format(len(weighted_walks)))
    weighted_model = Word2Vec(min_count=1)
    weighted_model.build_vocab(weighted_walks)
    additional_model = KeyedVectors.load_word2vec_format(ADDITIONAL_CORPUS_PATH, binary=True, unicode_errors='ignore')
    weighted_model.build_vocab([list(additional_model.vocab.keys())[:UPPER_BOUND_ADDITIONAL_DATA]], update=True)
    weighted_model.intersect_word2vec_format(ADDITIONAL_CORPUS_PATH, binary=True, lockf=1.0, unicode_errors='ignore')
    weighted_model.train(weighted_walks, total_examples=weighted_model.corpus_count, epochs=weighted_model.iter)
    weighted_model.save("trained_node2vec.model")


def load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id, model_name): #dict_lemmas_filt,
    medical_model = Word2Vec.load(model_name)
    similar_dict = {lemma: medical_model.most_similar(lemma, topn=15) for lemma in dict_lemmas_full if not pattern.match(lemma)}
    similar_lemmas_dict = {}
    for lemma, similar_lemmas in similar_dict.items():
        for similar_lemma, cosine_dist in similar_lemmas:
            if cosine_dist > HIGH_COSINE_DIST and similar_lemma in dict_lemmas_full.keys() \
                    and part_of_speech_node_id[similar_lemma] == part_of_speech_node_id[lemma]:
                if lemma not in similar_lemmas_dict.keys():
                    similar_lemmas_dict[lemma] = [similar_lemma]
                else:
                    similar_lemmas_dict[lemma].append(similar_lemma)
    all_values = [item for sublist in similar_lemmas_dict.values() for item in sublist]
    most_freq = set([i for i in all_values if all_values.count(i) > 11])
    similar_lemmas_dict_filtered = {}
    for k, v in similar_lemmas_dict.items():
        stable = set(list(dict.fromkeys(v))) - most_freq
        similar_lemmas_dict_filtered[k] = list(stable)[:5]
    similar_lemmas_dict_filtered = dict(sorted({k : v for k, v in similar_lemmas_dict_filtered.items() if len(v) > 0}.items()))
    write_dict_in_file(similar_lemmas_dict_filtered)
    for lemma, similar_lemmas in similar_lemmas_dict_filtered.items():
        for similar_lemma in similar_lemmas:
            dict_lemmas_full[lemma].append(dict_lemmas_full[similar_lemma][0])


def get_embeddings(data_dict, model1, model2):
    n2v_embeddings_to_cluster = [model1[word] for word in data_dict.keys()]
    w2v_embeddings_to_cluster = [model2[word] for word in data_dict.keys()]
    n2v_transformed_embeddings = TSNE(n_components=2, perplexity=8).fit_transform(n2v_embeddings_to_cluster)
    w2v_transformed_embeddings = TSNE(n_components=2, perplexity=8).fit_transform(w2v_embeddings_to_cluster)
    return n2v_transformed_embeddings, w2v_transformed_embeddings


def visualize_embeddings(lemmas_list, n2v_model_name, w2v_model_name):
    n2v_medical_model = Word2Vec.load(n2v_model_name)
    w2v_medical_model = Word2Vec.load(w2v_model_name)
    labeled_lemmas = label_lemmas(lemmas_list.keys())
    diseases = {k: v for k, v in labeled_lemmas.items() if v == 0}
    symptoms = {k: v for k, v in labeled_lemmas.items() if v == 1}
    docs = {k: v for k, v in labeled_lemmas.items() if v == 2}
    drugs = {k: v for k, v in labeled_lemmas.items() if v == 3}
    times = {k: v for k, v in labeled_lemmas.items() if v == 4}
    # diseases
    n2v_embeddings_disease, w2v_embeddings_disease = get_embeddings(diseases, n2v_medical_model, w2v_medical_model)
    # symptoms
    n2v_embeddings_symptoms, w2v_embeddings_symptoms = get_embeddings(symptoms, n2v_medical_model, w2v_medical_model)
    # docs
    n2v_embeddings_docs, w2v_embeddings_docs = get_embeddings(docs, n2v_medical_model, w2v_medical_model)
    # drugs
    n2v_embeddings_drugs, w2v_embeddings_drugs = get_embeddings(drugs, n2v_medical_model, w2v_medical_model)
    # times
    n2v_embeddings_times, w2v_embeddings_times = get_embeddings(times, n2v_medical_model, w2v_medical_model)

    # chunks_1 = chunks(n2v_embeddings_to_cluster, 20)[7]
    # chunks_2 = chunks(w2v_embeddings_to_cluster, 20)[7]
    # chunk_lemmas = chunks(lemmas_list, 20)[7]
    # n2v_transformed_embeddings = TSNE(n_components=2, perplexity=8).fit_transform(chunks_1)
    # w2v_transformed_embeddings = TSNE(n_components=2, perplexity=8).fit_transform(chunks_2)
    # for i, similar_lemma in enumerate(chunk_lemmas):
    # for i, similar_lemma in enumerate(lemmas_list):
    #     pyplot.annotate(similar_lemma, xy=(n2v_transformed_embeddings[i, 0], n2v_transformed_embeddings[i, 1]))
    #     pyplot.annotate(similar_lemma, xy=(w2v_transformed_embeddings[i, 0], w2v_transformed_embeddings[i, 1]))

    n2v_dis = pyplot.scatter(n2v_embeddings_disease[:, 0], n2v_embeddings_disease[:, 1], color='r', marker="*")
    w2v_dis = pyplot.scatter(w2v_embeddings_disease[:, 0], w2v_embeddings_disease[:, 1], color='r', marker="x")
    n2v_sym = pyplot.scatter(n2v_embeddings_symptoms[:, 0], n2v_embeddings_symptoms[:, 1], color='b', marker="*")
    w2v_sym = pyplot.scatter(w2v_embeddings_symptoms[:, 0], w2v_embeddings_symptoms[:, 1], color='b', marker="x")
    n2v_docs = pyplot.scatter(n2v_embeddings_docs[:, 0], n2v_embeddings_docs[:, 1], color='g', marker="*")
    w2v_docs = pyplot.scatter(w2v_embeddings_docs[:, 0], w2v_embeddings_docs[:, 1], color='g', marker="x")
    n2v_drgs = pyplot.scatter(n2v_embeddings_drugs[:, 0], n2v_embeddings_drugs[:, 1], color='y', marker="*")
    w2v_drgs = pyplot.scatter(w2v_embeddings_drugs[:, 0], w2v_embeddings_drugs[:, 1], color='y', marker="x")
    n2v_time = pyplot.scatter(n2v_embeddings_times[:, 0], n2v_embeddings_times[:, 1], color='c', marker="*")
    w2v_time = pyplot.scatter(w2v_embeddings_times[:, 0], w2v_embeddings_times[:, 1], color='c', marker="x")

    pyplot.legend(((n2v_dis, w2v_dis), (n2v_sym, w2v_sym), (n2v_docs, w2v_docs), (n2v_drgs, w2v_drgs), (n2v_time, w2v_time)), ('Болезнь', 'Симптом', 'Врач', 'Лекарство', 'Временная метка'))
    pyplot.show()


def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
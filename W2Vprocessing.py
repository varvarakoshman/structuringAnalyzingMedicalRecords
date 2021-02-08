from gensim.models import Word2Vec, KeyedVectors
from Constants import *
import re

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
    medical_model.train(sentences, total_examples=UPPER_BOUND_ADDITIONAL_DATA, epochs=medical_model.iter)
    medical_model.save("trained.model")


def load_trained_word2vec(dict_lemmas_full, part_of_speech_node_id): #dict_lemmas_filt,
    medical_model = Word2Vec.load("trained.model")
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
        stable = set(v) - most_freq
        if 0 < len(stable) <= 10:
            similar_lemmas_dict_filtered[k] = stable
    jjj = []
    # pprint.pprint(similar_lemmas_dict_filtered)
    for lemma, similar_lemmas in similar_lemmas_dict_filtered.items():
        for similar_lemma in similar_lemmas:
            dict_lemmas_full[lemma].append(dict_lemmas_full[similar_lemma][0])
    # pprint.pprint(lemmas)
    # for lemma in dict_lemmas_filt.keys():

    # for lemma, similar_lemmas in similar_lemmas_dict_filtered.items():
    #     for similar_lemma in similar_lemmas:
    #         if lemma in dict_lemmas_filt.keys():
    #             dict_lemmas_filt[lemma].append(dict_lemmas_full[similar_lemma][0])
    # ttt = []
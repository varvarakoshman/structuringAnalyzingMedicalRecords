# Path constants
DATA_PATH = r"data/final_corpus"
ORIGINAL_DATA_PATH = "original_sentences_separeted"
RESULT_PATH = "results_main"
RESULT_PATH_FILTERED = "results_filtered"
RESULT_PATH_OLD = "results_main_old"
ADDITIONAL_CORPUS_PATH = "additionalCorpus/all_norm-sz100-w10-cb0-it1-min100.w2v"
EXCEL_EDGES_GEPHI_PATH = "medicalTextTrees/gephi_edges_import.csv"
EXCEL_NODES_GEPHI_PATH = "medicalTextTrees/gephi_nodes_import.csv"
MERGED_PATH = "medicalTextTrees/merged_extended.txt"
MERGED_PATH_FILTERED = "medicalTextTrees/merged_extended_old.txt"
MERGED_PATH_OLD = "medicalTextTrees/merged_extended_old.txt"
CLASSIFIED_DATA_PATH = "medicalTextTrees/classified_sentences"
LONG_DATA_PATH = "medicalTextTrees/classified_sentences/long_sentences"
STABLE_DATA_PATH = "medicalTextTrees/classified_sentences/stable_sentences"
MANY_ROOTS_DATA_PATH = "medicalTextTrees/classified_sentences/many_roots_sentences"
VIS_PATH_W2V = "medicalTextTrees/merged_extended_visualize_w2v.txt"
VIS_PATH_N2V = "medicalTextTrees/merged_extended_visualize_n2v.txt"
MEDICAL_DICTIONARY = "medicalTextTrees/support_med_data.xls"
ALL_LEMMAS_PATH_N2V = "medicalTextTrees/all_lemmas_n2v.txt"
ALL_LEMMAS_PATH_W2V = "medicalTextTrees/all_lemmas_w2v.txt"
RAW_DATA_PATH = "data/separeted_sentences_txt_extra_7777"
PARSED_RAW_PAVLOV = "data/parsed_pavlov/"
PARSED_RAW_PAVLOV_UNIQUE = "data/parsed_pavlov_unique/"

ALGO_RESULT_W2V = "medicalTextTrees/algos_results/merged_extended_w2v.txt"
ALGO_RESULT_N2V = "medicalTextTrees/algos_results/merged_extended_n2v.txt"
ALGO_RESULT_W2V_FILT = "medicalTextTrees/algo_results_filtered/merged_extended_w2v.txt"
ALGO_RESULT_N2V_FILT = "medicalTextTrees/algo_results_filtered/merged_extended_n2v.txt"

# Word2vec training constants
UPPER_BOUND_ADDITIONAL_DATA = 50000
HIGH_COSINE_DIST = 0.75
LOAD_TRAINED = True
RUN_WITH_W2V = True
WRITE_IN_FILES = False

# Additional constants
EMPTY_STR = ""
DOT = "."
COMMA = ','
SPACE = " "
UNDERSCORE = '_'
NEW_LINE = "\n"
SEMI_COLON = '; '

# DEEPPAVLOV SPECIFIC
PUNCT_RELATION = 'punct' # 'PUNC' - for parusPipe
NUM_POS = 'NUM'
ADJ_POS = 'ADJ'
NOUN_POS = 'NOUN'
PROPN_POS = 'PROPN'

# PARUS_PIPE SPECIFIC - was used at the start, but then switched to deeppavlov
# PUNCT_RELATION = 'PUNC'
# NUM_POS = 'M'

SENT_NUM = 5000
WORDS_IN_SENT = 35
WORD_LIMIT = 5  # 5

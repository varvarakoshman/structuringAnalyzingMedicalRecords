from const.Constants import *
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import os
import pandas as pd
import shutil
import re

pattern_year = re.compile('^[0-9]{4}$')
pattern_full_date = re.compile('^([0-9]+\.){2}[0-9]+$')
pattern_part_date = re.compile('^[0-9]+\.[0-9][1-9]+$')
only_letters = re.compile("^[a-zA-Z]+$")

df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'toDelete1', 'toDelete2']
months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь',
              'декабрь']
day_times = ['утро', 'день', 'вечер', 'ночь']
seasons = ['лето', 'осень', 'зима', 'весна']
time_labels = ['год', 'г', 'месяц', 'неделя', 'сутки', 'день', 'час', 'минута', 'секунда']


# read partly parsed Vidal reference book and do pre-processing before usage
def read_vidal():
    df_columns = ['INN', 'MNN', 'DrugsSingle', 'DrugsMultiple', 'Group', 'Interactions', 'Indications', 'Contraindications']
    with open('data/dataset_vidal_inns.csv', encoding='utf-8') as f:
        this_df = pd.read_csv(f, sep=';', names=df_columns)
    medications = [ent for ent in list(this_df['DrugsSingle']) if isinstance(ent, str)]
    result_set = set([m for med in medications for m in med.split('\\n')])
    replaced = [res.replace('®', '') for res in result_set]
    df = pd.DataFrame(replaced, columns=['Name'])
    df.to_csv("medicalTextTrees/vidal.csv", sep=',', encoding='utf-8')


# walk though the files in the specified directory and map content to a df with specified columns
def _read_folder_to_df(folder, df_columns):
    files = os.listdir(folder)
    stable_df = []
    for file in files:
        if file != 'Store':
            full_dir = os.path.join(PARSED_RAW_PAVLOV_UNIQUE, file)
            name = file.split('.')[0]
            with open(full_dir, 'rb') as f:
                this_df = pd.read_csv(f, sep='\t', names=df_columns)
                this_df_filtered = this_df[this_df.deprel != PUNCT_RELATION.lower()]  # filter nodes with punctuation
                if this_df['id'].duplicated().any():  # in one sentence can be found multiple syntactic subtrees
                    start_of_subtree_df = list(this_df.groupby(this_df.id).get_group(1).index)
                    boundaries = start_of_subtree_df + [max(list(this_df.index)) + 1]
                    list_of_dfs = [this_df.iloc[boundaries[n]:boundaries[n + 1]] for n in range(len(boundaries) - 1)]
                    local_counter = 1
                    for df in list_of_dfs:
                        df['sent_name'] = name + UNDERSCORE + str(local_counter)
                        stable_df.append(df)
                        local_counter += 1
                else:
                    this_df['sent_name'] = name
                    if this_df_filtered.shape[0] < WORDS_IN_SENT:         # for word2vec
                        stable_df.append(this_df)
    return stable_df


# leave only non-repeating sentences in df
def _filter_unique_sents(stable_df):
    dfs_grouped_by = {}
    for df in stable_df:
        sent_len = len(df)
        if sent_len not in dfs_grouped_by.keys():
            dfs_grouped_by[sent_len] = [df]
        else:
            dfs_grouped_by[sent_len].append(df)
    dfs_filtered = []
    sent_count = 0
    for _, dfs in dfs_grouped_by.items():
        if len(dfs) > 1:
            sent_history = []
            for df in dfs:
                if list(df.form)[0] == list(df.form)[0] and len(df) > 1:
                    sent_words = SPACE.join(list(df.form))
                    if sent_words not in sent_history and not sent_count > SENT_NUM:
                        sent_count += 1
                        sent_history.append(sent_words)
                        dfs_filtered.append(df)
    return dfs_filtered


# util method for debug purposes
# compute stats on num of unique sentences and distribution of sentences' lengths
def _compute_some_stats(dfs_filtered):
    unique_sents = {}
    len_sent = {}
    for df in dfs_filtered:
        this_df = df[df.deprel != PUNCT_RELATION]
        sent_n = list(this_df.sent_name)[0]
        words = list(this_df.form)
        if len(words) not in len_sent.keys():
            len_sent[len(words)] = [sent_n]
        else:
            len_sent[len(words)].append(sent_n)
        sent = UNDERSCORE.join(list(this_df.sent_name)[0].split(UNDERSCORE)[:2])
        # file_name = '_'.join(file.split('.')[0].split('_')[:2])
        if sent not in unique_sents.keys():
            unique_sents[sent] = 1
        else:
            unique_sents[sent] += 1


# delete useless columns, relations of type PUNC and reindex
def _remove_meta_info(dfs_filtered):
    trees_df = pd.concat(dfs_filtered, axis=0, ignore_index=True)
    trees_df = trees_df.drop(columns=['xpostag', 'toDelete1', 'toDelete2'], axis=1)
    strange_sentences = ['143096_4', '306218_1', '250555_5', '129889_5', '331269_13']  # TODO: seach for broken sentences should be automated !
    trees_df = trees_df.loc[~trees_df.sent_name.isin(strange_sentences)]
    trees_df_filtered = trees_df[trees_df.deprel != PUNCT_RELATION]
    trees_df_filtered = trees_df_filtered.reset_index(drop=True)
    trees_df_filtered.index = trees_df_filtered.index + 1
    return trees_df_filtered


def read_data():
    stable_df = _read_folder_to_df(PARSED_RAW_PAVLOV_UNIQUE, df_columns)
    dfs_filtered = _filter_unique_sents(stable_df)
    # compute_some_stats(dfs_filtered) # - for development only
    trees_df_filtered = _remove_meta_info(dfs_filtered)
    return trees_df_filtered


def replace_time_constructions(df):
    for month in months:
        df.loc[df['lemma'] == month, 'lemma'] = '#месяц'
    for day_time in day_times:
        df.loc[df['lemma'] == day_time, 'lemma'] = '#времясуток'
    for season in seasons:
        df.loc[df['lemma'] == season, 'lemma'] = '#сезон'
    df['lemma'] = df['lemma'].apply(_func)
    # replacement is needed mostly to change (month -> Noun) to (month -> Num)
    # and ensure there are no errors in parsing
    replaced_numbers = [k for k, v in df.lemma.str.contains('#').to_dict().items() if v == True]
    for num in replaced_numbers:
        df.loc[num, 'upostag'] = NUM_POS


def _func(lemma):
    if not pd.isnull(lemma):
        if pattern_year.match(lemma):
            return '#год'
        elif pattern_full_date.match(lemma):
            return '#пдата'
        elif pattern_part_date.match(lemma):
            return '#чдата'
        elif lemma == '@card@':
            return '#число'
    return lemma


# PRE-PROCESSING
# method splits input data in 3 datasets:
# 1) stable (ready to run an algorithm),
# 2) very long-read (sentences are too long and need to be split in 2 parts),
# 3) many-rooted (case for compound sentences and incorrect parser's results (Ex: 5 roots in a sentence of length 10)
# and copies files in corresponding directories
# fix 2) and 3) manually and then run the main algorithm, which will walk through these directories and add all files.
def sort_the_data():
    files = os.listdir(DATA_PATH)
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            this_df = this_df[this_df.deprel != PUNCT_RELATION]
        if this_df.groupby(this_df.deprel).get_group('ROOT').shape[0] > 1:
            shutil.copy(full_dir, MANY_ROOTS_DATA_PATH)
        elif this_df.shape[0] > 23:
            name_split = file.split(DOT)
            shutil.copy(os.path.join(ORIGINAL_DATA_PATH, DOT.join([name_split[0], name_split[1]])), LONG_DATA_PATH)
        else:
            shutil.copy(full_dir, STABLE_DATA_PATH)


# compare new sentences with existent, pick ones that don't duplicate
def pick_new_sentences():
    files = os.listdir(RAW_DATA_PATH)
    sent_set = set()
    unique_sent_names = []
    for file in files:
        full_dir = os.path.join(RAW_DATA_PATH, file)
        with open(full_dir, encoding='utf-8') as f:
            sent = f.readlines()[0].split('\n')[0]
            if sent not in sent_set:
                sent_set.add(sent)
                unique_sent_names.append(file)
    for file in unique_sent_names:
        shutil.copy(os.path.join(PARSED_RAW_PAVLOV, file), PARSED_RAW_PAVLOV_UNIQUE)
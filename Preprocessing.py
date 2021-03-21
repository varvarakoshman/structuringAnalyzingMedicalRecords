import math

from Constants import *
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


def read_vidal():
    df_columns = ['INN', 'MNN', 'DrugsSingle', 'DrugsMultiple', 'Group', 'Interactions', 'Indications', 'Contraindications']
    with open('data/dataset_vidal_inns.csv', encoding='utf-8') as f:
        this_df = pd.read_csv(f, sep=';', names=df_columns)
    medications = [ent for ent in list(this_df['DrugsSingle']) if isinstance(ent, str)]
    result_set = set([m for med in medications for m in med.split('\\n')])
    replaced = [res.replace('®', '') for res in result_set]
    df = pd.DataFrame(replaced, columns=['Name'])
    df.to_csv("medicalTextTrees/vidal.csv", sep=',', encoding='utf-8')


def read_data():
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    # len_30 = ['100701_2', '112228_5', '233471_2', '339012_3', '366067_1', '109912_0', '255861_0', '281689_1', '148632_19', '192213_6', '246011_9', '254050_32', '359728_3', '361037_9', '368882_3_1', '361086_8', '152888_21_1', '129821_30', '270150_19', '246011_11', '221783_0_1', '227529_2', '266948_0_1', '359728_6', '361175_2', '51183_6']
    # len_30 = ['100701_2', '112228_5', '233471_2', '339012_3', '366067_1', '109912_0', '255861_0', '281689_1', '148632_19', '192213_6', '246011_9', '254050_32', '359728_3']
    stable_df = []
    # long_df = []
    unique_sents = {}
    for file in files:
        if file == 'Store':
            hhh = []
        if file != 'Store':
            full_dir = os.path.join(DATA_PATH, file)
            name = file.split('.')[0]
            with open(full_dir, 'rb') as f:
                this_df = pd.read_csv(f, sep='\t', names=df_columns)
                this_df_filtered = this_df[this_df.deprel != 'PUNC']
                if this_df['id'].duplicated().any():
                    start_of_subtree_df = list(this_df.groupby(this_df.id).get_group(1).index)
                    boundaries = start_of_subtree_df + [max(list(this_df.index)) + 1]
                    list_of_dfs = [this_df.iloc[boundaries[n]:boundaries[n + 1]] for n in range(len(boundaries) - 1)]
                    local_counter = 1
                    for df in list_of_dfs:
                        df['sent_name'] = name + '_' + str(local_counter)
                        # full_df.append(df)
                        stable_df.append(df)
                        local_counter += 1
                else:
                    this_df['sent_name'] = name
                    # stable_df.append(this_df)           # for strong equality
                    if this_df_filtered.shape[0] < 30:         # for word2vec
                        stable_df.append(this_df)
                    # else:
                    #     long_df.append(this_df)
                    # full_df.append(this_df)
    dfs_grouped_by = {}
    for df in stable_df: # replace to full
        sent_len = len(df)
        if sent_len not in dfs_grouped_by.keys():
            dfs_grouped_by[sent_len] = [df]
        else:
            dfs_grouped_by[sent_len].append(df)
    dfs_filtered = []
    for _, dfs in dfs_grouped_by.items():
        if len(dfs) > 1:
            sent_history = []
            for df in dfs:
                if list(df.form)[0] == list(df.form)[0]:
                    try:
                        sent_words = ' '.join(list(df.form))
                    except TypeError as te:
                        jjjj = []
                    # dfs_filtered.append(df)
                    if sent_words not in sent_history:
                        sent_history.append(sent_words)
                        dfs_filtered.append(df)
    # trees_df = pd.concat(stable_df, axis=0, ignore_index=True)
    len_sent = {}
    for df in dfs_filtered:
        this_df = df[df.deprel != 'PUNC']
        sent_n = list(this_df.sent_name)[0]
        words = list(this_df.form)
        if len(words) not in len_sent.keys():
            len_sent[len(words)] = [sent_n]
        else:
            len_sent[len(words)].append(sent_n)
        sent = '_'.join(list(this_df.sent_name)[0].split('_')[:2])
        # file_name = '_'.join(file.split('.')[0].split('_')[:2])
        if sent not in unique_sents.keys():
            unique_sents[sent] = 1
        else:
            unique_sents[sent] += 1

    trees_df = pd.concat(dfs_filtered, axis=0, ignore_index=True)
    # long_df = pd.concat(long_df, axis=0, ignore_index=True)
    # delete useless data
    trees_df = trees_df.drop(columns=['xpostag', 'feats'], axis=1)
    # trees_df.drop(index=[11067], inplace=True)
    trees_df.loc[13742, 'deprel'] = 'разъяснит'

    # delete relations of type PUNC and reindex

    # long_df = long_df.drop(columns=['xpostag', 'feats'], axis=1)
    # long_df_filtered = long_df[long_df.deprel != 'PUNC']
    # long_df_filtered = long_df_filtered.reset_index(drop=True)
    # long_df_filtered.index = long_df_filtered.index + 1

    trees_df_filtered = trees_df[trees_df.deprel != 'PUNC']
    trees_df_filtered = trees_df_filtered.reset_index(drop=True)
    trees_df_filtered.index = trees_df_filtered.index + 1
    # trees_df_filtered.loc[12239, 'deprel'] = '1-компл'
    # trees_df_filtered.loc[12239, 'head'] = 2
    #
    # trees_full_df = pd.concat(full_df, axis=0, ignore_index=True)
    # trees_full_df = trees_full_df.reset_index(drop=True)
    # trees_full_df.index = trees_full_df.index + 1
    # trees_full_df.drop(columns=['upostag', 'xpostag', 'feats'], axis=1)
    # trees_full_df = trees_full_df[trees_full_df.deprel != 'PUNC']

    # replaced_numbers = [k for k, v in trees_full_df.lemma.str.contains('#').to_dict().items() if
    #                     v == True]
    # for num in replaced_numbers:
    #     trees_full_df.loc[num, 'upostag'] = 'Num'

    replaced_numbers = [k for k, v in trees_df_filtered.lemma.str.contains('#').to_dict().items() if v == True]
    for num in replaced_numbers:
        trees_df_filtered.loc[num, 'upostag'] = 'Num'

    # target_sents = list({'44112_8', '38674_5', '55654_2', '35628_5'})
    # target_sents = list({'44112_8', '55654_2', '32867_6', '57809_7'})  # TEST
    # target_sents = list({'55654_2', '35628_5', '32867_6', '57809_7', '57126_7'})  # TEST
    # target_sents = list({'57809_7', '57126_7'})  # TEST
    # target_sents = list({'55338_41', '58401_7'})  # TEST
    # target_sents = list({'46855_3', '48408_0', '37676_3', '56109_5', '56661_0', '54743_1'}) # TEST !!!!trickyyyy
    # target_sents = list({'37535_4', '31635_2', '39786_8'}) # TEST !!!!trickyyyy
    # target_sents = list({'48408_0', '37676_3', '32191_0', '56109_5', '56661_0', '54743_1'}) # TEST
    # target_sents = list({'46855_3', '48408_0', '37676_3', '56661_0'})  # TEST !!!!!!!!
    # target_sents = list({'46855_3', '37676_3', '54743_1'})  # TEST
    # target_sents = list({'58282_3', '46855_3', '37676_3'}) # TEST
    # target_sents = list({'32191_2', '58282_3', '55066_0', '46855_3', '48408_0'})
    # target_sents = list({'53718_0', '46007_0', '56109_2', '41184_0'}) # test for plain
    # target_sents = list({'167529_9', '152369_9', '172030_9', '172030_23', '48408_0'}) # meeeeeess

    # test_df = trees_df_filtered.loc[trees_df_filtered.sent_name.isin(target_sents)] # TEST
    # trees_full_df.loc[trees_full_df.index.isin(replaced_numbers)].assign(upostag = 'N')

    # trees_df_filtered = trees_df_filtered.head(513)
    # trees_df_filtered = trees_df_filtered.head(339)
    # trees_df_filtered = trees_df_filtered.head(411)
    # trees_df_filtered = trees_df_filtered.head(431)
    # trees_df_filtered = trees_df_filtered.head(4824)
    return trees_df_filtered


# PRE-PROCESSING
# method splits input data in 3 datasets:
# 1) stable (ready to run an algorithm),
# 2) very long-read (sentences are too long and need to be split in 2 parts),
# 3) many-rooted (case for compound sentences and incorrect parser's results (Ex: 5 roots in a sentence of length 10)
# and copies files in corresponding directories
# fix 2) and 3) manually and then run the main algorithm, which will walk through these directories and add all files.
def sort_the_data():
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            this_df = this_df[this_df.deprel != 'PUNC']
        if this_df.groupby(this_df.deprel).get_group('ROOT').shape[0] > 1:
            shutil.copy(full_dir, MANY_ROOTS_DATA_PATH)
        elif this_df.shape[0] > 23:
            name_split = file.split(DOT)
            shutil.copy(os.path.join(ORIGINAL_DATA_PATH, DOT.join([name_split[0], name_split[1]])), LONG_DATA_PATH)
        else:
            shutil.copy(full_dir, STABLE_DATA_PATH)


def replace_time_constructions(df):
    months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь',
              'декабрь']
    day_times = ['утро', 'день', 'вечер', 'ночь']
    seasons = ['лето', 'осень', 'зима', 'весна']
    for month in months:
        df.loc[df['lemma'] == month, 'lemma'] = '#месяц'
    for day_time in day_times:
        df.loc[df['lemma'] == day_time, 'lemma'] = '#времясуток'
    for season in seasons:
        df.loc[df['lemma'] == season, 'lemma'] = '#сезон'
    df['lemma'] = df['lemma'].apply(func)


def func(lemma):
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


# compare new sentences with existent, pick ones that don't duplicate
def pick_new_sentences():
    existing_sent = []
    existing_len = []
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            sent_str = EMPTY_STR.join(list(this_df.form))
            existing_sent.append(sent_str)
            existing_len.append(len(sent_str))
    files_new = os.listdir(r"parus_results_tags_additional")
    for file in files_new:
        full_dir = os.path.join(r"parus_results_tags_additional", file)
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            sent_str = EMPTY_STR.join(list(this_df.form))
            if len(sent_str) not in existing_len:
                if sent_str not in existing_sent:
                    shutil.copy(full_dir, "brand_new_sentences")

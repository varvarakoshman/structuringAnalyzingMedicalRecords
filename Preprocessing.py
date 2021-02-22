#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import os
import pandas as pd

from Constants import DATA_PATH


def read_data():
    files = os.listdir(DATA_PATH)
    df_columns = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel']
    full_df = []
    stable_df = []
    long_df = []
    for file in files:
        full_dir = os.path.join(DATA_PATH, file)
        name = file.split('.')[0]
        with open(full_dir, encoding='utf-8') as f:
            this_df = pd.read_csv(f, sep='\t', names=df_columns)
            this_df_filtered = this_df[this_df.deprel != 'PUNC']
            if this_df['id'].duplicated().any():
                start_of_subtree_df = list(this_df.groupby(this_df.id).get_group(1).index)
                boundaries = start_of_subtree_df + [max(list(this_df.index)) + 1]
                list_of_dfs = [this_df.iloc[boundaries[n]:boundaries[n + 1]] for n in range(len(boundaries) - 1)]
                local_counter = 1
                for df in list_of_dfs:
                    df['sent_name'] = name + '_' + str(local_counter)
                    full_df.append(df)
                    stable_df.append(df)
                    local_counter += 1
            else:
                this_df['sent_name'] = name
                # stable_df.append(this_df)           # for strong equality
                if this_df_filtered.shape[0] < 23:         # for word2vec
                    stable_df.append(this_df)
                else:
                    long_df.append(this_df)
                full_df.append(this_df)
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
                sent_words = ' '.join(list(df.form))
                if sent_words not in sent_history:
                    sent_history.append(sent_words)
                    dfs_filtered.append(df)
    # trees_df = pd.concat(stable_df, axis=0, ignore_index=True)
    trees_df = pd.concat(dfs_filtered, axis=0, ignore_index=True)
    long_df = pd.concat(long_df, axis=0, ignore_index=True)
    # delete useless data
    trees_df = trees_df.drop(columns=['xpostag', 'feats'], axis=1)
    # trees_df.drop(index=[11067], inplace=True)
    trees_df.loc[13742, 'deprel'] = 'разъяснит'

    # delete relations of type PUNC and reindex

    long_df = long_df.drop(columns=['xpostag', 'feats'], axis=1)
    long_df_filtered = long_df[long_df.deprel != 'PUNC']
    long_df_filtered = long_df_filtered.reset_index(drop=True)
    long_df_filtered.index = long_df_filtered.index + 1

    trees_df_filtered = trees_df[trees_df.deprel != 'PUNC']
    trees_df_filtered = trees_df_filtered.reset_index(drop=True)
    trees_df_filtered.index = trees_df_filtered.index + 1
    # trees_df_filtered.loc[12239, 'deprel'] = '1-компл'
    # trees_df_filtered.loc[12239, 'head'] = 2

    trees_full_df = pd.concat(full_df, axis=0, ignore_index=True)
    trees_full_df = trees_full_df.reset_index(drop=True)
    trees_full_df.index = trees_full_df.index + 1
    trees_full_df.drop(columns=['upostag', 'xpostag', 'feats'], axis=1)
    trees_full_df = trees_full_df[trees_full_df.deprel != 'PUNC']

    replaced_numbers = [k for k, v in trees_full_df.lemma.str.contains('#').to_dict().items() if
                        v == True]
    for num in replaced_numbers:
        trees_full_df.loc[num, 'upostag'] = 'Num'

    replaced_numbers = [k for k, v in trees_df_filtered.lemma.str.contains('#').to_dict().items() if
                        v == True]
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

    # trees_df_filtered = trees_full_df.loc[trees_full_df.sent_name.isin(target_sents)] # TEST
    # trees_full_df.loc[trees_full_df.index.isin(replaced_numbers)].assign(upostag = 'N')

    # trees_df_filtered = trees_df_filtered.head(513)
    # trees_df_filtered = trees_df_filtered.head(339)
    # trees_df_filtered = trees_df_filtered.head(411)
    # trees_df_filtered = trees_df_filtered.head(431)
    return trees_full_df, trees_df_filtered, long_df_filtered

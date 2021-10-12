#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import csv
import operator
import os
import re
from collections import defaultdict
from itertools import permutations
import networkx as nx
import netgraph  # pip install netgraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from Tree import Edge
from WikidataEntity import WikidataEntity
from const.Constants import *
from Preprocessing import time_labels
from sql.postgres_repo import get_entity_fields, select_all_ref, select_all_main

pattern_verb_check = re.compile('(([Z])*([QIRPCS])*)+')
pattern_help_verbs_in_row = re.compile("^Z{2,}.*$")
pattern_not_help_verb = r'[^Z]'
has_russian_letters = re.compile('^.*[а-яА-Я]+.*$')


def create_needed_directories():
    if not os.path.exists(CLASSIFIED_DATA_PATH):
        os.makedirs(CLASSIFIED_DATA_PATH)
    if not os.path.exists(LONG_DATA_PATH):
        os.makedirs(LONG_DATA_PATH)
    if not os.path.exists(STABLE_DATA_PATH):
        os.makedirs(STABLE_DATA_PATH)
    if not os.path.exists(MANY_ROOTS_DATA_PATH):
        os.makedirs(MANY_ROOTS_DATA_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)


def write_tree_in_table(whole_tree):
    with open(EXCEL_EDGES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_1, open(
            EXCEL_NODES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_2:
        writer_1 = csv.writer(csv_file_1, delimiter=',')
        writer_2 = csv.writer(csv_file_2, delimiter=',')
        writer_1.writerow(['Source', 'Target', 'Weight'])
        writer_2.writerow(['Id', 'Label'])
        for edge in whole_tree.edges:
            writer_1.writerow([edge.node_from, edge.node_to, edge.weight + 1])
        for node in whole_tree.nodes:
            writer_2.writerow([node.id, (node.lemma, node.form, node.sent_name)])


def read_med_data():
    df_disease = pd.read_excel(MEDICAL_DICTIONARY, sheet_name='Болезнь')
    diseases = [x.lower() for x in list(df_disease.Name)]
    df_symp = pd.read_excel(MEDICAL_DICTIONARY, sheet_name='Симптом')
    symptoms = [x.lower() for x in df_symp.Name.to_list()]
    df_doc = pd.read_excel(MEDICAL_DICTIONARY, sheet_name='Врач')
    docs = [x.lower() for x in df_doc.Name.to_list()]
    df_drug = pd.read_excel(MEDICAL_DICTIONARY, sheet_name='Лекарство')
    drugs = [x.lower() for x in df_drug.Name.to_list()]
    return diseases, symptoms, docs, drugs


# 'Болезнь' - 0, 'Симптом' - 1, 'Врач' - 2, 'Лекарство' - 3, 'Временная метка' - 4
def label_lemmas(lemmas_list):
    diseases, symptoms, docs, drugs = read_med_data()
    times = ['#месяц', '#времясуток', '#сезон'] + time_labels
    lemmas_labels = {}
    for lemma in lemmas_list:
        if lemma in diseases:
            label = 0
        elif lemma in symptoms:
            label = 1
        elif lemma in docs:
            label = 2
        elif lemma in drugs:
            label = 3
        elif lemma in times:
            label = 4
        else:
            label = -1
        if label != -1:
            lemmas_labels[lemma] = label
    return lemmas_labels


def label_classes(classes_words, classes_count_passive_verbs):
    diseases, symptoms, docs, drugs = read_med_data()
    labels = ['Болезнь', 'Симптом', 'Врач', 'Лекарство', 'Событие', 'Временная метка']
    dis, sym, doc, drg, event, time = labels
    # times = months + day_times + seasons
    times = ['#месяц', '#времясуток', '#сезон'] + time_labels
    class_labels = {}
    for class_id, words in classes_words.items():
        freqs = dict(zip([dis, sym, doc, drg, event, time], [0, 0, 0, 0, 0, 0]))
        for lemma in words:
            if lemma in diseases:
                freqs[dis] += 1
            elif lemma in symptoms:
                freqs[sym] += 1
            elif lemma in docs:
                freqs[doc] += 1
            elif lemma in drugs:
                freqs[drg] += 1
            elif lemma in times:
                freqs[time] += 1
        # if all(value == 0 for value in freqs.values()):
        #     freqs[event] = classes_count_passive_verbs[class_id]
        sorted_d = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
        max_count = max(sorted_d.values())
        max_labels = [k for k, v in sorted_d.items() if v == max_count]
        if max(sorted_d.values()) > 0:
            possible_labels = max_labels
        else:
            possible_labels = []
        if class_id in classes_count_passive_verbs.keys() and classes_count_passive_verbs[class_id] > 0:
            possible_labels.append(labels[4])
        class_labels[class_id] = possible_labels
    return class_labels


# POST-PROCESSING
def merge_in_file(path_from, path_to):
    files = os.listdir(path_from)
    files_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    writer = open(path_to, 'w', encoding='utf-8')
    try:
        for file in files_sorted:
            full_dir = os.path.join(path_from, file)
            try:
                with open(full_dir, encoding='utf-8') as reader:
                    class_entries = reader.readlines()
            finally:
                reader.close()
            for entry in class_entries:
                writer.write(entry)
            writer.write('\n')
    finally:
        writer.close()


# POST-PROCESSING
def filter_classes(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_form_lemma):
    count = 0
    sent_unique_class_content = {}
    classes_wordgroups = {}
    classes_sents = {}
    surviving_strings_classes = {}
    all_forms = set()
    for class_id, root_ids in classes_part.items():
        vertex_seq = {}
        count += 1
        temp_sent_set = set()
        vertex_seq_ids = {}
        for vertex_id in root_ids:
            sent_id = whole_tree.get_node(vertex_id).sent_name
            temp_sent_set.add(sent_id)
            vertex_seq[vertex_id] = whole_tree.simple_dfs(vertex_id, classes_part_list[class_id])
        if len(vertex_seq.items()) > 0 and len(vertex_seq[list(vertex_seq)[0]]) > 1:
            for v_id, entries in vertex_seq.items():
                sent_name = remapped_sent_rev[entries[0].sent_name]
                for ent in entries:  # FOR WRITING ALL LEMMAS
                    all_forms.add(ent.form)
                if class_id not in classes_wordgroups.keys():
                    classes_wordgroups[class_id] = [tuple(entries)]
                    classes_sents[class_id] = [sent_name]
                else:
                    classes_wordgroups[class_id].append(tuple(entries))
                    classes_sents[class_id].append(sent_name)
                vertex_seq_ids[v_id] = list(
                    map(lambda list_entry: str(whole_tree.get_edge(list_entry.id)[0].weight) + list_entry.form,
                        entries))
        sent_label = tuple(temp_sent_set)
        if sent_label not in sent_unique_class_content.keys():
            new_entries_set = set([tuple(i) for i in list(vertex_seq_ids.values())])
            sent_unique_class_content[sent_label] = new_entries_set
            for new_array_label in list(new_entries_set):
                surviving_strings_classes[new_array_label] = class_id
        else:
            already_checked = set()
            for v_id, new_array in vertex_seq_ids.items():
                new_array_label = tuple(new_array)
                if new_array_label not in already_checked:
                    completely_new = new_array_label not in sent_unique_class_content[sent_label]
                    if completely_new:
                        filter_condition = False
                        is_new_nested_in_existing = False
                        ids_to_filter = set()
                        for existing_array in sent_unique_class_content[sent_label]:
                            is_existing_nested_in_new = all(
                                entr in new_array for entr in existing_array) and existing_array != new_array
                            is_new_nested_in_existing = is_new_nested_in_existing or all(
                                entr in existing_array for entr in new_array)
                            filter_condition = filter_condition or (is_existing_nested_in_new)
                            if is_existing_nested_in_new:
                                ids_to_filter.add(existing_array)
                        if filter_condition:
                            sent_unique_class_content[sent_label] = sent_unique_class_content[
                                                                        sent_label] - ids_to_filter
                        if not is_new_nested_in_existing:
                            sent_unique_class_content[sent_label].add(new_array_label)
                    already_checked.add(new_array_label)
                    surviving_strings_classes[new_array_label] = class_id
    classes_ids_filtered = set(
        [surviving_strings_classes[sent_group] for sent_groups in sent_unique_class_content.values() for sent_group in
         list(sent_groups)])
    classes_wordgroups_filtered = {k: v for k, v in classes_wordgroups.items() if k in classes_ids_filtered}
    classes_sents_filtered = {k: v for k, v in classes_sents.items() if k in classes_ids_filtered}
    all_lemmas = set([dict_form_lemma[form] for form in all_forms])
    write_all_lemmas(all_lemmas)
    return classes_wordgroups_filtered, classes_sents_filtered


# filter long subsequences of prepositions
def check1st(node_sequence):
    seq_pos_tags = list(map(lambda node: node.pos_tag, node_sequence))
    target_indices = [index for index, pos_tag in enumerate(seq_pos_tags) if pos_tag == 'S' or pos_tag == 'C']
    if len(target_indices) > 0:
        # prep_max_len = 0
        # curr_len = 1
        # for i in range(0, len(target_indices) - 1):
        #     if target_indices[i + 1] - target_indices[i] == 1:
        #         curr_len += 1
        #     else:
        #         curr_len = 1
        #     if curr_len > prep_max_len:
        #         prep_max_len = curr_len
        if len(target_indices) > len(seq_pos_tags) - len(target_indices):
            return False  # filter if a sequence of prepositions takes the most of a string
    return True


def get_tags_merged(node_sequence, dict_form_lemma, verbs_to_filter):
    return EMPTY_STR.join(
        [node.pos_tag if dict_form_lemma[node.form] not in verbs_to_filter else 'Z' for node in node_sequence])


# filter sequences with meaningless verbs
def check2nd(node_sequence, dict_form_lemma, verbs_to_filter):
    # Ex: была где был откуда : "ZCZC"
    mapped_tags_str = get_tags_merged(node_sequence, dict_form_lemma, verbs_to_filter)
    matched_substring = pattern_verb_check.match(mapped_tags_str)
    if matched_substring is not None and len(matched_substring.group(0)) > len(mapped_tags_str) - len(
            matched_substring.group(0)):
        return False
    return True


def get_squashed_seq(node_sequence, mapped_tags_str):
    start_index = re.search(pattern_not_help_verb, mapped_tags_str).start()
    squashed_seq = node_sequence[start_index:]
    return squashed_seq


def filter_meaningless_classes(classes_wordgroups_filtered, dict_form_lemma):
    meaningful_classes = {}
    meaningless_classes = {}
    verbs_to_filter = ['быть', 'стать']
    for class_id, node_sequences_list in classes_wordgroups_filtered.items():
        node_sequence = node_sequences_list[
            0]  # checking the 1st entry of a class (considering others don't differ in structure)
        has_help_verb_as_root = dict_form_lemma[node_sequence[0].form] in verbs_to_filter
        if check1st(node_sequence) and (
                not has_help_verb_as_root or check2nd(node_sequence, dict_form_lemma, verbs_to_filter)):
            mapped_tags_str = get_tags_merged(node_sequence, dict_form_lemma, verbs_to_filter)
            if has_help_verb_as_root and pattern_help_verbs_in_row.match(mapped_tags_str):
                squashed_seq_list = []
                for node_seq in node_sequences_list:
                    squashed_seq_list.append(get_squashed_seq(node_seq, mapped_tags_str))
                meaningful_classes[class_id] = squashed_seq_list
            else:
                meaningful_classes[class_id] = node_sequences_list
        else:
            meaningless_classes[class_id] = node_sequences_list
    return meaningful_classes, meaningless_classes


def get_all_words(meaningful_classes, dict_form_lemma):
    classes_words = {}
    classes_count_passive_verbs = {}
    for class_id, node_sequences_list in meaningful_classes.items():
        # count_passive_verbs = len(list(filter(lambda node: node.pos_tag == 'V' and node.pos_extended[-3] == 'p',
        #                                       [node for node_seq in node_sequences_list for node in node_seq])))
        count_passive_verbs = len(list(filter(lambda node: node.pos_tag == 'VERB' and 'Voice=Pass' in node.pos_extended,
                                              [node for node_seq in node_sequences_list for node in node_seq])))
        classes_count_passive_verbs[class_id] = count_passive_verbs
        words_in_class = set([node.form for node_seq in node_sequences_list for node in node_seq])
        classes_words[class_id] = list(map(lambda form: dict_form_lemma[form], words_in_class))
    return classes_words, classes_count_passive_verbs


def write_all_lemmas(all_lemmas):
    filename = ALL_LEMMAS_PATH_W2V
    try:
        with open(filename, 'w', encoding='utf-8') as filehandle:
            for lemma in all_lemmas:
                filehandle.write("%s\n" % lemma)
    finally:
        filehandle.close()


def write_classes_in_txt(whole_tree, meningful_classes, classes_sents_filtered, new_classes_mapping, dict_rel_rev,
                          path, class_id_labels_full):
    count = 1
    for class_id, node_seq_list in meningful_classes.items():
        filename = path + '/%s.txt' % (str(count))
        try:
            with open(filename, 'w', encoding='utf-8') as filehandle:
                # if len(class_labels) > 0:  # empty dict comes for useless classes, which are also logged in file
                #     filehandle.write("label: %s\n" % (SPACE.join(str(i) for i in class_labels[new_classes_mapping[class_id][0]])))
                # filehandle.write("class_id: %s\n" % (str(class_id)))
                if len(class_id_labels_full) > 0:
                    if class_id in class_id_labels_full.keys():
                        s = COMMA.join(lab for lab in list(class_id_labels_full[class_id]))
                        filehandle.write("label: %s\n" % (s))
                for repeat_count, node_seq in enumerate(node_seq_list):
                    joined_res_str = SPACE.join(list(map(lambda node: '(' + str(dict_rel_rev[whole_tree.get_edge(node.id)[0].weight]) + ') ' + node.form + ' /' + node.pos_tag + '/ ', node_seq)))
                    try:
                        filehandle.write("sent=%s: %s\n" % (classes_sents_filtered[class_id][repeat_count], joined_res_str))  # WRONG!!!! SENT!!!!
                    except IndexError as ke:
                        gggg = []
        finally:
            filehandle.close()
        count += 1


def extract_all_unique_phrases_from_merged_file():
    try:
        with open(ALGO_RESULT_N2V, 'r', encoding='utf-8') as reader:
            class_entries = reader.readlines()
    finally:
        reader.close()
    unique_phrases = set()
    for line in class_entries:
        if line != NEW_LINE:
            words = line.split(':')[1].split(NEW_LINE)[0].split(SPACE)
            words.remove(EMPTY_STR)
            unique_phrases.add(tuple(words[1::2]))
    filename = "medicalTextTrees/all_unique_phrases_forms.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as writer:
            for phrase in unique_phrases:
                writer.write("%s\n" % SPACE.join(phrase))
    finally:
        writer.close()


# POST-PROCESSING
def write_in_file_old(classes_part, classes_part_list, whole_tree, remapped_sent_rev, dict_rel_rev):
    count = 0
    classes_words = {}
    for k, v in classes_part.items():
        vertex_seq = {}
        count += 1
        for vertex in v:
            curr_height = max(whole_tree.heights[vertex])
            vertex_seq[vertex] = whole_tree.simple_dfs(vertex, classes_part_list[k])
            if len(vertex_seq.items()) > 0 and len(vertex_seq[list(vertex_seq)[0]]) > 1:
                filename = RESULT_PATH_OLD + '/%s.txt' % (str(count))
                try:
                    with open(filename, 'w', encoding='utf-8') as filehandle:
                        for _, value in vertex_seq.items():
                            for val in value:
                                node = whole_tree.get_node(val[0])
                                if node.res_class is None:
                                    node.res_class = count
                            words = list(map(lambda list_entry: list_entry[2], value))
                            if count not in classes_words.keys():
                                classes_words[count] = words
                            filehandle.write("len=%d h=%d sent=%s %s: %s\n" % (
                            len(value), curr_height, value[0][1], remapped_sent_rev[value[0][3]],
                            SPACE.join(SPACE.join(list(map(
                                lambda list_entry: '(' + str(dict_rel_rev[list_entry[4]]) + ') ' + str(list_entry[2]),
                                value))).split(' ')[1:])))
                finally:
                    filehandle.close()
    return classes_words


def get_joint_lemmas(path1, path2):
    try:
        with open(path1, 'r', encoding='utf-8') as reader1, open(path2, 'r', encoding='utf-8') as reader2:
            lemmas1 = reader1.readlines()
            lemmas2 = reader2.readlines()
    finally:
        reader1.close()
        reader2.close()
    joint_lemmas = set(lemmas1).intersection(set(lemmas2))
    cleared_lemmas = list(
        filter(lambda lemma: lemma != '.', list(map(lambda lemma: lemma.split('\n')[0], joint_lemmas))))
    return cleared_lemmas


def write_dict_in_file(similar_lemmas_dict_filtered):
    filename = "medicalTextTrees/n2v_similar_words.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as filehandle:
            for word, similar_words in similar_lemmas_dict_filtered.items():
                filehandle.write("%s : %s\n" % (word, tuple(similar_words)))
    finally:
        filehandle.close()


# def write_tree_in_table(whole_tree, dict_rel_rev, labels):
#     source_1 = 'medicalTextTrees/gephi_edges_import_word2vec.csv'
#     source_2 = 'medicalTextTrees/gephi_nodes_import_word2vec.csv'
#     with open(source_1, "w", newline='', encoding='utf-8') as csv_file_1, open(
#             source_2, "w", newline='', encoding='utf-8') as csv_file_2:
#         writer_1 = csv.writer(csv_file_1, delimiter=',')
#         writer_2 = csv.writer(csv_file_2, delimiter=',')
#         writer_1.writerow(['Source', 'Target', 'Weight'])
#         writer_2.writerow(['Id', 'Label', 'Class', 'Class_Label'])
#         included_nodes = list(filter(lambda x: x.res_class is not None, whole_tree.nodes))
#         included_ids = [node.id for node in included_nodes]
#         for edge in whole_tree.edges:
#             if edge.node_to in included_ids and edge.node_from in included_ids:
#                 writer_1.writerow([edge.node_from, edge.node_to, dict_rel_rev[edge.weight]])
#         for node in included_nodes:
#             writer_2.writerow(
#                 [node.id, (node.lemma, node.form, node.sent_name), node.res_class, labels[node.res_class]])


def sort_already_logged(path):
    try:
        with open(path, 'r', encoding='windows-1251') as reader:
            class_entries = reader.readlines()
    finally:
        reader.close()
    count = 0
    result_dict = {}
    for line in class_entries:
        if line != NEW_LINE:
            if count not in result_dict.keys():
                result_dict[count] = [line]
            else:
                result_dict[count].append(line)
        else:
            count += 1
    result_dict_filtered = dict(sorted(result_dict.items(), key=lambda x: len(x[1]), reverse=True))
    return result_dict_filtered


def write_sorted_res_in_file(result_dict, path):
    try:
        with open(path, 'w', encoding='utf-8') as filehandle:
            for class_values in result_dict.values():
                for class_value in class_values:
                    filehandle.write(class_value)
                filehandle.write(NEW_LINE)
    finally:
        filehandle.close()


def squash_classes(whole_tree, meaningful_classes_filtered, dict_lemmas_similar, dict_form_lemma_int):
    class_id_repr = {}
    for class_id, entries_list in meaningful_classes_filtered.items():
        class_weights = list(
            sorted([whole_tree.get_edge(entries_list[0][i].id)[0].weight for i in range(1, len(entries_list[0]))]))
        class_lemmas = list(sorted(set(
            [dict_lemmas_similar[dict_form_lemma_int[entry.form]] for entries in entries_list for entry in entries])))
        class_id_repr[class_id] = tuple(class_lemmas + class_weights)
    grouped_classes = defaultdict(list)
    for key, val in sorted(class_id_repr.items()):
        grouped_classes[val].append(key)
    meaningful_classes_filtered_squashed = {}
    new_classes_mapping = {}
    count = 1
    for classes_group in grouped_classes.values():
        merged_class = meaningful_classes_filtered[classes_group[0]]
        new_classes_mapping[count] = [classes_group[0]]
        for i in range(1, len(classes_group)):
            new_classes_mapping[count].append(classes_group[i])
            merged_class = merged_class + meaningful_classes_filtered[classes_group[i]]
        meaningful_classes_filtered_squashed[count] = merged_class
        count += 1
    return meaningful_classes_filtered_squashed, new_classes_mapping


def add_label_to_list(tup, label_q_id):
    q_id = tup[0]
    label = tup[1]
    norm_label = tup[2]
    if label not in label_q_id.keys():
        label_q_id[label] = {q_id}
    else:
        label_q_id[label].add(q_id)
    if norm_label not in label_q_id.keys():
        label_q_id[norm_label] = {q_id}
    else:
        label_q_id[norm_label].add(q_id)


def add_label_to_list_csv(df, label_q_id, q_id_categories):
    df_columns = df.columns
    for index, row in df.iterrows():
        q_id = row['q_id']
        label = row['entity_label']
        norm_label = row['entity_label_normal']
        if label not in label_q_id.keys():
            label_q_id[label] = {q_id}
        else:
            label_q_id[label].add(q_id)
        if norm_label not in label_q_id.keys():
            label_q_id[norm_label] = {q_id}
        else:
            label_q_id[norm_label].add(q_id)
        if 'instance_of' in df_columns:
            inst = EMPTY_STR if type(row['instance_of']) == float and np.isnan(row['instance_of']) else row['instance_of']
            sub =  EMPTY_STR if type(row['subclass_of']) == float and np.isnan(row['subclass_of']) else row['subclass_of']
            part = EMPTY_STR if type(row['part_of']) == float and np.isnan(row['part_of']) else row['part_of']
            q_id_categories[q_id] = tuple([inst, sub, part])


def label_data_with_wiki(meaningful_classes_filtered_squashed_sort, dict_form_lemma_str, class_labels, new_classes_mapping):
    class_id_tokens = {}
    class_extended_repeats = {}
    label_q_id = {}
    q_id_categories = {}
    remember_disambiguated_tokens = {}
    df_main = pd.read_csv("/Users/vkoschman/PycharmProjects/equalSubtreeProblem/wikidata_main_entities.csv")
    df_ref = pd.read_csv("/Users/vkoschman/PycharmProjects/equalSubtreeProblem/wikidata_aliases_for_entities.csv")
    add_label_to_list_csv(df_main, label_q_id, q_id_categories)
    add_label_to_list_csv(df_ref, label_q_id, q_id_categories)
    all_words_in_db = set([word for key in label_q_id.keys() for word in key.split(SPACE)])
    new_classid_label = {}
    # create search tokens of diff length from class content
    for class_merged_id, class_entries in meaningful_classes_filtered_squashed_sort.items():
        labels = set([class_label for old_id in new_classes_mapping[class_merged_id] for class_label in class_labels[old_id]])
        if class_merged_id not in new_classid_label.keys():
            new_classid_label[class_merged_id] = labels
        else:
            new_classid_label[class_merged_id].update(labels)
        # if time_label not in labels:
        for class_entry in class_entries:
            words = [dict_form_lemma_str[node.form] for node in class_entry]
            pos_tags = [node.pos_tag for node in class_entry]
            if NOUN_POS in pos_tags:
                filtered_indices = [i for i, x in enumerate(pos_tags) if x == NOUN_POS or x == ADJ_POS]
                start_index = 0
                subseqs = set()
                if len(filtered_indices) > 0:
                    curr_len = 1
                    for i in range(0, len(filtered_indices) - 1):
                        if filtered_indices[i + 1] - filtered_indices[i] == 1:
                            curr_len += 1
                        else:
                            target_indices = filtered_indices[start_index:start_index + curr_len]
                            word_seq = tuple(set([words[index] for index in target_indices]))
                            subseqs.add(word_seq)
                            start_index = curr_len
                            curr_len = 1
                    subseqs.add(tuple([words[index] for index in filtered_indices[start_index:start_index + curr_len]]))
                if class_merged_id not in class_id_tokens.keys():
                    class_id_tokens[class_merged_id] = subseqs
                else:
                    class_id_tokens[class_merged_id].update(subseqs)
        if class_merged_id in class_id_tokens.keys():
            new_extended_set = set()
            for repeat in list(class_id_tokens[class_merged_id]):
                repeat_filtered = [word for word in repeat if word in all_words_in_db]
                if len(repeat_filtered) > 1:
                    for i in range(1, len(repeat_filtered) + 1):
                        new_extended_set.update(
                            list(map(lambda x: SPACE.join(x), list(permutations(repeat_filtered, i)))))
                else:
                    if len(repeat_filtered) != 0:
                        new_extended_set.add(repeat_filtered[0])
            class_extended_repeats[class_merged_id] = new_extended_set
    # load trained W2V model
    w2v_joined_model = Word2Vec.load('trained_node2vec_joined.model')
    class_id_labels = {}
    for class_id, search_tokens in class_extended_repeats.items():
        for search_token in search_tokens:
            if search_token in label_q_id.keys():
                q_ids = list(label_q_id[search_token])
                if len(q_ids) > 1:
                    if search_token not in remember_disambiguated_tokens.keys():
                        most_probable = disambiguate_entities(w2v_joined_model, q_id_categories, q_ids, search_token)
                        remember_disambiguated_tokens[search_token] = most_probable
                    else:
                        most_probable = remember_disambiguated_tokens[search_token]
                else:
                    most_probable = pick_first_category(q_id_categories, q_ids[0])
                if most_probable != EMPTY_STR:
                    if class_id not in class_id_labels.keys():
                        class_id_labels[class_id] = {most_probable}
                    else:
                        class_id_labels[class_id].add(most_probable)
    return class_id_labels, new_classid_label


def pick_first_category(q_id_categories, q_id):
    categories = q_id_categories[q_id]
    instance_of = categories[0]
    subclass_of = categories[1]
    part_of = categories[2]
    most_probable = EMPTY_STR
    if instance_of != EMPTY_STR:
        most_probable = get_russian_match(instance_of)
    elif subclass_of != EMPTY_STR:
        most_probable = get_russian_match(subclass_of)
    elif part_of != EMPTY_STR:
        most_probable = get_russian_match(part_of)
    return most_probable


def disambiguate_meanings(w2v_joined_model, possible_options, search_token):
    possible_labels_similarity_score = {}
    for possible_label in possible_options:
        possible_label = possible_label.strip()
        if possible_label in w2v_joined_model.wv.vocab:
            try:
                possible_labels_similarity_score[possible_label] = w2v_joined_model.wv.similarity(possible_label, search_token)
            except KeyError as ke:
                gg = []
    options_sorted = dict(sorted(possible_labels_similarity_score.items(), key=operator.itemgetter(1), reverse=True))
    if len(options_sorted) > 0:
        most_probable = next(iter(options_sorted))
    else:
        most_probable = EMPTY_STR
    return most_probable


def disambiguate_entities(w2v_joined_model, q_id_categories, q_ids, search_token):
    possible_labels_similarity_score = {}
    for q_id in q_ids:
        possible_label = pick_first_category(q_id_categories, q_id)
        if possible_label != EMPTY_STR and possible_label in w2v_joined_model.wv.vocab:
            try:
                possible_labels_similarity_score[possible_label] = w2v_joined_model.wv.similarity(possible_label, search_token)
            except KeyError as ke:
                gg = []
    options_sorted = dict(sorted(possible_labels_similarity_score.items(), key=operator.itemgetter(1), reverse=True))
    if len(options_sorted) > 0:
        most_probable = next(iter(options_sorted))
    else:
        most_probable = EMPTY_STR
    return most_probable


# def get_all_wikidata_entities(label_getter):
#     all_ref_entities = get_entity_fields(select_all_ref)
#     q_id_aliases = {}
#     for tup in all_ref_entities:
#         q_id = tup[0]
#         label = label_getter(tup)
#         if q_id not in q_id_aliases.keys():
#             q_id_aliases[q_id] = [label]
#         else:
#             q_id_aliases[q_id].append(label)
#     all_main_entities = get_entity_fields(select_all_main)
#     all_wikidata_entities = []
#     for main_entity in all_main_entities:
#         q_id = main_entity[0]
#         aliases = q_id_aliases[q_id] if q_id in q_id_aliases.keys() else []
#         instance_of = get_russian_match(main_entity[3]) if main_entity[3] != EMPTY_STR else EMPTY_STR
#         subclass_of = get_russian_match(main_entity[4]) if main_entity[4] != EMPTY_STR else EMPTY_STR
#         part_of = get_russian_match(main_entity[5]) if main_entity[5] != EMPTY_STR else EMPTY_STR
#         wikidata_entity = WikidataEntity(q_id, label=label_getter(main_entity), instance_of=instance_of,
#                                          subclass_of=subclass_of, part_of=part_of, aliases=aliases)
#         all_wikidata_entities.append(wikidata_entity)
#     return all_wikidata_entities

def get_all_wikidata_entities():
    all_ref_entities = get_entity_fields(select_all_ref)
    q_id_aliases = {}
    for tup in all_ref_entities:
        q_id = tup[0]
        norm_label = tup[2]
        if q_id not in q_id_aliases.keys():
            q_id_aliases[q_id] = [norm_label]
        else:
            q_id_aliases[q_id].append(norm_label)
    all_main_entities = get_entity_fields(select_all_main)
    all_wikidata_entities = []
    for main_entity in all_main_entities:
        q_id = main_entity[0]
        aliases = q_id_aliases[q_id] if q_id in q_id_aliases.keys() else []
        instance_of = get_split_russian(main_entity[3]) if main_entity[3] != EMPTY_STR else EMPTY_STR
        subclass_of = get_split_russian(main_entity[4]) if main_entity[4] != EMPTY_STR else EMPTY_STR
        part_of = get_split_russian(main_entity[5]) if main_entity[5] != EMPTY_STR else EMPTY_STR
        wikidata_entity = WikidataEntity(q_id, label=main_entity[2], instance_of=instance_of,
                                         subclass_of=subclass_of, part_of=part_of, aliases=aliases)
        all_wikidata_entities.append(wikidata_entity)
    return all_wikidata_entities


def get_split_russian(entity):
    possible_options = []
    for tag in entity.split(COMMA):
        if has_russian_letters.match(tag):
            possible_options.append(tag)
    return possible_options


def get_russian_match(entity):
    for tag in entity.split(COMMA):
        if has_russian_letters.match(tag):
            return tag
    return EMPTY_STR


def construct_db_tree(all_wikidata_entities):
    all_edges = []
    all_db_labels = {}
    edge_labels = {}
    count = 0
    for wikidata_entity in all_wikidata_entities:
        parents = {
            "instance of": wikidata_entity.instance_of,
            "subclass of": wikidata_entity.subclass_of,
            "part of": wikidata_entity.part_of
        }
        # parents = [wikidata_entity.instance_of, wikidata_entity.subclass_of, wikidata_entity.part_of]
        labels = [wikidata_entity.label] + wikidata_entity.aliases
        for name in labels:
            if name not in all_db_labels.keys():
                all_db_labels[name] = count
                count += 1
            for weight, parnts in parents.items():
                if len(parnts) > 0:
                    for parent in parnts:
                        if parent not in all_db_labels.keys():
                            all_db_labels[parent] = count
                            count += 1
                        edge = tuple([parent, name])
                        all_edges.append(Edge(parent, name))
                        edge_labels[edge] = weight
    all_edges = all_edges
    return all_edges, edge_labels, all_db_labels


def draw_db_network(db_tree_edges, edge_labels, all_db_labels):
    # edges_list = [tuple([edge.node_from, edge.node_to]) for edge in db_tree_edges]
    # total_n_nodes = len(set([label for edge in edges_list for label in edge]))
    # G = nx.cubical_graph()
    # G = nx.Graph()
    # for edge in db_tree_edges:
    #     G.add_edge(edge.node_from, edge.node_to)
    # nx.draw(G)  # тип по умолчанию spring_layout
    # nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color='b')

    G = nx.Graph()
    # G = nx.cubical_graph()
    edges_list = list(db_tree_edges)[:50]
    # edges_list = [edge for edge in edges_list_init if edge[0] != 'таксон' and edge[0] != 'часть' and edge[0] != 'часть молекулы' and edge[0] != 'монотипия' and edge[0] != 'домашнее животное'
    #               and edge[1] != 'волейбол' and edge[0] != 'часть тела животного' and edge[0] != 'фазовый переход' and edge[0] != 'физический процесс' and edge[1] != 'неформальная экономика']
    # # edges_list = list(set([edge for edge in db_tree_edges if edge[0] == 'терапия' or edge[0] == 'атидепрессант' or edge[0] == 'заболевание'
    #                         or edge[0] == 'медицинская процедура' or edge[0] == 'вирусное заболевание']))[:50]
    edge_labels_50 = {edge: edge_labels[edge] for edge in edges_list}
    with open(EXCEL_EDGES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_1, open(
            EXCEL_NODES_GEPHI_PATH, "w", newline='', encoding='utf-8') as csv_file_2:
        writer_1 = csv.writer(csv_file_1, delimiter=',')
        writer_2 = csv.writer(csv_file_2, delimiter=',')
        writer_1.writerow(['Source', 'Target', 'Label'])
        writer_2.writerow(['Id', 'Label'])
        nodes = {}
        for edge, edge_label in edge_labels_50.items():
            node_id_1 = all_db_labels[edge[0]]
            node_id_2 = all_db_labels[edge[1]]
            writer_1.writerow([node_id_1, node_id_2, edge_label])
            if node_id_1 not in nodes.keys():
                nodes[node_id_1] = edge[0]
            if node_id_2 not in nodes.keys():
                nodes[node_id_2] = edge[1]
        for node_id, node_label in nodes.items():
            writer_2.writerow([node_id, node_label])
    G.add_edges_from(edges_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_color='red', node_size=30)
    nx.draw_networkx_labels(G, pos, font_size=13, horizontalalignment="right")
    nx.draw_networkx_edges(G, pos, edgelist=edges_list, edge_color='black', arrows=False)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_50, font_color='black', font_size=8)
    # nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    plt.show()

    # decide on a layout
    # pos = nx.layout.spring_layout(graph)
    #
    # # Create an interactive plot.
    # # NOTE: you must retain a reference to the object instance!
    # # Otherwise the whole thing will be garbage collected after the initial draw
    # # and you won't be able to move the plot elements around.
    # plot_instance = netgraph.InteractiveGraph(graph, node_positions=pos)
    # ######## drag nodes around #########
    #
    # # To access the new node positions:
    # node_positions = plot_instance.node_positions
    # netgraph.InteractiveGraph(graph,
    #                           node_positions=pos,
    #                           edge_positions=pos)

    # Graph creation:

    # labs = {(1, 2): "1 to 2"}

    # Create an interactive plot.
    # NOTE: you must retain a reference to the object instance!
    # Otherwise the whole thing will be garbage collected after the initial draw
    # and you won't be able to move the plot elements around.

    # pos = nx.layout.spring_layout(graph)
    #
    # ######## drag nodes around #########
    #
    # # To access the new node positions:
    # # plot_instance = netgraph.InteractiveGraph(G, node_shape=node_shapes, node_positions=pos, edge_positions=pos)
    # netgraph.InteractiveGraph(graph,
    #                           node_positions=pos,
    #                           edge_positions=pos,)
    #                           # edge_labels=edge_labels)

    # node_positions = plot_instance.node_positions
    # edge_positions = plot_instance.edge_positions
    gggg =[]



import re

import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from pymystem3 import Mystem

from WikidataEntity import WikidataEntity
from const.Constants import EMPTY_STR
from sql.postgres_crud import get_entity_fields, save_entity, select_qids_query, update_entity, \
    update_main_normal_label, select_main_qid_names_query, update_ref_normal_label, select_ref_qid_names_query

medical_properties = ["P636", "P673", "P486", "P715", "P699", "P780", "P923", "P924", "P2452", "P1748", "P557",
                      "P2892", "P4338", "P3550", "P3841", "P4495", "P5270", "P1694", "P1693", "P1554", "P1550",
                      "P1323", "P696", "P595", "P494", "P1692", "P1461", "P667", "P2275", "P4250", "P2176", "P1995"]

INSTANCE_OF = "P31"
SUBCLASS_OF = "P279"
PART_OF = "P527"
pattern_empty = re.compile("^Q[0-9]+$")  # not translated in Russian
SPARQL_API_ENDPOINT = SPARQLWrapper("https://query.wikidata.org/sparql")
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 10000
# BATCH_SIZE = 10  # TEST
IDS_LIMIT = 50  # no more ids allowed to be fetched at once


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_entities_qids_by_search_term_sparql(search_term, limit, offset):
    # time.sleep(5)
    SPARQL_API_ENDPOINT.setQuery("""
    SELECT ?subject ?subjectLabel WHERE {
      ?subject wdt:%s ?object .

      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "ru" .
      }
    } LIMIT %d OFFSET %d
    """ % (search_term, limit, offset))
    SPARQL_API_ENDPOINT.setReturnFormat(JSON)
    results = SPARQL_API_ENDPOINT.query().convert()
    return results


def get_by_qids(qids_batch):
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'languages': 'ru|en',
    }
    qids_query_str = "|".join(qids_batch)
    params['ids'] = qids_query_str
    response_json = requests.get(WIKIDATA_API_ENDPOINT, params=params).json()
    return response_json


def get_ref_labels_joined(response_claims, relation_type):
    target_ids = []
    if relation_type in response_claims:
        for ref in response_claims[relation_type]:
            if 'mainsnak' in ref and 'datavalue' in ref['mainsnak'] and 'value' in ref['mainsnak']['datavalue']:
                ref_id = ref['mainsnak']['datavalue']['value']['id']
                target_ids.append(ref_id)
    response_json = get_by_qids(target_ids)
    labels = []
    for target_qid in target_ids:
        if 'ru' in response_json['entities'][target_qid]['labels']:
            label = response_json['entities'][target_qid]['labels']['ru']['value']
        elif len(response_json['entities'][target_qid]['labels']) > 0:
            label = response_json['entities'][target_qid]['labels']['en']['value']
        else:
            label = ''
        labels.append(label)
    labels_concat = ', '.join(labels)
    return labels_concat.lower()


def get_aliases_joined(aliases):
    labels = set()
    for alias in aliases:
        labels.add(alias['value'].lower())
    return labels


def get_labels_and_desc_by_qids(filtered_res_dict_corrected):
    qids = list(filtered_res_dict_corrected.keys())
    qids_batched = list(chunks(qids, IDS_LIMIT))
    wiki_entities_list = []
    for qids_batch in qids_batched:
        response_json = get_by_qids(qids_batch)
        for qid in qids_batch:
            new_entity = WikidataEntity(qid)
            label = response_json['entities'][qid]['labels']['ru']['value'].lower()
            new_entity.label = label
            description_opt = response_json['entities'][qid]['descriptions']
            if len(description_opt) > 0:
                if 'ru' in description_opt:
                    description = response_json['entities'][qid]['descriptions']['ru']['value'].lower()
                else:
                    description = response_json['entities'][qid]['descriptions']['en']['value'].lower()
            else:
                description = None
            new_entity.description = description
            aliases_labels = []
            if 'ru' in response_json['entities'][qid]['aliases']:
                aliases_labels = get_aliases_joined(response_json['entities'][qid]['aliases']['ru'])
            new_entity.aliases = aliases_labels
            claims = response_json['entities'][qid]['claims']
            instance_of_labels = get_ref_labels_joined(claims, INSTANCE_OF)
            subclass_of_labels = get_ref_labels_joined(claims, SUBCLASS_OF)
            part_of_labels = get_ref_labels_joined(claims, PART_OF)
            new_entity.instance_of = instance_of_labels
            new_entity.subclass_of = subclass_of_labels
            new_entity.part_of = part_of_labels
            wiki_entities_list.append(new_entity)
    return wiki_entities_list


def fetch_wikidata():
    for medical_property in medical_properties:
        limit = BATCH_SIZE
        offset = 0
        curr_df_len = BATCH_SIZE
        count = 0
        while curr_df_len == BATCH_SIZE:
            results = get_entities_qids_by_search_term_sparql(medical_property, limit, offset)
            results_df = pd.io.json.json_normalize(results['results']['bindings'])
            filtered_res_dict = {k: v for k, v in
                                 dict(zip(results_df['subject.value'].to_list(),
                                          results_df['subjectLabel.value'].to_list())).items()
                                 if not pattern_empty.match(v)}
            existing_q_ids = [x[0] for x in get_entity_fields(select_qids_query)]
            filtered_res_dict_corrected = {}
            count += len(filtered_res_dict_corrected)
            for k, v in filtered_res_dict.items():
                key = k.split('/')[-1]
                if key not in existing_q_ids:
                    filtered_res_dict_corrected[key] = v.lower()
            wikidata_entities = get_labels_and_desc_by_qids(filtered_res_dict_corrected)
            for wikidata_entity in wikidata_entities:
                save_entity(wikidata_entity)
            if curr_df_len != BATCH_SIZE:
                print("%d entities with property %s were saved in DB" % (count, medical_property))
            curr_df_len = len(results_df)
            offset += BATCH_SIZE


def fill_normal_labels_ref(queries):
    select_query, update_query = queries
    qid_name_list = get_entity_fields(select_query)
    grouped_by_q_id = {}
    for i in range(0, len(qid_name_list)):
        if qid_name_list[i][0] not in grouped_by_q_id.keys():
            grouped_by_q_id[qid_name_list[i][0]] = [qid_name_list[i][1]]
        else:
            grouped_by_q_id[qid_name_list[i][0]].append(qid_name_list[i][1])
    mystem = Mystem()
    q_id_normal_label = {}
    for q_id, labels in grouped_by_q_id.items():
        for label in labels:
            if q_id not in q_id_normal_label.keys():
                q_id_normal_label[q_id] = [tuple([label, EMPTY_STR.join(mystem.lemmatize(label)[:-1])])]
            else:
                q_id_normal_label[q_id].append(tuple([label, EMPTY_STR.join(mystem.lemmatize(label)[:-1])]))
    for q_id, label_norm_label_tup_list in q_id_normal_label.items():
        for labels_tup in label_norm_label_tup_list:
            update_entity(update_query, [labels_tup[1], q_id, labels_tup[0]])


def fill_normal_labels_main(queries):
    select_query, update_query = queries
    qid_name_list = get_entity_fields(select_query)
    main_qid_name_dict = dict(zip([tup[0] for tup in qid_name_list], [tup[1] for tup in qid_name_list]))
    mystem = Mystem()
    q_id_normal_label = {}
    for q_id, label in main_qid_name_dict.items():
        q_id_normal_label[q_id] = EMPTY_STR.join(mystem.lemmatize(label)[:-1])
    for q_id, norm_label in q_id_normal_label.items():
        update_entity(update_query, [norm_label, q_id])


def fill_normal_labels():
    fill_normal_labels_main([select_main_qid_names_query, update_main_normal_label])
    fill_normal_labels_ref([select_ref_qid_names_query, update_ref_normal_label])


if __name__ == '__main__':
    fetch_wikidata()
    # fill_normal_labels()

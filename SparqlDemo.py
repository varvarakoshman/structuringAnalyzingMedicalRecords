import re
import time

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import requests
import psycopg2
from psycopg2 import Error

from WikidataEntity import WikidataEntity

# medical_properties = ["P636", "P688", "P689", "P702", "P769", "P780", "P923", "P924", "P925", "P926", "P927", "P928",
#                       "P970", "P1050", "P1057", "P1060",
#                       "P1199", "P1604", "P1605", "P1606", "P1660", "P1677", "P1909", "P1910", "P1911", "P1912", "P1913",
#                       "P1914", "P1915", "P1916", "P1917",
#                       "P1924", "P1995", "P2175", "P2176", "P2239", "P2286", "P2288", "P2289", "P2329", "P2789", "P2841",
#                       "P3094", "P3189", "P3190", "P3205",
#                       "P3261", "P3262", "P3310", "P3354", "P3355", "P3356", "P3357", "P3358", "P3359", "P3464", "P3489",
#                       "P3490", "P3491", "P3493", "P4044",
#                       "P4425", "P4426", "P4545", "P4777", "P4843", "P4954", "P5131", "P5132", "P5133", "P5134", "P5248",
#                       "P5446", "P5572", "P5642", "P2293",
#                       "P1193", "P1603", "P2710", "P2712", "P2717", "P2718", "P2844", "P2854", "P3457", "P3487", "P3488",
#                       "P3492", "P4250", "P2275", "P593",
#                       "P667", "P1402", "P1461", "P1692", "P1748", "P486", "P492", "P493", "P494", "P557", "P563",
#                       "P592", "P594", "P595", "P604", "P637",
#                       "P638", "P639", "P652", "P653", "P663", "P665", "P668", "P672", "P673", "P696", "P698", "P699",
#                       "P704", "P715", "P1055", "P1323",
#                       "P1395", "P1550", "P1554", "P1583", "P1690", "P1691", "P1693", "P1694", "P1925", "P1928", "P1929",
#                       "P1930", "P2074", "P2646", "P2892",
#                       "P2941", "P2944", "P3098", "P3201", "P3291", "P3292", "P3329", "P3331", "P3345", "P3550", "P3637",
#                       "P3640", "P3720", "P3841", "P3885",
#                       "P3945", "P3956", "P3982", "P4058", "P4229", "P4233", "P4235", "P4236", "P4317", "P4338", "P4394",
#                       "P4395", "P4495", "P4670", "P5082",
#                       "P5209", "P5270", "P5329", "P5375", "P5376", "P5415", "P5450", "P5458", "P5468", "P5496", "P5501",
#                       "P5806", "P5843", "P6220"]

# DONE: "P636", "P688" (3 pages), "P673", "P486", "P715", "P699", "P780", "P923", "P924"

medical_properties = ["P2452", "P1748", "P557", "P2892", "P4338", "P3550", "P3841", "P4495", "P5270", "P1694", "P1693",
                      "P1554", "P1550", "P1323", "P696", "P595", "P494", "P1692", "P1461", "P667", "P2275", "P4250",
                      "P2176", "P1995"]

INSTANCE_OF = "P31"
SUBCLASS_OF = "P279"
PART_OF = "P527"
pattern_empty = re.compile("^Q[0-9]+$")  # not translated in Russian
SPARQL_API_ENDPOINT = SPARQLWrapper("https://query.wikidata.org/sparql")
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 10000
# BATCH_SIZE = 10  # TEST
IDS_LIMIT = 50  # no more ids allowed to be fetched at once

# noinspection SqlDialectInspection
insert_main_query = """INSERT INTO wiki_main_entity (q_id, entity_label, entity_description, instance_of, subclass_of, part_of) VALUES (%s, %s, %s, %s, %s, %s)"""
insert_ref_query = """INSERT INTO wiki_ref_entity (entity_label, q_id) VALUES (%s, %s)"""
# noinspection SqlDialectInspection
select_qids_qiery = """select q_id from wiki_main_entity"""


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
    try:
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
        try:
            labels_concat = ', '.join(labels)
        except TypeError as te:
            ggg = []
    except KeyError as ke:
        jjj = []
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
        try:
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
        except KeyError as ke:
            jjj = []
    return wiki_entities_list


def fetch_wikidata():
    for medical_property in medical_properties:
        limit = BATCH_SIZE
        offset = 0
        curr_df_len = BATCH_SIZE
        while curr_df_len == BATCH_SIZE:
            results = get_entities_qids_by_search_term_sparql(medical_property, limit, offset)
            results_df = pd.io.json.json_normalize(results['results']['bindings'])
            filtered_res_dict = {k: v for k, v in
                                 dict(zip(results_df['subject.value'].to_list(),
                                          results_df['subjectLabel.value'].to_list())).items()
                                 if not pattern_empty.match(v)}
            existing_q_ids = [x[0] for x in get_all_existing_qids()]
            filtered_res_dict_corrected = {}
            for k, v in filtered_res_dict.items():
                key = k.split('/')[-1]
                if key not in existing_q_ids:
                    filtered_res_dict_corrected[key] = v.lower()
            wikidata_entities = get_labels_and_desc_by_qids(filtered_res_dict_corrected)
            for wikidata_entity in wikidata_entities:
                save_entity(wikidata_entity)
            print("Entities with property %s were saved in DB", medical_property)
            curr_df_len = len(results_df)
            offset += BATCH_SIZE
    finish = []


def save_entity(wiki_entity):
    try:
        connection = psycopg2.connect(host="localhost",
                                      port="5432",
                                      database="wikidump")
        cursor = connection.cursor()
        cursor.execute(insert_main_query, (
            wiki_entity.qid, wiki_entity.label, wiki_entity.description, wiki_entity.instance_of,
            wiki_entity.subclass_of,
            wiki_entity.part_of))
        for alias in wiki_entity.aliases:
            cursor.execute(insert_ref_query, (alias, wiki_entity.qid))
        connection.commit()
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def get_all_existing_qids():
    q_ids = []
    try:
        connection = psycopg2.connect(host="localhost",
                                      port="5432",
                                      database="wikidump")
        cursor = connection.cursor()
        cursor.execute(select_qids_qiery)
        q_ids = cursor.fetchall()
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return q_ids


if __name__ == '__main__':
    fetch_wikidata()

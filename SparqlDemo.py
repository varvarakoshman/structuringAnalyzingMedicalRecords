import re

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import requests
from wikidata.client import Client

medical_properties = ["P636", "P688", "P689", "P702", "P769", "P780", "P923", "P924", "P925", "P926", "P927", "P928",
                      "P970", "P1050", "P1057", "P1060",
                      "P1199", "P1604", "P1605", "P1606", "P1660", "P1677", "P1909", "P1910", "P1911", "P1912", "P1913",
                      "P1914", "P1915", "P1916", "P1917",
                      "P1924", "P1995", "P2175", "P2176", "P2239", "P2286", "P2288", "P2289", "P2329", "P2789", "P2841",
                      "P3094", "P3189", "P3190", "P3205",
                      "P3261", "P3262", "P3310", "P3354", "P3355", "P3356", "P3357", "P3358", "P3359", "P3464", "P3489",
                      "P3490", "P3491", "P3493", "P4044",
                      "P4425", "P4426", "P4545", "P4777", "P4843", "P4954", "P5131", "P5132", "P5133", "P5134", "P5248",
                      "P5446", "P5572", "P5642", "P2293",
                      "P1193", "P1603", "P2710", "P2712", "P2717", "P2718", "P2844", "P2854", "P3457", "P3487", "P3488",
                      "P3492", "P4250", "P2275", "P593",
                      "P667", "P1402", "P1461", "P1692", "P1748", "P486", "P492", "P493", "P494", "P557", "P563",
                      "P592", "P594", "P595", "P604", "P637",
                      "P638", "P639", "P652", "P653", "P663", "P665", "P668", "P672", "P673", "P696", "P698", "P699",
                      "P704", "P715", "P1055", "P1323",
                      "P1395", "P1550", "P1554", "P1583", "P1690", "P1691", "P1693", "P1694", "P1925", "P1928", "P1929",
                      "P1930", "P2074", "P2646", "P2892",
                      "P2941", "P2944", "P3098", "P3201", "P3291", "P3292", "P3329", "P3331", "P3345", "P3550", "P3637",
                      "P3640", "P3720", "P3841", "P3885",
                      "P3945", "P3956", "P3982", "P4058", "P4229", "P4233", "P4235", "P4236", "P4317", "P4338", "P4394",
                      "P4395", "P4495", "P4670", "P5082",
                      "P5209", "P5270", "P5329", "P5375", "P5376", "P5415", "P5450", "P5458", "P5468", "P5496", "P5501",
                      "P5806", "P5843", "P6220"]
INSTANCE_OF = "P31"
SUBCLASS_OF = "P279"
pattern_empty = re.compile("^Q[0-9]+$")  # not translated in Russian
SPARQL_API_ENDPOINT = SPARQLWrapper("https://query.wikidata.org/sparql")
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 10000
IDS_LIMIT = 50  # no more ids allowed to be fetched at once


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_entities_qids_by_search_term_sparql(search_term, limit, offset):
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


# Ex: 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q7189713&format=xml&languages=ru'
def get_labels_and_desc_by_qids(qids):
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'languages': 'ru|en',
        # 'ids': 'Q7189713'
    }
    qids_batched = list(chunks(qids, IDS_LIMIT))
    qid_info_dict = {}
    for qids_batch in qids_batched:
        qids_query_str = "|".join(qids_batch)
        params['ids'] = qids_query_str
        response_json = requests.get(WIKIDATA_API_ENDPOINT, params=params).json()
        try:
            for qid in qids_batch:
                label = response_json['entities'][qid]['labels']['ru']['value']
                description_opt = response_json['entities'][qid]['descriptions']
                if len(description_opt) > 0:
                    if len(description_opt['ru']) > 0:
                        description = response_json['entities'][qid]['descriptions']['ru']['value']
                    else:
                        description = response_json['entities'][qid]['descriptions']['en']['value']
                    target_tuple = tuple([label, description])
                else:
                    target_tuple = tuple([label, None])
                qid_info_dict[qid] = target_tuple
        except KeyError as ke:
            jjj = []
    return qid_info_dict


def fetch_wikidata():
    client = Client()  # doctest: +SKIP
    entity = client.get('Q79785', load=True)
    all_data = {}
    count = 0
    for medical_property in medical_properties:
        curr_prep_dict = {}
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
            curr_prep_dict.update(filtered_res_dict)
            curr_df_len = len(results_df)
            offset += BATCH_SIZE
            count += 1
            if count == 4:
                break
        all_data.update(curr_prep_dict)
        if count == 4:
            break
    corrected_data = {k.split('/')[-1]: v.lower() for k, v in all_data.items()}
    qid_info_dict = get_labels_and_desc_by_qids(list(corrected_data.keys()))
    hhh = {}


def process_wikidata(all_dictionaries):
    corrected = {k.split('/')[-1]: v.lower() for k, v in all_dictionaries.items()}

if __name__ == '__main__':
    fetch_wikidata()

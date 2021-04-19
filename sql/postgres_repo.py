import psycopg2
from psycopg2 import Error

# noinspection SqlDialectInspection
insert_main_query = """INSERT INTO wiki_main_entity (q_id, entity_label, entity_description, instance_of, subclass_of, part_of) VALUES (%s, %s, %s, %s, %s, %s)"""
insert_ref_query = """INSERT INTO wiki_ref_entity (entity_label, q_id) VALUES (%s, %s)"""
# noinspection SqlDialectInspection
select_qids_query = """select q_id from wiki_main_entity"""

# noinspection SqlDialectInspection
select_main_qid_names_query = """select q_id, entity_label from wiki_main_entity"""
# noinspection SqlDialectInspection
select_ref_qid_names_query = """select q_id, entity_label from wiki_ref_entity"""
update_main_normal_label = """UPDATE wiki_main_entity SET entity_label_normal = (%s) WHERE q_id = (%s)"""
update_ref_normal_label = """UPDATE wiki_ref_entity SET entity_label_normal = (%s) WHERE q_id = (%s) and entity_label = (%s)"""
# noinspection SqlDialectInspection
select_all_ref = """SELECT q_id, entity_label, entity_label_normal from wiki_ref_entity"""
# noinspection SqlDialectInspection
select_all_main = """SELECT q_id, entity_label, entity_label_normal, instance_of, subclass_of, part_of from wiki_main_entity"""


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


def get_entity_fields(query):
    result = []
    try:
        connection = psycopg2.connect(host="localhost",
                                      port="5432",
                                      database="wikidump")
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return result


def update_entity(query, params):
    result = []
    try:
        connection = psycopg2.connect(host="localhost",
                                      port="5432",
                                      database="wikidump")
        cursor = connection.cursor()
        cursor.execute(query, params)
        connection.commit()
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return result

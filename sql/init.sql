-- create a table for storing wiki-data entity main information
CREATE TABLE wiki_main_entity(
   q_id text PRIMARY KEY,
   entity_label text,
   entity_description text,
   entity_label_normal text,
   instance_of text,
   subclass_of text,
   part_of text
);

-- create a table for storing aliases for wiki-data entities' names
CREATE TABLE wiki_ref_entity(
   id serial PRIMARY KEY,
   entity_label text,
   entity_label_normal text,
   q_id text,
   CONSTRAINT fk_main_entity
      FOREIGN KEY(q_id)
	  REFERENCES wiki_main_entity(q_id)
);

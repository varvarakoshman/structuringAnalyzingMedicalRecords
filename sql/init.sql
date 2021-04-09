CREATE TABLE wiki_main_entity(
   q_id text PRIMARY KEY,
   entity_label text,
   entity_description text,
   instance_of text,
   subclass_of text,
   part_of text
);

CREATE TABLE wiki_ref_entity(
   id serial PRIMARY KEY,
   entity_label text,
   q_id text,
   CONSTRAINT fk_main_entity
      FOREIGN KEY(q_id)
	  REFERENCES wiki_main_entity(q_id)
);

-- drop table wiki_ref_entity;
-- drop table wiki_main_entity;
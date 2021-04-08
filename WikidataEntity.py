class WikidataEntity:
    def __init__(self, qid, name=None, aliases=None, instance_of=None, subclass_of=None, part_of=None):
        self.qid = qid
        self.name = name
        self.aliases = aliases
        self.instance_of = instance_of
        self.subclass_of = subclass_of
        self.part_of = part_of
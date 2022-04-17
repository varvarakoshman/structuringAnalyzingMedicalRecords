class WikidataEntity:
    def __init__(self, qid, label=None, description=None, aliases=None, instance_of=None, subclass_of=None, part_of=None):
        self.qid = qid
        self.label = label
        self.description = description
        self.instance_of = instance_of
        self.subclass_of = subclass_of
        self.part_of = part_of
        self.aliases = aliases
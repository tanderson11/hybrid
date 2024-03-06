from reactionmodel.parser import ConfigParser

from .hybrid import FixedThresholdPartitioner

SCHEMES_BY_NAME = {
    'FixedThresholdPartitioner': FixedThresholdPartitioner
}

def partition_scheme_from_dictionary(dictionary):
    scheme_name = dictionary.pop('name')
    scheme_class = SCHEMES_BY_NAME[scheme_name]
    return scheme_class(**dictionary)

class HybridConfigParser(ConfigParser):
    @classmethod
    def parse_dictionary(cls, dictionary):
        dictionary = dictionary.copy()
        partition = dictionary.pop('partition')
        partition = partition_scheme_from_dictionary(partition)
        dictionary['partition_function'] = partition.partition_function
        return dictionary

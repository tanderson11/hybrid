import unittest
import os
import numpy as np

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta, EndpointTest

decaying_isomerization_tests = discover_tests(os.path.dirname(__file__), './decaying_isomerization', include_check=True)
schlogl_tests = discover_tests(os.path.dirname(__file__), './schlogl')

class DecayingIsomerizationCollection(FilesystemTestMeta):
    test_collection = decaying_isomerization_tests

class TestDecayingIsomerization(EndpointTest, metaclass=DecayingIsomerizationCollection):
    def end_routine(self, result):
        y_end = super().end_routine(result)
        if 'model_ss' not in self.test_name:
            return y_end

        p = self.specification.parameters
        resampled = {}
        resampled['S3'] = y_end['S3']
        resampled['S2'] = np.random.binomial(y_end['S12'], p['c1']/(p['c1'] + p['c2']))
        resampled['S1'] = y_end['S12'] - resampled['S2']

        return resampled

class SchloglCollection(FilesystemTestMeta):
    test_collection = schlogl_tests

class TestSchlogl(EndpointTest, metaclass=SchloglCollection):
    pass

if __name__ == '__main__':
    unittest.main()
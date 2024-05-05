import unittest
import os
import numpy as np
import pandas as pd
import pathlib
from typing import NamedTuple

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta, EndpointTest

decaying_isomerization_tests = discover_tests(os.path.dirname(__file__), './decaying_isomerization', include_check=True)

class TestDecayingIsomerizationMeta(FilesystemTestMeta):
    test_collection = decaying_isomerization_tests

class TestDecayingIsomerization(EndpointTest, metaclass=TestDecayingIsomerizationMeta):
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

if __name__ == '__main__':
    unittest.main()
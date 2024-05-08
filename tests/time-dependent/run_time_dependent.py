import unittest
import os
import numpy as np
from numba import jit

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta
from tests.sbml.sbml_tests import TestWithMeanChecks

time_dependent_tests = discover_tests(os.path.dirname(__file__), '*', include_check=True)

class TimeDependentCollection(FilesystemTestMeta):
    test_collection = time_dependent_tests

class TestTimeDependent(TestWithMeanChecks, metaclass=TimeDependentCollection):
    do_yscores = False
    def apply_overrides(self, specification):
        specification = super().apply_overrides(specification)
        specification.t.t_eval = np.linspace(0.0, 5.0, 51)

        if 'simple' in self.test_name:
            r = specification.model.all_reactions[0]
            @jit(nopython=True)
            def k_of_t(t):
                return np.array(t)
            self.reaction_to_k = {r: k_of_t}
        return specification

if __name__ == '__main__':
    unittest.main()
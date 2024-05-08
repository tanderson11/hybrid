import unittest
import os
import numpy as np
from numba import jit

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta, MeanTest

time_dependent_tests = discover_tests(os.path.dirname(__file__), '*', include_check=False)

class TimeDependentCollection(FilesystemTestMeta):
    test_collection = time_dependent_tests

class TestTimeDependent(MeanTest, metaclass=TimeDependentCollection):
    t_eval = np.linspace(0, 5.0, 51)
    do_yscores = False
    def apply_overrides(self, specification):
        specification = super().apply_overrides(specification)

        if 'simple' in self.test_name:
            r = specification.model.all_reactions[0]
            @jit(nopython=True)
            def k_of_t(t):
                return np.array(t)
            self.reaction_to_k = {r: k_of_t}
        return specification

if __name__ == '__main__':
    unittest.main()
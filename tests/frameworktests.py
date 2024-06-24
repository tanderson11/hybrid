import unittest

import numpy as np

from hybrid.hybrid import NThresholdPartitioner

class PartitionTests(unittest.TestCase):
    def test_divide_by_stoichiometry(self):
        N = np.array([[0.0, 1.0, -10.0],[2.0, -1.0, 1.0]])
        kinetic_order_matrix = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 1.0]])

        p = NThresholdPartitioner(100.0)

        y = np.array([100.0, 100.0])
        partitioned = p.partition_function(N, kinetic_order_matrix, y, None)
        self.assertFalse(partitioned.deterministic.any())

        y = np.array([101.0, 201.0])
        #import pudb; pudb.set_trace()

        partitioned = p.partition_function(N, kinetic_order_matrix, y, None)

        self.assertTrue((partitioned.deterministic == np.array([True, True, False])).all())
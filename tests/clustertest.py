import sys
import unittest
import os

import unit_tests

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: 'python clustertest i' where i == ith test to run from SBML suite, arbitrarily sorted."
    i = int(sys.argv[1])
    suite = unittest.TestLoader().loadTestsFromTestCase(unit_tests.TestSBML)
    print(suite)
    for j, test in enumerate(sorted(suite)):
        if i != j:
            continue
        test.run()
        # move artifact to our own destination
        unit_tests.sbml_root


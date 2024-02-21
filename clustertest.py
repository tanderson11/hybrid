import sys
import unittest
import os

import unit_tests

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage: 'python clustertest path/to/save i' where i == ith test to run from SBML suite, arbitrarily sorted."
    #save_path = os.path(sys.argv[1])
    i = int(sys.argv[2])
    suite = unittest.TestLoader().loadTestsFromTestCase(unit_tests.TestSBML)
    print(suite)
    for j, test in enumerate(suite):
        if i != j:
            continue
        test.run()
        # move artifact to our own destination
        unit_tests.sbml_root


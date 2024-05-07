import sys
import unittest
import os

import sbml_tests

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: 'python clustertest i' where i == ith test to run from SBML suite, arbitrarily sorted."
    i = int(sys.argv[1])
    print("I", i)
    suite = unittest.TestLoader().loadTestsFromTestCase(sbml_tests.TestSBML)
    suite_list = list(suite)
    runner = unittest.TextTestRunner()
    print(suite_list[i])
    runner.run(suite_list[i])


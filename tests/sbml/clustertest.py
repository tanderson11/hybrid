import unittest
import argparse

from tests.suites import suites

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=suites.keys(), help="name of test suite to run")
    parser.add_argument('test', type=int, help="index of test to run from suite, arbitrarily sorted")
    args = parser.parse_args()
    i = args.test
    print("I", i)
    suite = unittest.TestLoader().loadTestsFromTestCase(suites[args.suite])
    suite_list = list(suite)
    runner = unittest.TextTestRunner()
    print(suite_list[i])
    runner.run(suite_list[i])


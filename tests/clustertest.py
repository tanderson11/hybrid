import unittest
import argparse
import re

from tests.suites import suites

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=suites.keys(), help="name of test suite to run")
    parser.add_argument('test', type=int, help="index of test to run from suite, arbitrarily sorted")
    parser.add_argument('-f', '--filterregex', type=str, default=None, help='regex pattern used to filter out non-matching tests before indexing')

    args = parser.parse_args()
    i = args.test
    print("I", i)
    suite = unittest.TestLoader().loadTestsFromTestCase(suites[args.suite])
    if args.filterregex is not None:
        pattern = re.compile(args.filterregex)
        tests = [t for t in suite if re.match(pattern, t.id())]
    suite_list = list(suite)
    runner = unittest.TextTestRunner()
    print(suite_list[i])
    runner.run(suite_list[i])


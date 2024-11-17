import unittest
import argparse
import re

from tests.suites import suites

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=suites.keys(), help="name of test suite to run")
    parser.add_argument('test', type=int, help="index of test to run from suite, arbitrarily sorted")
    parser.add_argument('-f', '--filterregex', type=str, default=None, help='regex pattern used to filter out non-matching tests before indexing')
    parser.add_argument('-k', '--trials', type=int, default=10000, help='number of trials to run')
    parser.add_argument('-i', '--runnerid', type=str, default=0, help='id of runner for output path')

    args = parser.parse_args()
    i = args.test
    print("I", i)
    suite = unittest.TestLoader().loadTestsFromTestCase(suites[args.suite])
    print([t.id() for t in suite])
    if args.filterregex is not None:
        pattern = re.compile(args.filterregex)
        tests = [t for t in suite if re.match(pattern, t.id())]
    else:
        tests = list(suite)
    runner = unittest.TextTestRunner()
    tests[i].set_n(args.trials)
    if args.runnerid != 0:
        tests[i].set_runner_id(args.runnerid)
    print(tests[i])
    runner.run(tests[i])
import unittest
import subprocess
import shlex
import argparse
import os
import re

from tests.suites import suites

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=suites.keys(), help="name of test suite to run")
    parser.add_argument('-n', '--maxjobs', type=int, default=64, help="maximum number of jobs to run at once")
    parser.add_argument('-t', '--time', type=int, default=12, help="number of hours allocated for each job")
    parser.add_argument('-m', '--memory', type=int, default=16, help='gb of memory allocated for each job')
    parser.add_argument('-f', '--filterregex', type=str, default='.*', help='regex pattern applied to test name, test only executed if name matches pattern')
    args = parser.parse_args()

    tests = list(unittest.TestLoader().loadTestsFromTestCase(suites[args.suite]))
    if args.filterregex is not None:
        pattern = re.compile(args.filterregex)
        tests = [t for t in tests if re.match(pattern, t.id())]
    ntests = len(tests)

    env = os.environ.copy()
    env['TEST_SUITE'] = args.suite
    env['TEST_FILTER'] = args.filterregex
    subprocess.run(
        shlex.split(f'sbatch -o {args.suite}-%A_%a.out --array=0-{ntests-1}%{args.maxjobs} --job-name={args.suite} --time={args.time}:0:0 --mem={args.memory}G submit.sh'),
        env=env
    )

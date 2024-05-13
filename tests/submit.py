import unittest
import subprocess
import shlex
import argparse
import os

from tests.suites import suites

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=suites.keys(), help="name of test suite to run")
    parser.add_argument('-n', '--maxjobs', type=int, default=16, help="index of test to run from suite, arbitrarily sorted")
    parser.add_argument('-t', '--time', type=int, default=8, help="number of hours allocated for each job")
    parser.add_argument('-m', '--memory', type=int, default=16, help='gb of memory allocated for each job')
    args = parser.parse_args()

    ntests = len(list(unittest.TestLoader().loadTestsFromTestCase(suites[args.suite])))
    env = os.environ.copy()
    env['TEST_SUITE'] = args.suite
    subprocess.run(
        shlex.split(f'sbatch -o {args.suite}-%A_%a.out --array=0-{ntests-1}%{args.maxjobs} --job-name={args.suite} --time={args.time}:0:0 --mem={args.memory}G submit.sh'),
        env=env
    )

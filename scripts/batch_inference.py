#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

#script = 'bash {0!s}/run.sh julia -C "generic" -J /project/image.so ' + \
#         '/project/scripts/experiments/attention/attention.jl'
script = 'bash {0!s}/run.sh julia -C "generic" ' + \
         '/project/scripts/experiments/exp1/run.jl'

def att_tasks(args):
    tasks = [(t, ) for t in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scenes', type = int, default = 30,
                        help = 'number of scenes')
    parser.add_argument('--chains', type = int, default = 1,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 30,
                        help = 'job duration (min)')



    args = parser.parse_args()

    n = args.scenes * args.chains
    tasks, kwargs, extras = att_tasks(args)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '4',
        'mem-per-cpu' : '2GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'gpu',
        'gres' : 'gpu:1',
        'requeue' : None,
        'job-name' : 'rooms',
        'output' : os.path.join(os.getcwd(), 'output/slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=n)))
    batch.run(n = n, check_submission = False)

if __name__ == '__main__':
    main()

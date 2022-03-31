#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

datasets = ['vss_pilot']

script = 'bash {0!s}/run.sh julia ' + \
         '/project/scripts/stimuli/render_rooms.jl'

def att_tasks(args):
    tasks = [(args.dataset, t) for t in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', type = str, default = 'vss_pilot',
                        choices = datasets,
                        help = 'Dataset to render')
    parser.add_argument('--scenes', type = int, default = 60,
                        help = 'number of scenes')
    parser.add_argument('--duration', type = int, default = 10,
                        help = 'job duration (min)')
    args = parser.parse_args()

    n = args.scenes
    tasks, kwargs, extras = att_tasks(args)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '4',
        'mem-per-cpu' : '1GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'scavenge',
        'requeue' : None,
        'job-name' : 'rooms',
        'output' : os.path.join(os.getcwd(), 'env.d/spaths/slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    job_script = batch.job_file(chunk = n, tmp_dir = '/spaths/slurm')
    job_script = '\n'.join(job_script)
    print("Template Job:")
    print(job_script)
    # batch.run(n = njobs, script = job_script)

if __name__ == '__main__':
    main()

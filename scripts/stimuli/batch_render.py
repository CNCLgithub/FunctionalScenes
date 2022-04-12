#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

datasets = ['vss_pilot_11f_32x32']

script = 'bash {0!s}/env.d/run.sh julia ' + \
         '/project/scripts/stimuli/render_rooms.jl'

def att_tasks(args):
    tasks = [(args.dataset, t) for t in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', type = str, default = datasets[0],
                        choices = datasets,
                        help = 'Dataset to render')
    parser.add_argument('--scenes', type = int, default = 60,
                        help = 'number of scenes')
    parser.add_argument('--duration', type = int, default = 10,
                        help = 'job duration (min)')
    args = parser.parse_args()

    n = args.scenes
    tasks, kwargs, extras = att_tasks(args)

    slurm_out = os.path.join(os.getcwd(), 'env.d/spaths/slurm')
    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '4',
        'mem-per-cpu' : '1GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'scavenge',
        'requeue' : None,
        'job-name' : 'rooms',
        'chdir' : os.getcwd(),
        'output' : os.path.join(slurm_out, '%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    job_script = batch.job_file(chunk = n, tmp_dir = slurm_out)
    job_script = '\n'.join(job_script)
    print("Template Job:")
    print(job_script)
    batch.run(n = n, script = job_script)

if __name__ == '__main__':
    main()

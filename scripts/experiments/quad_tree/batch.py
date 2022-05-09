#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
import pandas as pd
from slurmpy import sbatch

script = 'bash {0!s}/env.d/run.sh julia ' + \
        '/project/scripts/experiments/quad_tree/run.jl'
# script = 'bash {0!s}/run.sh julia -C "generic" ' + \
#          '/project/scripts/experiments/exp1/run.jl'

def att_tasks(args, df):
    tasks = []
    for (ri, r) in df.iterrows():
        # base scene
        tasks.append((r['id'], r['door'], args.chains, 'A'))
        # shifted scene
        tasks.append((f"--move {r.move}", f"--furniture {r.furniture}",
                      r['id'], r['door'], args.chains, 'A'))
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scenes', type = str,
                        default = 'vss_pilot_11f_32x32_restricted',
                        help = 'number of scenes') ,
    parser.add_argument('--chains', type = int, default = 5,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 60,
                        help = 'job duration (min)')



    args = parser.parse_args()
    df_path = f"/spaths/datasets/{args.scenes}/scenes.csv"
    df = pd.read_csv(df_path)
    print(df)

    tasks, kwargs, extras = att_tasks(args, df)
    # tasks = tasks[:1]
    print(len(tasks))

    interpreter = '#!/bin/bash'
    slurm_out = os.path.join(os.getcwd(), 'env.d/spaths/slurm')
    resources = {
        'cpus-per-task' : '4',
        'mem-per-cpu' : '2GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'scavenge',
        'gres' : 'gpu:1',
        'requeue' : None,
        'job-name' : 'rooms',
        'chdir' : os.getcwd(),
        'output' : f"{slurm_out}/%A_%a.out"
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    job_script = batch.job_file(chunk = len(tasks), tmp_dir = slurm_out)
    job_script = '\n'.join(job_script)
    print("Template Job:")
    print(job_script)
    batch.run(n = len(tasks), script = job_script)


if __name__ == '__main__':
    main()

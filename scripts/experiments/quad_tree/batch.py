#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
import pandas as pd
from slurmpy import sbatch

script = 'bash {0!s}/env.d/run.sh ' + \
        '/project/scripts/experiments/quad_tree/run.sh'

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
                        default = 'ccn_2023_exp',
                        help = 'number of scenes') ,
    parser.add_argument('--chains', type = int, default = 5,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 60,
                        help = 'job duration (min)')



    args = parser.parse_args()
    df_path = f"/spaths/datasets/{args.scenes}/scenes.csv"
    df = pd.read_csv(df_path)

    tasks, kwargs, extras = att_tasks(args, df)
    # run one job first to test and profile
    # tasks = tasks[:args.chains]

    interpreter = '#!/bin/bash'
    slurm_out = os.path.join(os.getcwd(), 'env.d/spaths/slurm')
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '8GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'psych_scavenge',
        'gres' : 'gpu:1',
        'requeue' : None,
        'job-name' : 'rooms-ccn',
        'chdir' : os.getcwd(),
        'output' : f"{slurm_out}/%A_%a.out",
        'exclude' : 'r811u30n01' # ran into CUDA error
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

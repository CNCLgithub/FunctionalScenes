#!/usr/bin/env python3

'''

Scrapes data from database, then parses it to anonymize and make good for analysis

'''

from __future__ import division, print_function
import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table


# Mostly from http://psiturk.readthedocs.io/en/latest/retrieving.html
def read_db(db_path, table_name, codeversions, mode):
    data_column_name = "datastring"
    engine = create_engine("sqlite:///" + db_path)
    metadata = MetaData()
    metadata.bind = engine
    table = Table(table_name, metadata, autoload=True)
    s = table.select()
    rows = s.execute()

    rawdata = []
    statuses =  [3, 4, 5, 7]
    for row in rows:
        if (row['status'] in statuses and
            row['mode'] == mode and
            row['codeversion'] in codeversions):
            str_data = row[data_column_name]
            proc_data = json.loads(str_data)
            rawdata.append(proc_data)

    conddict = {}
    for part in rawdata:
        uniqueid = part['workerId'] + ':' + part['assignmentId']
        conddict[uniqueid] = part['condition']
    data = [part['data'] for part in rawdata]

    for part in data:
        for record in part:
            record['trialdata']['uniqueid'] = record['uniqueid']
            record['trialdata']['condition'] = conddict[record['uniqueid']]

    trialdata = pd.DataFrame([record['trialdata'] for part in data for
                              record in part if
                              ('IsInstruction' in record['trialdata'] and
                               not record['trialdata']['IsInstruction'])])

    qdat = []
    for part in rawdata:
        thispart = part['questiondata']
        thispart['uniqueid'] = part['workerId'] + ':' + part['assignmentId']
        qdat.append(thispart)
    questiondata = pd.DataFrame(qdat)

    return trialdata, questiondata

def parse_row(tname):

    # scene data
    tpath, _ = os.path.splitext(tname)
    splits = tpath.split('_')
    if len(splits) == 2:
        base = True
        scene, _ = splits
        furniture = None
        move = None
    else:
        base = False
        scene, furniture, move = splits

    new_row = {
        'base' : base,
        'scene' : scene,
        'furniture' : furniture,
        'move' : move}
    return new_row


def main():

    parser = argparse.ArgumentParser(description = "Parses MOT Exp:1 data",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--exp", type = str, help = "Path to trial dataset",
                        default = '2e_1p_30s')
    parser.add_argument("--table_name", type = str, default = "2e_1p_30s",
                        help = 'Table name')
    parser.add_argument("--exp_flag", type = str, nargs ='+', default = ["2.0"],
                        help = 'Experiment version flag')
    parser.add_argument("--mode", type = str, default = "debug",
                        choices = ['debug', 'sandbox', 'live'],
                        help = 'Experiment mode')
    parser.add_argument("--trialsbyp", type = int, default = 120,
                        help = 'Number of trials expected per subject')
    parser.add_argument("--trialdata", type = str,
                        default = '/experiments/pilot/parsed_trials.csv',
                        help = 'Filename to dump parsed trial data')
    parser.add_argument("--questiondata", type = str,
                        default = '/experiments/pilot/parsed_questions.csv',
                        help = 'Filename to dump parsed trial data')
    args = parser.parse_args()

    exp_src = os.path.join('/experiments', args.exp)
    os.path.isdir(exp_src) or os.mkdir(args.exp)

    db = os.path.join(exp_src, 'participants.db')
    trs, qs = read_db(db, args.table_name,
                      args.exp_flag, args.mode)


    cl_qs = qs.rename(index=str, columns={'uniqueid': 'WID'})

    trs = trs.dropna()
    trs = trs.rename(index=str,
                     columns={'ReactionTime':'RT',
                              'uniqueid':'WID'})
    # row_data = pd.concat(trs.apply(parse_row, axis=1).tolist())
    # trs = trs[['TrialName', 'WID', 'RT', 'Response', 'condition', 'TrialOrder']]
    # trs = trs.merge(row_data, on = ['TrialName', 'WID'])

    trs = trs.merge(trs.TrialName.apply(
        lambda s: pd.Series(parse_row(s))),
                    left_index=True, right_index=True)

    # Make sure we have required responses per participant
    trialsbyp = trs.groupby('WID').aggregate({"TrialOrder" : lambda x : max(x) + 1})
    print(trialsbyp)
    good_wids = trialsbyp[trialsbyp.TrialOrder  == args.trialsbyp].index
    trs = trs[trs.WID.isin(good_wids)]


    # Assign random identifiers to each participant
    wid_translate = {}
    for i, wid in enumerate(good_wids):
        wid_translate[wid] = i

    trs["ID"] = trs.WID.apply(lambda x: wid_translate[x])


    out = os.path.join(exp_src, 'parsed_trials.csv')
    trs.to_csv(out, index=False)

    cl_qs = cl_qs[cl_qs.WID.isin(good_wids)]
    cl_qs["ID"] = cl_qs.WID.apply(lambda x: wid_translate[x])

    out = os.path.join(exp_src, 'parsed_questions.csv')
    cl_qs[["ID", "instructionloops", "comments"]].to_csv(out, index=False)

if __name__ == '__main__':
    main()

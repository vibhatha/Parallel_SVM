#!/bin/python

import sys
import numpy as np

def loadfile(infile):
    data = []
    with open(infile, 'r') as inf:
        data = inf.readlines()
    return data

def savefile(outfile, dataset, ids):
    with open(outfile,'w') as of:
        for id in ids:
            of.write(dataset[id])

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print('usage: datasplit.py <infile> <ratio> [no fixseed]')
        sys.exit(-1)
    
    dataset = loadfile(sys.argv[1])
    total = len(dataset)
    infile = sys.argv[1]
    if len(dataset) <= 0:
        print('load empty file, quit')
        sys.exit(-1)
    
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    if ratio < 0:
        print('ratio should be in (0,1), reset it to 0.2')
        ratio = 0.2
        testcnt = int(total*ratio)
    elif ratio > 1:
        print('set absolute test number as %d'%ratio)
        testcnt = int(ratio)
    else:
        testcnt = int(total*ratio)
    
    if len(sys.argv) <= 3:
        print('fix random seed to 123')
        np.random.seed(seed= 123)

    id = np.arange(total)
    perm = np.random.permutation(id)
    
    test = perm[:testcnt]
    train = perm[testcnt:]
    savefile('train-' + infile, dataset, train)
    savefile('test-' + infile, dataset, test)
    

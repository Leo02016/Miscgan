import argparse
import numpy as np
from scipy.io import savemat
import os
import networkx as nx

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_dir', dest='output_dir', default='./output_dir_bc', help='results are saved here')
    args = parser.parse_args()
    data = dict()

    data['GAE'] = np.load('{}/GAE_network.npy'.format(args.output_dir))
    original = np.load('{}/org_network.npy'.format(args.output_dir))
    if os.path.exists('{}/BA.npy'.format(args.output_dir)) and os.path.exists('{}/ER.npy'.format(args.output_dir)):
        data['BA'] = np.load('{}/BA.npy'.format(args.output_dir))
        data['ER'] = np.load('{}/ER.npy'.format(args.output_dir))
    else:
        n = original.shape[0]
        m = int(np.sum(original))
        random_G = nx.gnm_random_graph(n, m)
        random_bara_G = nx.generators.random_graphs.barabasi_albert_graph(n, 500)
        data['ER'] = nx.to_numpy_matrix(random_G)
        data['BA'] = nx.to_numpy_matrix(random_bara_G)
    data['original'] = original
    netgan = np.load('{}/netgan_network.npy'.format(args.output_dir))[1]
    n = original.shape[0]
    if netgan.shape[0] < n:
        Net = np.zeros((n, n))
        Net[:netgan.shape[0], :netgan.shape[1]] = netgan
        netgan = Net
    data['netgan'] = netgan
    data['music'] = np.load('{}/output_network.npy'.format(args.output_dir))

    savemat('{}/all.mat'.format(args.output_dir), data)

main()
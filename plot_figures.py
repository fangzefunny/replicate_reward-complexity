'''
Info: Replication the result of a published paper 

Paper: https://gershmanlab.com/pubs/LaiGershman21.pdf
Original repo: http://github.com/lucylai96/plm/

@Zeming 
'''
import os 
import argparse
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 

import scipy.stats

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--fig_idx', '-f', help='figure idx', default='fig2')
args = parser.parse_args()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SEC0: BASIC FUNCTION   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def normfit(data, confidence=0.95):
    ''' equivalent Matlab normfit function 

    adapted from:
    https://stackoverflow.com/questions/56440249/equivalent-python-code-of-normfit-in-matlab
    '''    
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    var = np.var(data, ddof=1)
    varCI_upper = var * (n - 1) / (scipy.stats.chi2.ppf((1-confidence) / 2, n - 1))
    varCI_lower = var * (n - 1) / (scipy.stats.chi2.ppf(1-(1-confidence) / 2, n - 1))
    sigma = np.sqrt(var)
    sigmaCI_lower = np.sqrt(varCI_lower)
    sigmaCI_upper = np.sqrt(varCI_upper)

    return m, sigma, [m - h, m + h], [sigmaCI_lower, sigmaCI_upper]

def Fig2():

    conds = [ 'HC', 'SZ']
    colors = [ 'b', 'r' ]
    setsize = [ 2, 3, 4, 5, 6]

    with open( f'{path}/data/results_collins_data_14.pkl', 'rb')as handle:
        outcome = pickle.load( handle) 
    
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 15})
    # show the rate distortion curve for each set size
    for i, sz in enumerate( setsize):

        # choose which subplot 
        plt.subplot( 2, 3, i+1)

        # plot the theoretical curve
        plt.plot( outcome[ 'Rate_theo'][ :, i], outcome[ 'Val_theo'][ :, i], 
                color='k', linewidth=3)

        # enumerate the human data 
        for j, _ in enumerate( conds):
            plt.scatter( outcome[ 'Rate_data'][ :, i, j], outcome[ 'Val_data'][ :,i, j], 
                color=colors[j], s=55)
        
        if i == 0:
            plt.legend( ['theory']+conds, loc=4)
        if i > 2:
            plt.xlabel( 'Pi complexity')
        if (i==0) and (i==3):
            plt.ylabel( 'reward')
        plt.ylim([.2, 1.1])
        plt.xlim([ 0, 1.2])
        plt.title( f'Set size={sz}')

    # plot the Rate over set size
    plt.subplot( 2, 3, 6)
    mus  = np.zeros([len(setsize), 2]) 
    stds = np.zeros([len(setsize), 2]) 
    for j, _ in enumerate(conds):
        for i, _ in enumerate( setsize):
            a = outcome[ 'Rate_data'][ :, i, j]
            mu, _, mu_interval,_ = normfit( a[~np.isnan(a)])
            mus[i, j]  = mu
            stds[i, j] = np.diff(mu_interval)[0]/2
        plt.plot( setsize, mus[:,j], 'o-', color=colors[j], linewidth=3)
        plt.errorbar( setsize, mus[:,j], stds[:,j], color=colors[j])
        plt.title( 'Pi comp. vs set size')
        plt.xlabel( 'set size')
        plt.ylabel( 'Pi complexity')
    plt.ylim([.3, .7])    
    plt.legend( conds)

    try:
        plt.savefig( f'{path}/figures/Gershman21_fig2')
    except:
        os.mkdir('figures')
        plt.savefig( f'{path}/figures/Gershman21_fig2')

def plot_figures( fig_idx):

    if fig_idx=='fig2':
        Fig2()
    
    
if __name__ == '__main__':

    # show figure
    plot_figures(args.fig_idx)
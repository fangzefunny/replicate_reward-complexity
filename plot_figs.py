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

from preprocess_and_fits import analyze, simluate_data, normfit

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--fig_idx', '-f', help='figure idx', default='fig5')
args = parser.parse_args()


def plot_figures( fig_idx):

    conds = [ 'HC', 'SZ']
    colors = [ 'b', 'r' ]
    setsize = [ 2, 3, 4, 5, 6]

    # try to get the result data,
    if fig_idx=='fig2':
        outcome = analyze( 'human', mode='Rate_Reward')
    elif fig_idx=='fig5':
        simluate_data( 'G_model_t')
        outcome = analyze( 'G_model_t', mode='Rate_Reward')
    elif fig_idx=='optimal':
        simluate_data( 'optimal')
        outcome = analyze( 'optimal', mode='Rate_Reward')

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
    if fig_idx=='fig2':
        plt.ylim([.3, .7])
    elif fig_idx=='fig5':
        plt.ylim([.25, .6])    
    plt.legend( conds)

    try:
        plt.savefig( f'{path}/figures/Gershman21_{fig_idx}')
    except:
        os.mkdir('figures')
        plt.savefig( f'{path}/figures/Gershman21_{fig_idx}')


def Fig7():

    with open( f'{path}/data/params_dict2.pkl', 'rb')as handle:
        params_mat = pickle.load( handle) 

    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size': 15})
    params_name = [ 'α_v', 'α_θ', 'α_a', 'β']
    conds = [ 'HC', 'SZ']
    is_szs = params_mat[ :, -1]
    x = np.arange(params_mat.shape[0])
    for i, parameter in enumerate(params_name):
        plt.subplot( 2, 2, i+1)
        for j, _ in enumerate( conds):
            param_summary = params_mat[ :, (is_szs==j)]
            plt.scatter( x, param_summary, color='b')
            plt.title( f'{parameter}')
    plt.savefig( f'{path}/figures/Gershman21_fig7')

def Fig_Icare():

    # analyze the data 
    Rate_Rew = analyze( 'optimal', mode='Rate_Reward')
    Set_Size = analyze( 'optimal', mode='Set_Size')

    # show figure 
    plt.figure(figsize=(10,4))
    plt.rcParams.update({'font.size': 15})
    setsizes = np.array( [ 2, 3, 4, 5, 6])
    trial_per_sitmuli = np.arange( 1, 10)
    conds = [ 'HC', 'SZ'] 

    plt.subplot( 1, 3, 1)
    for zi, _ in enumerate(setsizes):
        plt.plot( trial_per_sitmuli, Set_Size[ :, zi, 0], 'o-', linewidth=1)
    plt.xlabel( 'Trial per Sitmuli')
    plt.ylabel( 'Accuracy')
    plt.title( 'Set Size Effect of HC')

    plt.subplot( 1, 3, 2)
    for zi, _ in enumerate(setsizes):
        plt.plot( trial_per_sitmuli, Set_Size[ :, zi, 1], 'o-', linewidth=1)
    plt.xlabel( 'Trial per Sitmuli')
    plt.ylabel( 'Accuracy')
    plt.title( 'Set Size Effect of SZ')

    plt.subplot( 1, 3, 3)
    cond_colors = [ 'b', 'r' ]
    mus  = np.zeros([len(setsizes), 2]) 
    stds = np.zeros([len(setsizes), 2]) 
    for j, _ in enumerate(conds):
        for i, _ in enumerate( setsizes):
            a = Rate_Rew[ 'Rate_data'][ :, i, j]
            mu, _, mu_interval,_ = normfit( a[~np.isnan(a)])
            mus[i, j]  = mu
            stds[i, j] = np.diff(mu_interval)[0]/2
        plt.plot( setsizes, mus[:,j], 'o-', color=cond_colors[j], linewidth=3)
        plt.errorbar( setsizes, mus[:,j], stds[:,j], color=cond_colors[j])
    plt.title( 'Pi comp. vs set size')
    plt.xlabel( 'set size')
    plt.ylabel( 'Pi complexity')
    
    plt.savefig( f'{path}/figures/SetSize.png')

    
if __name__ == '__main__':

    # show figure
    plot_figures(args.fig_idx) 
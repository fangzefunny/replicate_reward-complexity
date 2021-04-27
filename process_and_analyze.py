import os 
import pickle 
import numpy as np 
import pandas as pd 

from scipy.special import psi, logsumexp 

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))

'''
SEC0: Basic Functions
'''

## estimate mutual information from data 
def MI_from_data( xs, ys, prior=None):
    '''Hutter estimator of MI
    '''

    nX = len(np.unique(xs))
    nY = len(np.unique(ys))
    counts = np.zeros( [nX, nY])

    if prior==None:
        # if there is no prior
        # use empirical prior
        prior = 1 / (nX * nY)

    # estimate the mutual information 
    for i, x in enumerate(np.unique(xs)):
        for j, y in enumerate(np.unique(ys)):
            counts[ i, j] = prior + np.sum( (xs==x) * (ys==y))

    # https://papers.nips.cc/paper/2001/file/fb2e203234df6dee15934e448ee88971-Paper.pdf equation 4
    n = np.sum( counts)
    nX = np.sum( counts, axis=0, keepdims=True)
    nY = np.sum( counts, axis=1, keepdims=True)
    P = psi( counts+1) - psi( nX+1) - psi(nY+1) + psi(n+1)
    MI = np.sum( counts * P ) / n
    if MI > 3:
        print(MI)

    return MI

# Blahut Arimoto algorithm to
# get rate-distortion curve

def Blahut_Arimoto( distort, p_x, 
                    beta,
                    tol=1e-3, max_iter=50):
    
    # init for iteration
    nX, nY = distort.shape[0], distort.shape[1]
    p_y1x  = np.ones([nX, nY]) / nY
    p_y    = ( p_x.T @ p_y1x).T
    done   = False 
    i      = 0

    while not done:

        # cache  the old channel for convergence check
        old_p_y1x = p_y1x

        # update p(y|x) ∝ p(y)exp( -βD(x,y))
        log_p_y1x = - beta * distort + np.log(p_y.T + np.finfo(float).eps) 
        p_y1x     = np.exp( log_p_y1x - logsumexp( log_p_y1x, axis=-1, keepdims=True)) 

        # update p_y = ∑_x p(x)p(y|x)
        p_y = (p_x.T @ p_y1x).T 
    
        # counter 
        i += 1

        # check convergence
        if np.sum(abs( p_y1x - old_p_y1x)) < tol:
            done = True 
        
        if i >= max_iter:
            # print( f'''
            #         The BA algorithm reaches maximum iteration {max_iter},
            #         the outcome might be inaccurate
            #         ''')
            done = True

    return p_y1x, p_y 

'''
SEC1: Preprocessing
'''

def pre_process():
    '''Split the data set into data per subject
    '''
        
    # load data 
    csv_data = pd.read_csv(f'{path}/data/collins14_SZ.csv') 
    csv_data = csv_data.drop( columns = ['Unnamed: 0'])

    # change the columns of the data
    header = [ 'id', 'block', 'setSize', 'trial',
            'state', 'image', 'folder', 'iter',
            'correct_act', 'action', 'key', 'cor',
            'reward', 'rt', 'is_sz', 'pcor', 'delay']
    csv_data.columns = header 

    # change the state and action value to match python
    csv_data.state       -= 1 
    csv_data.action      -= 1
    csv_data.correct_act -= 1

    # separate the whole data set 
    data = dict()
    subjects = np.unique( csv_data.id)

    # split the data into HC group and SZ group
    # HC: represents the normal human group, is_SZ==0
    # SZ: represents the patient group,      is_SZ==1
    for sub in subjects:
        data[sub] = csv_data[ (csv_data.id==sub)]

    # save data as pkl file 
    with open( f'{path}/data/collins_data_14.pkl', 'wb')as handle:
        pickle.dump( data, handle) 

'''
SEC2: Analyze the data  
'''

def Rate_Reward( data, prior):
    '''Analyze the data

    Analyze the data to get the rate distortion curve,

    Input:
        data

    Output:
        Theoretical rate and distortion
        Empirical rate and distortion 
    '''
 
    # prepare an array of tradeoff
    betas = np.logspace( np.log10(.1), np.log10(10), 50)

    # create placeholder
    # the challenge of creating the placeholder
    # is that the length of the variable change 
    # in each iteration. To handle this method, 
    # my strategy is to create a matrix with 
    # the maxlength of each variable and then, use 
    # nanmean to summary the variables
    
    # get the number of subjects
    num_sub = len(data.keys())
    max_setsize = 5
    
    # create a placeholder
    results = dict()
    summary_Rate_data = np.empty( [ num_sub, max_setsize, 2]) + np.nan
    summary_Val_data  = np.empty( [ num_sub, max_setsize, 2]) + np.nan
    summary_Rate_theo = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan
    summary_Val_theo  = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan

    # run Blahut-Arimoto
    for subi, sub in enumerate(data.keys()):

        #print(f'Subject:{subi}')
        sub_data  = data[ sub]
        blocks    = np.unique( sub_data.block) # all blocks for a subject
        setsize   = np.zeros( [len(blocks),])
        Rate_data = np.zeros( [len(blocks),])
        Val_data  = np.zeros( [len(blocks),])
        Rate_theo = np.zeros( [len(blocks), len(betas)])
        Val_theo  = np.zeros( [len(blocks), len(betas)])
        #errors    = np.zeros( [len(blocks),])
        #bias_state= np.zeros( [len(blocks), 6]) 

        # estimate the mutual inforamtion for each block 
        for bi, block in enumerate(blocks):
            idx      = (sub_data.block == block)
            states   = sub_data.state[idx].values
            actions  = sub_data.action[idx].values
            cor_acts = sub_data.correct_act[idx].values
            rewards  = sub_data.reward[idx].values
            is_sz    = int(sub_data.is_sz.values[0])
            
            # estimate some critieria 
            #errors[bi]    = np.sum( actions != cor_acts) / len( actions)
            Rate_data[bi] = MI_from_data( states, actions, prior)

            Val_data[bi] = np.mean( rewards)

            # estimate the theoretical RD curve
            S_card  = np.unique( states)
            A_card  = range(3)
            nS      = len(S_card)
            nA      = len(A_card)
            
            # calculate distortion fn (utility matrix) 
            Q_value = np.zeros( [ nS, nA])
            for i, s in enumerate( S_card):
                a = int(cor_acts[states==s][0]) # get the correct response
                Q_value[ i, a] = 1

            # init p(s) 
            p_s     = np.zeros( [ nS, 1])
            for i, s in enumerate( S_card):
                p_s[i, 0] = np.mean( states==s)
            p_s += np.finfo(float).eps
            p_s = p_s / np.sum( p_s)
            
            # run the Blahut-Arimoto to get the theoretical solution
            for betai, beta in enumerate(betas):
                
                # get the optimal channel for each tradeoff
                pi_a1s, p_a = Blahut_Arimoto( -Q_value, p_s,
                                              beta)
                # calculate the expected distort (-utility)
                # EU = ∑_s,a p(s)π(a|s)Q(s,a)
                theo_util  = np.sum( p_s * pi_a1s * Q_value)
                # Rate = β*EU - ∑_s p(s) Z(s) 
                # Z(s) = log ∑_a p(a)exp(βQ(s,a))  # nSx1
                Zstate     = logsumexp( beta * Q_value + np.log(p_a.T), 
                                    axis=-1, keepdims=True)
                theo_rate  = beta * theo_util - np.sum( p_s * Zstate)

                # record
                Rate_theo[ bi, betai] = theo_rate
                Val_theo[ bi, betai]  = theo_util

            setsize[bi] = len(np.unique( states))

        for zi, sz in enumerate([ 2, 3, 4, 5, 6]):
            summary_Rate_data[ subi, zi, is_sz] = np.nanmean( Rate_data[ (setsize==sz),])
            summary_Val_data[ subi, zi, is_sz]  = np.nanmean(  Val_data[ (setsize==sz),])
            summary_Rate_theo[ subi, :, zi] = np.nanmean( Rate_theo[ (setsize==sz), :],axis=0)
            summary_Val_theo[ subi, :, zi]  = np.nanmean(  Val_theo[ (setsize==sz), :],axis=0)

    # prepare for the output 
    results[ 'Rate_theo'] = np.nanmean( summary_Rate_theo, axis=0)
    results[  'Val_theo'] = np.nanmean(  summary_Val_theo, axis=0)
    results[ 'Rate_data'] = summary_Rate_data
    results[  'Val_data'] = summary_Val_data

    return results

def analyze():

    # load data
    with open( f'{path}/data/collins_data_14.pkl', 'rb')as handle:
        data = pickle.load( handle)  
    
    # analyze the data 
    prior = .1 # concentration prior for the dirichlet process
    outcome = Rate_Reward( data, prior)

    # save the analysis 
    with open( f'{path}/data/results_collins_data_14.pkl', 'wb')as handle:
        pickle.dump( outcome, handle) 

if __name__ == '__main__':

    # preprocess 
    pre_process() 

    # analyze
    analyze()


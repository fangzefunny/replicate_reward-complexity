import os 
import pickle
import scipy.stats
import numpy as np 
import pandas as pd 
import multiprocessing as mp 

from scipy.special import psi, logsumexp 
from scipy.optimize import minimize
from agents import *

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))
# find the machine epsilon
eps_ = np.finfo(float).eps 

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
    with open( f'{path}/data/human_data.pkl', 'wb')as handle:
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

def analyze( agent, mode):

    # load data
    with open( f'{path}/data/{agent}_data.pkl', 'rb')as handle:
        data = pickle.load( handle)  
    
    # analyze the data 
    if mode == 'Rate_Reward':
        prior = .1 # concentration prior for the dirichlet process
        RatRew = Rate_Reward( data, prior)
        return RatRew 

    if mode == 'Set_Size':
        SetSize = set_size_effect( data)
        return SetSize 

'''
SEC3: Data simulation

In this section, I will
    - fit inividuals' parameters 
    - simulate the data using fitted parameters
'''

class model:

    def __init__( self, agent, sub_idx):
        self.agent = agent
        self.sub   = sub_idx

    def nLL( self, data, params):
        '''Calculate -log p(b|G,T,θ)

        -log p(b|G,T,θ) = ∑_bi -log(bi|G,T,θ)
        
        Input:

            b: sampled block data
            bi: each data point
            T: task 
            G: self.agent 
            θ: agent's parameters 
        
        return:
            NLL: the goodness of fit given model and a specific 
                 set of parameters
        '''
        neg_log_like = 0.
        nS = len( data.state.unique())
        nA = 3 # this is hardcrafted for this experiment, 
               # remember to change it when implementing other data set
        agent = self.agent( nS, nA, params)

        for t in range( data.shape[0]):

            # get st, at for each time step,
            # turn them to int format to faciliate indexing
            state  = int( data.state.values[t])
            action = int( data.action.values[t])
            reward = data.reward.values[t]

            # store 
            agent.memory.push( state, action, reward, t+1)

            # evaluate the action: π(at|st)
            pi_at1st = agent.eval_action( state, action) 

            # calculate NLL: -log π(at|st)
            # add and machine epislon to the log
            # to prevent NaN numerical problem
            neg_log_like += -np.log( pi_at1st + eps_)

            # model learn from the experience 
            agent.update()

        return neg_log_like

    def likelihood_( self, params):
        '''Calculate -log p(Data|G,T,θ)

        p(Data|G,T,θ) = ∏_b p(b|G,T,θ)
        -log p(Data|G,T,θ) = ∑_b -log p(b|G,T,θ)

        Input:
            Data: subject data
            b: block data
            T: task 
            G: self.agent 
            θ: agent's parameters 
        
        return:
            NLL: the goodness of fit given model and a specific 
                 set of parameters

        '''
        # split the data into data in each block
        tot_nll = 0.
        blocks = np.unique(self.data.block.values)
        for bi in blocks:
            block_data = self.data[ self.data.block==bi]
            tot_nll += self.nLL( block_data, params)
        return tot_nll

    def fit( self, data, bnds, seed=2021, init=[], verbose=False):
        '''Fit model using MLE

        θ* = argmin_θ -log p(Data|G,T,θ)
        
        Among them, data, task (T) and agent (G) are fixed.

        Input: 

            data: Data
            bnd: bounds of the parameters
            seed: random seed. very important when running 
                the parallel computing
            init: initilaization of the parameters.

        Output:
            θ* : optimal parameters
            -log p(data|G,θ*): lowest LL
        '''
        # prepare for the fit 
        np.random.seed( seed)
        self.data = data 
        n_params = len( bnds)

        # if we do not know what is a good initialization
        # point that guarantee to find the global minima 
        # we just randomize the initialization 
        if len(init) == 0:
            # init parameter 
            param0 = list() 
            for i in range( n_params):
                # random init from the bounds
                i0 = bnds[i][0] + (bnds[i][1] - bnds[i][0])*np.random.rand()
                param0.append( i0)
        else:
            param0 = init
        if verbose:
            print( f'''
                    Init with params:
                    {param0}
                    ''')
            
        # start fit 
        result = minimize( self.likelihood_, param0,
                            bounds=bnds, 
                            options={'disp':False})

        # result.x: θ*
        # result.fun: -log p(Data|G,T,θ*) 
        if verbose:
            print( f'''
                    optimal params: {result.x}
                    lowest mle: {result.fun}
                    ''')

        return result.x, result.fun

    def simulate( self, data, params):
        '''Generate synthesis reponse

        Data ~ p(DATA|G,T,θ*)
        '''
        nS = len( data.state.unique())
        nA = 3 # this is hardcrafted for this experiment, 
               # remember to change it when implementing other data set
        agent = self.agent( nS, nA, params)
        # I add this extra variable hoping it may help 
        data['prob'] = float('nan')  

        # iterate the trial to generate data 
        # and record the response
        for t in range( data.shape[0]):

            # get st, at for each time step,
            # turn them to int format to faciliate indexing
            state  = int( data.state.values[t])
            action = int( agent.get_action(state))
            correct_act = int(data.correct_act.values[t])
            reward = np.sum( correct_act == action)

            pi_a1s = agent.eval_action( state, correct_act)

            # learn from memory from the experience 
            agent.memory.push( state, action, reward, t+1)
            agent.update()
        
            # record the generated trajectory
            data['action'][t]     = action
            data['prob'][t]       = pi_a1s
            data['reward'][t]     = reward

        return data 

def fit_subject_data( process_model, n_cores=0):
    '''Fit to each subject's data 

    In the raw the data, each ID means the data from one subject
    In the preprocessing, we had splited the data into data in 
    each subject. 

    What we do here, 

    - load the data in each subject
    - further split the data into data in each block
    - use the block data as the unit to fit the data 

    To accelerate the fit, I use parallel computing.
    You need to decide how many cores you want to use.
    I limited the max cores as 6, which is a feasible
    number for most computers.

    '''
    # decide how many parallel CPU cores
    if n_cores==0:
        n_cores = int( mp.cpu_count())
    if n_cores>=6:
        n_cores = 6

    # choose what agent to use 
    if process_model == 'G_model_t':
        what_agent = G_model_t
    elif process_model == 'G_model':
        what_agent = G_model
    elif process_model == 'optimal':
        what_agent = RLbaseline
    else:
        raise Exception( 'choose the correct model')

    # load the subject data
    with open( f'{path}/data/human_data.pkl', 'rb')as handle:
        data = pickle.load( handle)  

    # pre assign the storage,
    # note that number of parameter is defined by Gershman's model
    num_sub = len(data.keys())
    params_name = [ 'α_v', 'α_θ', 'α_a', 'β']
    num_param = 4 
    params_dict = np.zeros( [ num_sub, num_param])
    pool = mp.Pool( n_cores)

    for subi, sub in enumerate(data.keys()):

        # display the fitting progress 
        print( f'fit subject {subi}')

        # data in each subject and load the RL data
        sub_data = data[ sub]
        agent = model( what_agent, sub)
        bnds = ( (.0, .95), (.0, .95), (.0, .95), (.0001, 80))
        
        # init the fitting, what we do here is
        # we need to fit multiplt times to find the 
        # best global minima approximation.

        # to accelerate the fitting, I use parallel computing
        # trick. 6 fits as one batch. 
        # If the lowest mle loss is small (<400) 
        #    then we terminate and move towards next subject, 
        # else  
        #    we run another batch
        min_loss = np.inf 
        done = False
        i = 0
        num_fits = 6
        min_bar = 400

        while not done:
            
            # init the pool and fit using parallel method
            seed = 2020 + (subi+ i)*10 
            results = [ pool.apply_async(agent.fit, 
                        args=( sub_data, bnds, seed+2*j, [], True)
                        ) for j in range(num_fits)]
            
            # record the min fit
            for p in results:
                params, loss = p.get()
                if loss < min_loss:
                    min_loss = loss 
                    params_opt = params

            # accept the fit if the 
            # nll is smaller than 1000
            if (min_loss<min_bar) or (i>0):
                done = True 

            i += 1

        print( f'''
                Final decision {params_opt}:
                With loss: {min_loss}
                ''')
        params_dict[subi, :] = params_opt

    params_df = pd.DataFrame( params_dict, columns=params_name)
    params_df.to_csv( f'{path}/data/params_{process_model}.csv')

def simluate_data( process_model):

    if process_model == 'G_model_t':
        what_agent = G_model_t
    elif process_model == 'G_model':
        what_agent = G_model
    elif process_model == 'optimal':
        what_agent = RLbaseline
    else:
        raise Exception( 'choose the correct model')

    # load data
    with open( f'{path}/data/human_data.pkl', 'rb')as handle:
        human_data = pickle.load( handle) 
    params = pd.read_csv(f'{path}/data/params_{process_model}.csv', index_col=0).to_numpy()

    # for each subject
    sim_data = dict()
    for subi, sub in enumerate(human_data.keys()):

        # init a blank data frame to store the simluated data
        sub_data = human_data[ sub]
        sub_params = params[ subi, :]
        headers = [ 'id', 'block', 'setSize', 'trial',
                    'state', 'image', 'folder', 'iter',
                    'correct_act', 'action', 'key', 'cor',
                    'reward', 'rt', 'is_sz', 'pcor', 
                    'delay', 'prob']
        sim_sub_data = pd.DataFrame( columns=headers)
        blocks  = np.unique(sub_data.block.values)
        agent = model( what_agent, sub)
        
        for bi in blocks: 
            block_data = sub_data.loc[ sub_data.block == bi
                            ].reset_index(drop=True).copy()
            sim_block_data = agent.simulate( block_data, sub_params)
            sim_sub_data = pd.concat( [sim_sub_data, sim_block_data], 
                                                     axis=0, sort=True)
        sim_data[sub] = sim_sub_data
    
    # save the simluate data 
    with open( f'{path}/data/{process_model}_data.pkl', 'wb')as handle:
        pickle.dump( sim_data, handle) 

def set_size_effect( data):
    '''
    '''
    # prepare to analyze the set size effect
    setsizes = np.array([ 2, 3, 4, 5, 6])
    nums = np.zeros([1, 1, 2])
    trial_per_sitmuli = np.arange( 1, 10)
    accs = np.zeros([len(trial_per_sitmuli), len(setsizes), 2])

    # iterate over data in each subject to collect
    # the accuracy across the trials per stimuli
    for sub in data.keys():
        sub_data = data[sub]
        is_sz = int(sub_data.is_sz.values[0])
        nums[0, 0, is_sz] += 1
        for zi, sz in enumerate(setsizes):
            for ii, it in enumerate(trial_per_sitmuli):
                idx = ((sub_data.setSize == sz) &
                       (sub_data.iter == it))
                acc = np.mean(sub_data.prob.values[idx])
                accs[ ii, zi, is_sz] += acc 
    accs /= nums 
    return accs 

if __name__ == '__main__': 

    # preprocessing 
    pre_process()

    # fit the model to human data 
    fit_subject_data( 'G_model_t', n_cores=0)

    








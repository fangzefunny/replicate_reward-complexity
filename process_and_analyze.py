import os 
import pickle 
import numpy as np 
import pandas as pd 

from scipy.special import psi, logsumexp 
from scipy.optimize import minimize

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

def prepare_fig2():
    # preprocess 
    pre_process() 
    # analyze
    analyze()

'''
SEC3: Simluate data  
'''

class collins_task:

    def __init__( self, nS):
        self.nS = nS
        self.nA = 3
        self.trial_per_stimuli = 11
        self.t  = 0
        self.reset()
    
    def reset( self):
        self.reward_fn = np.zeros([ self.nS, self.nA])
        for s in range(self.nS):
            idx = np.random.choice(3)
            self.reward_fn[ s, idx] = 1.
        self.states = np.random.permutation( np.tile( np.arange(self.nS), 
                                            [self.trial_per_stimuli,]))
        self.T = self.trial_per_stimuli * self.nS
        self.done = False

    def stimuli( self):
        self.s = self.states[self.t]
        return self.s 
    
    def step( self, act):
        r = self.reward_fn[ self.s, act]
        self.t += 1
        if self.t >= self.T:
            self.done = True
        correct_act = np.argmax(self.reward_fn[ self.s])
        return r, correct_act, self.done 
    
class model:

    def __init__( self, agent, task, sub_idx, is_sz):
        # insert an agent that allows us
        # to switch them 
        self.agent = agent
        self.task  = task
        self.sub   = sub_idx
        self.is_sz = is_sz

    def nLL( self, params):
        '''Calculate -log p(data|G,T,θ)

        Input:

            data: sampled data
            T: task 
            G: self.agent 
            θ: agent's parameters 
        
        return:
            NLL: the goodness of fit given model and a specific 
                 set of parameters
        '''
        data = self.data
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
            agent.memory.push( state, action, reward)

            # evaluate the action: π(at|st)
            pi_at1st = agent.eval_action( state, action) 

            # calculate NLL: -log π(at|st)
            # add and machine epislon to the log
            # to prevent NaN numerical problem
            neg_log_like += - np.log( pi_at1st + eps_)

            # model learn from the experience 
            agent.update()

        return neg_log_like

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
        result = minimize( self.nLL, param0,
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

    def simulate( self, params):
        '''Generate synthesis data

        Data ~ p(DATA|G,T,θ*)

        The simulation includes all set sizes
        '''
        data = dict()
        subs    = []
        setSizes= []
        states  = []
        actions = []
        correct_acts = []
        rewards = []
        is_sz   = []
        qvalues = []
        
        for nS in [ 2, 3, 4, 5, 6]:

            # init environment 
            env = self.task( nS)
            # get state and action cardinarity
            nS = self.env.nS
            nA = self.env.nA
            agent = self.agent( nS, nA, params)
            done = False

            while not done:

                # get st, at for each time step,
                # turn them to int format to faciliate indexing
                state  = int( env.stimuli())
                action = agent.get_action(state)
                reward, correct_act, done = env.step( action)

                # record the data 
                subs.append( self.sub)
                setSizes.append( nS)
                states.append( state)
                actions.append( action)
                correct_acts.append( correct_act)
                rewards.append( reward)
                is_sz.append( self.is_sz)
                qvalues.append( agent.q_value( state, action))
            
            data['setSize'] = setSizes
            data['state'] = states
            data['action'] = actions
            data['correct_act'] = correct_acts
            data['reward'] = rewards
            data['id'] = subs
            data['is_sz'] = is_sz
            data['qvalues'] = qvalues

        return data 

# the replay buffer to store the memory 
class simpleBuffer:
    
    def __init__( self):
        self.table = []
        
    def push( self, *args):
        self.table = tuple([ x for x in args]) 
        
    def sample( self ):
        return self.table

# define a agent used in Gershman's paper 
class gradient_based:

    def __init__( self, obs_dim, action_dim, params):
        self.obs_dim = obs_dim 
        self.action_dim = action_dim
        self.action_space = range( self.action_dim)
        self._init_critic()
        self._init_actor()
        self._init_marginal_obs()
        self._init_marginal_action()
        self._init_memory()
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.beta     = params[3]

    def _init_critic( self):
        self.v = np.zeros([ self.obs_dim, 1])

    def _init_actor( self):
        self.theta = np.zeros( [ self.obs_dim, self.action_dim]) + 1e-20
        self.pi    = np.ones( [ self.obs_dim, self.action_dim]) / self.action_dim
    
    def _init_marginal_obs( self):
        self.p_s   = np.ones( [ self.obs_dim, 1]) / self.obs_dim
    
    def _init_marginal_action( self):
        self.p_a   = np.ones( [ self.action_dim, 1]) / self.action_dim

    def _init_memory( self):
        self.memory = simpleBuffer()

    def value( self, obs):
        v_obs = self.v[ obs, 0]
        return v_obs 

    def q_value( self, obs, action):
        q_sa = self.v[ obs, 0] * self.theta[ obs, action] 
        return q_sa 
        
    def eval_action( self, obs, action):
        pi_obs_action = self.pi[ obs, action]
        return pi_obs_action
    
    def get_action( self, obs):
        pi_obs = self.pi[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
        
    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + 1e-20) \
                   - np.log( self.p_a[ action, 0] + 1e-20)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * β * I(s=st) * δ *[1- π(at|st)] --> [nS,] 
        I_s = np.zeros([self.obs_dim])
        I_s[obs] = 1.
        self.theta[ :, action] += self.lr_theta * rpe * \
                                  self.beta * I_s * (1 - pi_like) 

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log(self.p_a.T) + 1e-15
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + 1e-20
        self.p_a = self.p_a / np.sum( self.p_a)

class gershman_model2( gradient_based):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)

    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * [β + π(at|st)/(p(at)*N)]* δ *[1- π(at|st)] --> scalar 
        self.theta[ obs, action] += self.lr_theta * rpe * (1 - pi_like) \
                                   * (self.beta + pi_like/self.p_a[action, 0]/self.obs_dim)

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log(self.p_a.T + eps_) + eps_
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))

        if np.isnan(np.sum(self.pi)):
            print(1)
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

def synthetic_data():

    # load data
    with open( f'{path}/data/collins_data_14.pkl', 'rb')as handle:
        data = pickle.load( handle)  
    num_sub = len(data.keys())
    params_dict = dict()
    for subi, sub in enumerate(data.keys()):
        print( f'fit subject {subi}')
        sub_data = data[ sub]
        is_sz = sub_data.is_sz.values[0]
        sub_model = model( gershman_model2, collins_task, 
                           sub, is_sz)
        bnds = ( (.0001, .95), (.0001, .95), (.0001, .95), (.0001, 80))
        done = False
        i = 0 
        params_dict = np.zeros( [ num_sub, len(bnds)+1])
        temp_params = []
        temp_losses = []
        while not done:
            i += 1
            seed = 2020 + subi*10 + i
            params, loss = sub_model.fit( sub_data, bnds, seed=seed, verbose=True)
            temp_params.append( params)
            temp_losses.append( loss)
            min_loss, min_idx = np.min(temp_losses), np.argmin(temp_losses)
            params_opt = temp_params[ min_idx]
            # accept the fit if the nll is smaller than 1000
            if i >= 5:
                if (min_loss < 600) or (i>=10):
                    done = True 
        print( f'''
                Final decision {params_opt}:
                With loss: {min_loss}
                ''')
        params_dict[subi, :-1] = params_opt
        params_dict[subi, -1]  = is_sz 
    
    with open( f'{path}/data/params_dict.pkl', 'wb')as handle:
        pickle.dump( params_dict, handle) 

if __name__ == '__main__': 
    synthetic_data()









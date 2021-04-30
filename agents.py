import numpy as np 
from scipy.special import logsumexp 

# find the machine epsilon
eps_ = np.finfo(float).eps 

# the replay buffer to store the memory 
class simpleBuffer:
    
    def __init__( self):
        self.table = []
        
    def push( self, *args):
        self.table = tuple([ x for x in args]) 
        
    def sample( self ):
        return self.table

# Define a base model used in Gershman and Lai 2021 paper
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
        self.theta = np.zeros( [ self.obs_dim, self.action_dim]) + eps_
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
        raise NotImplementedError

'''
Process Model 1: 

Gershman model while the learning rate of the 
policy is not scaled by t.
'''
class G_model( gradient_based):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)

    def update(self):
        
        # collect sampeles 
        obs, action, reward, t = self.memory.sample() 

        # calculate v prediction: V_hat(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V_hat(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * δ *[1- π(at|st)] 
        #                                      * [β + π(at|st)/(p(at)*nS)]  --> scalar 
        # Note that I do not scale the learning rate by t as Gershman and Lai 21
        # because as said in the paper, this is trival operation
        self.theta[ obs, action] += self.lr_theta * rpe * (1 - pi_like) \
                                   * (self.beta + pi_like/self.p_a[action, 0]/self.obs_dim)

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log( self.p_a.T + eps_) 
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a)
        self.p_a = self.p_a / np.sum( self.p_a)

'''
Process Model 2: 

Gershman model with a learning rate of the 
policy scaled by t.
'''

class G_model_t( gradient_based):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)

    def update(self):
        
        # collect sampeles 
        obs, action, reward, t = self.memory.sample() 

        # calculate v prediction: V_hat(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V_hat(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * δ *[1- π(at|st)] 
        #                                      * [β + π(at|st)/(p(at)*nS)]  --> scalar 
        # Note that I do not scale the learning rate by t as Gershman and Lai 21
        # because as said in the paper, this is trival operation
        self.theta[ obs, action] += self.lr_theta / t * rpe * (1 - pi_like) \
                                   * (self.beta + pi_like/self.p_a[action, 0]/self.obs_dim)

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log( self.p_a.T + eps_) 
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))

        if np.isnan(np.sum(self.pi)):
            print(1)
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a)
        self.p_a = self.p_a / np.sum( self.p_a)

'''
Process Model 3: 

Basic RL baseline
'''

class BaseRLAgent:
    
    def __init__( self, obs_dim, action_dim):
        self.obs_dim = obs_dim 
        self.action_dim = action_dim
        self.action_space = range( self.action_dim)
        self._init_critic()
        self._init_actor()
        self._init_marginal_obs()
        self._init_marginal_action()
        self._init_memory()
        
    def _init_marginal_obs( self):
        self.po = np.ones( [self.obs_dim, 1]) * 1 / self.obs_dim

    def _init_marginal_action( self):
        self.pa = np.ones( [self.action_dim, 1]) * 1 / self.action_dim

    def _init_memory( self):
        self.memory = simpleBuffer()
        
    def _init_critic( self):
        self.q_table = np.ones( [self.obs_dim, self.action_dim]) * 1/self.action_dim

    def _init_actor( self):
        self.pi = np.ones( [ self.obs_dim, self.action_dim]) * 1 / self.action_dim
            
    def q_value( self, obs, action):
        q_obs = self.q_table[ obs, action ]
        return q_obs 
        
    def eval_action( self, obs, action):
        pi_obs_action = self.pi[ obs, action]
        return pi_obs_action
    
    def get_action( self, obs):
        pi_obs = self.pi[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
        
    def update(self):
        return NotImplementedError

class RLbaseline( BaseRLAgent):
    '''Implement the sto-update softmax RL

    st: the current state 
    at: the current action

    Update Q:
        Q(st, at) += lr_q * [reward - Q(st, at)]

    Update Pi: 
        loss(st, :) = max Q(st, :) - Q(st, :) 
        Pi( s_t, :) = exp( -beta * loss(st, :)) / sum_a exp( -beta * loss(st, :))
    '''
    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim,)
        self.lr_q = params[0]
        self.beta = params[1]

    def update( self):

        # collect sampeles 
        obs, action, reward, _ = self.memory.sample() 

        # calculate q prediction
        q_pred = self.q_value( obs, action)

        # update critic 
        self.q_table[ obs, action] += \
                    self.lr_q * ( reward - q_pred)

        # update policy
        log_pi = self.beta * self.q_table 
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
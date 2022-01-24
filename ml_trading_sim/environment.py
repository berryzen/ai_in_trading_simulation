import numpy as np
from numba.experimental import jitclass
from numba import int32, float32, boolean,float64,int64    

spec = [
    ('_episode_ended', boolean),
    ('old_allocation',float32[:]),
    ('stack',float32),
    ('notradescount',int32),
    ('maxwallet',float32),
    ('t',int32),
    ('maxwallet2',float32),
    ('countofactions',int32),
    ('observed_data',float64[:,:]),
    ('price_data2',float64[:,:]),
    ('sim_len',int32),
    ('state_size',int32),
    ('old_action',float32),
    ('new_allocation',float32[:]),
    ('action',float32),
    ('btc_allocation',float32),
    ('usdt_allocation',float32),
    ('max1_limit',float32),
    ('max2_limit',float32),
    ('maxstack',float32),
    ('closeprice',float64[:]),
    ('price_data',float64[:]),
    ('actionn',float64[:]),
]
@jitclass(spec)
class RelativeTrade(object):
    def __init__(self, observed_data,price_data2,sim_len,max1_limit, max2_limit):
        self._episode_ended = True
        self.old_allocation = np.array([0.5,0.5], dtype=np.float32)
        self.stack = 1.0
        self.notradescount = 0
        self.maxwallet = 0.0
        self.t = 0
        self.maxwallet2 = 0.0
        self.countofactions = 1
        self.observed_data = observed_data
        self.price_data = price_data2[0]
        self.closeprice = price_data2[1]
        self.state_size =observed_data.shape[0]+1 
        self.new_allocation = np.array([0.5,0.5], dtype=np.float32)
        self.action = 0.5
        self.usdt_allocation = 0.5
        self.btc_allocation = 0.5
        self.max1_limit = max1_limit
        self.max2_limit = max2_limit
        self.sim_len = sim_len
        self.maxstack = 0.0

    def reset(self):
        self._episode_ended = False
        self.old_allocation = np.array([0.5,0.5], dtype=np.float32)
        self.new_allocation = np.array([0.5,0.5], dtype=np.float32)
        self.stack = 1.0
        self.notradescount = 0
        self.maxwallet = 0.0
        self.maxwallet2 = 0.0
        self.t = 0
        self.old_action = 0.5
        next_state_line = self.observed_data[self.t] 
        action_taken = self.old_action
        next_state = np.append(next_state_line,np.array(action_taken),axis=None).astype(np.float32)
        self.action = 0.5
        self.usdt_allocation = 0.5
        self.btc_allocation = 0.5
        self.maxstack = 0.0
        return self.stack,self.maxwallet,self.maxwallet2, False, next_state

    def step(self, actionn):
        
        # Select trading action
        if (actionn[0]) > (actionn[1] or actionn[2]):
            action = 0.0
        elif actionn[1] > (actionn[0] or actionn[2]):
            action = 1.0
        elif actionn[2] > (actionn[1] or actionn[0]):
            action = self.old_action
        else:
            action = self.old_action
        
        done = False
        _episode_ended = False
        self.action = action
        self.btc_allocation = action
        self.usdt_allocation = 1.0-self.action
        self.new_allocation[0] = self.btc_allocation
        self.new_allocation[1] = self.usdt_allocation
        change_in_allocation = self.new_allocation - self.old_allocation
        allocation_movement = np.abs(np.sum(np.where(change_in_allocation < 0,change_in_allocation,0)))
        neg_allocation_movement = 1-allocation_movement
        
        # Trade + fee
        if allocation_movement>=0.1: 
            fee = 0.999
            self.stack = (self.stack * allocation_movement * fee)+(neg_allocation_movement*self.stack)
            self.notradescount = 0
            self.old_allocation = self.new_allocation
            self.old_action = self.action

        # Don't trade
        else: 
            self.notradescount += 1
            self.btc_allocation = self.old_allocation[0]
            self.usdt_allocation = self.old_allocation[1]
            self.action = self.old_action

        #Profit calculations
        next_t = self.t + 1
        usdt_price_next = self.price_data[next_t]
        btc_profit = self.btc_allocation * usdt_price_next
        usdt_profit = self.usdt_allocation *((-1*((usdt_price_next)-1))+1)
        prereward = btc_profit + usdt_profit
        self.stack = prereward*self.stack
        reward = (prereward-1)*100000

        if self.maxstack < self.stack:
            self.maxstack = self.stack
            self.max1_limit = self.t 

        self.t = self.t + 1

        # Agent didn't get over finishline
        if ((self.closeprice[self.t]/self.closeprice[0]) > self.stack*1.10) and (self.t > 20):
            _episode_ended = True

        # Agent did get over finishline
        if self.t == self.sim_len - 7: 
            _episode_ended = True
            
        next_state_line = self.observed_data[next_t] 
        action_taken = self.action
        next_state = np.append(next_state_line,np.array(action_taken),axis=None).astype(np.float64)
        return self.maxstack,self.t,self.stack, done, next_state,_episode_ended 
            
    @property
    def get_countofactions(self):
        return self.countofactions

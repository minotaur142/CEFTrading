import collections
import numpy as np
import torch
import pandas as pd
import enum
import gym
from gym import spaces
import random 

class Actions(enum.Enum):
    Hold = 0
    Sell = 1
    Buy = 2
    
    
class STEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data,close_prices,with_replacement=True):
        super(STEnv, self).__init__()
        self.data = data
        self.close_prices = close_prices
        self.with_replacement = with_replacement
        self.data_ind = 0 
        self.episode_data = self.prepare_episode_data()
        self.n_features = self.episode_data.shape[1]
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(10,self.n_features),dtype=np.float32)
        self.offset = 10
        self.state = self.episode_data[:10]
        self.have_position = False
        self.buy_price = 0
        self.funds = 2000
        self.shares = 0
        
    def prepare_episode_data(self):
        start = random.choice(range(self.data.shape[0] - 70))
        df = self.data.iloc[start:start+70].copy()
        self.close_price = self.close_prices.loc[df.index].values
        df['position'] = 0
        return df.values
    

    def close(self):
        pass
        
    
    def step(self,action):
        self.offset += 1
        new_state = self.episode_data[self.offset-10:self.offset]
        reward = 0
        
        if (self.offset == self.episode_data.shape[0]) | ((self.funds <= 0) & (self.have_position == False)):
            done = True
            new_state[-1,-1] = 0
            if self.have_position == True:
                sell_price = self.close_price[self.offset-1]
                reward = self.shares*sell_price + self.funds - 2000
            else:
                reward = self.funds - 2000
            self.reset()
        
        elif (action == 2) & (self.have_position == False): 
            done = False
            new_state[-1,-1] = 1
            self.have_position = True
            
            self.buy_price = self.close_price[self.offset-1]
            self.state = new_state
            self.shares = self.funds // self.buy_price
            self.funds -= self.shares * self.buy_price
        
        elif (action == 1) & (self.have_position == True):
            done = False
            new_state[-1,-1] = 0
            self.state = new_state
            self.have_position = False
            
            sell_price = self.close_price[self.offset-1]
            reward = self.shares*(sell_price) 
            
            
            self.funds += reward
            self.shares = 0
            
            
        else:
            done = False
            reward = 0
            if self.have_position == True:
                if action == 2:
                    reward -= 5
                new_state[-1,-1] = new_state[-2,-1] + 1
                if new_state[-1,-2] == 0:
                    income = new_state[-1,-5]*self.shares
                    reward += income
                    
            else:
                if action == 1:
                    reward -= 5
                new_state[-1,-1] = 0
            self.state = new_state
          
            
        return new_state, done, reward, {}
        

    def reset(self):
        if self.with_replacement == False:
            self.data_ind += 1
        self.episode_data = self.prepare_episode_data()
        self.state = self.episode_data[:10]
        self.offset = 10
        self.have_position = False
        self.funds = 2000
        self.shares = 0   
        
class ValEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data,close_prices,with_replacement=True):
        super(ValEnv, self).__init__()
        self.data = data
        self.data['position'] = 0
        self.column_dict = dict(zip(list(self.data.columns),range(self.data.shape[1])))
        self.data = self.data.values
        self.close_price = close_prices
        self.with_replacement = with_replacement
        self.data_ind = 0 
        
        self.n_features = self.data.shape[1]
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(10,self.n_features),dtype=np.float32)
        self.offset = 10
        self.state = self.data[:10]
        self.have_position = False
        self.buy_price = 0
        self.funds = 2000
        self.shares = 0
        
    

    def close(self):
        pass
        
    
    def step(self,action):
        cd = self.column_dict
        self.offset += 1
        new_state = self.data[self.offset-10:self.offset]
        reward = 0
        
        if (self.offset == self.data.shape[0]) | ((self.funds <= 0) & (self.have_position == False)):
            done = True
            new_state[-1,cd['position']] = 0
            if self.have_position == True:
                sell_price = self.close_price[self.offset-1]
                reward = self.shares*sell_price + self.funds - 2000
            else:
                reward = self.funds - 2000
            self.reset()
        
        elif (action == 2) & (self.have_position == False): 
            done = False
            new_state[-1,cd['position']] = 1
            self.have_position = True
            
            self.buy_price = self.close_price[self.offset-1]
            self.state = new_state
            self.shares = self.funds // self.buy_price
            self.funds -= self.shares * self.buy_price
        
        elif (action == 1) & (self.have_position == True):
            done = False
            new_state[-1,cd['position']] = 0
            self.state = new_state
            self.have_position = False
            
            sell_price = self.close_price[self.offset-1]
            reward = self.shares*(sell_price) 
            
            
            self.funds += reward
            self.shares = 0
            
            
        else:
            done = False
            reward = 0
            if self.have_position == True:
                new_state[-1,cd['position']] = new_state[-2,-1] + 1
                if new_state[-1,-2] == 0:
                    income = new_state[-1,cd['Dividends']]*self.shares
                    reward += income
                    
            else:
                new_state[-1,cd['position']] = 0
            self.state = new_state
          
            
        return new_state, done, reward, {}
        

    def reset(self):
        if self.with_replacement == False:
            self.data_ind += 1
        self.offset = 10
        self.have_position = False
        self.funds = 2000
        self.shares = 0 
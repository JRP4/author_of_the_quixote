import numpy as np
import time

class vanillaRNN:
    def __init__(self, n_x, n_h, seq_length, learning_rate):
        # hyperparameters
        self.n_x = n_x
        self.n_h = n_h
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # initialize model parameters
        self.Wxh = np.random.randn(n_h, n_x) * 0.01
        self.Whh = np.random.randn(n_h, n_h) * 0.01
        self.Why = np.random.randn(n_x, n_h) * 0.01
        self.bh = np.zeros((n_h, 1))
        self.by = np.zeros((n_x, 1))
        
        # memory vars for adagrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
            
    def forward_pass(self, inputs, targets, hprev):
        """
        inputs -- list of integers (tokenizer: char to int)
        targets -- list of integers (tokenizer: char to int)
        hprev -- the initial hidden state
        """
        x, h, y, p = {}, {}, {}, {}
        h[-1] = np.copy(hprev)
        loss = 0

        for t in range(len(inputs)):
            
            # one hot encoder of a char
            x[t] = np.zeros((self.n_x, 1))
            x[t][inputs[t]] = 1
            h[t] = np.tanh(np.dot(self.Wxh ,x[t]) + np.dot(self.Whh ,h[t-1]) + self.bh)
            y[t] = np.dot(self.Why,h[t]) + self.by
            p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))
            loss = loss - np.log(p[t][targets[t], 0])
            
        return loss, x, h, p
    
    def backpropagation(self, x, h, p, targets):
    
        dWxh, dWhy, dWhh = np.zeros_like(self.Wxh), np.zeros_like(self.Why), np.zeros_like(self.Whh)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(h[0])
        
        for t in reversed(range(self.seq_length)):
            dy = np.copy(p[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy ,h[t].T)
            dby += dy
            dh = np.dot(self.Why.T , dy) + dhnext
            dhraw = (1 - h[t] * h[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw ,x[t].T)
            dWhh += np.dot(dhraw , h[t-1].T)
            dhnext = np.dot(self.Whh.T , dhraw)
        for dpara in [dWxh, dWhh, dWhy, dby, dbh]:
            np.clip(dpara, -5, 5, out = dpara)
            
        return dWxh, dWhh, dWhy, dbh, dby


        
    

character_size=4
hidden_size=3
sequence_length=5

inputs=[0,1,2,2]
targets=[1,2,2,3]
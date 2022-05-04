# -*- coding: utf-8 -*-
"""
Last updated on Wed May 4 18:35:30 2022

@author: simon71701
"""
import torch.nn as nn
import torch.nn.functional as F
import sympy as sym
import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from tqdm import tqdm

## Classes

class SoftmaxRescale(nn.Module):
    def __init__(self):
        super(SoftmaxRescale, self).__init__()
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def forward(self, x):
        return rescale(x)
        
class Criterion(nn.Module):
    def __init__(self, num_competitors, idx, role=None):
        super(Criterion, self).__init__()    
        self.idx = idx
        self.role = role
        
        if role == 'Conquerer' or role==None:
            self.target = torch.zeros((num_competitors), dtype=float)
            self.target[idx] = 1
        
        if role == 'Equalist':
            self.target = torch.zeros((num_competitors), dtype=float)
            self.target += 1/num_competitors
            #print(self.target)
        
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def forward(self,x):
        total = sum(x)
        return F.binary_cross_entropy(x/total, self.target)

class Network(nn.Module):
    #num_competitors = total number of networks including this one
    def __init__(self, num_competitors, num_hidden, nodes_per_hidden, role=None, dropout=0):
        
        super(Network, self).__init__()
        
        self.num_competitors = num_competitors
        self.num_hidden = num_hidden
        self.nodes_per_hidden = nodes_per_hidden
        self.dropout = dropout
        self.role = role
        
        self.input_layer = nn.Sequential(
                                nn.Linear(self.num_competitors, self.nodes_per_hidden),
                                nn.Tanh(),
                                nn.Dropout(self.dropout)
        )
        
        self.hidden_layers = nn.ModuleList()
        
        for num in range(num_hidden):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(self.nodes_per_hidden, self.nodes_per_hidden),
                    nn.Tanh(),
                    nn.Dropout(self.dropout)
                )
            )
        
        self.output_layer = nn.Sequential(
                                nn.Linear(self.nodes_per_hidden, self.num_competitors),
                                nn.Tanh(),
                                nn.Softmax(dim=0),
                                SoftmaxRescale()
        )
        
    def rescale(self, sf_output):
        sf_output = sf_output - torch.mean(sf_output)
    
    def forward(self, x):
        x = x.float()
        
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        return x

## Functions

def rescale(sf_output):
        sf_output = (sf_output - torch.mean(sf_output))/(torch.std(sf_output))
        return sf_output

def npSoftmax(x):
    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def PredPreyStep(initial_condition, parameters, nets, step):
    dots = np.zeros(len(initial_condition))
    final = np.zeros(len(initial_condition))
    
    for i in range(len(initial_condition)):
        for j in range(len(parameters[i])):
            
            if i != j:
                dots[i] += initial_condition[i]*initial_condition[j]*parameters[i][j]
            
            if i == j:
                dots[i] += initial_condition[i]*parameters[i][j]
    
    
    for i in range(len(initial_condition)):
        final[i] = initial_condition[i] + dots[i]*step
        
        if nets[i].alive == False:
            final[i] = 0
            
        if final[i] < 0:
            final[i] = 0
            nets[i].alive = False
    

    return final

def compileParameters(r_scores):
    
    num_pops = len(r_scores)

    
    parameters = np.zeros((num_pops,num_pops))
    
    for i in range(num_pops):
        for j in range(num_pops):
            if i != j:
                parameters[i][j] = abs(np.mean((r_scores[i][j],r_scores[j][i]))) * np.sign(r_scores[i][j])
            else:
                parameters[i][j] = r_scores[i][j]
    
    return parameters

def displayPredPreyEqs(parameters):
    eqs = []
    variables = []
    
    for i in range(len(parameters)):
        variables.append(sym.Symbol('x_{0}'.format(i)))
    
    
    for i in range(len(parameters)):
        coef = sym.sympify(parameters[i][0]).evalf(4)
        expr = coef*variables[0]
        for j in range(1,len(parameters[i])):
            coef = sym.sympify(parameters[i][j]).evalf(4)
            expr += coef*variables[j]
        
        expr = variables[i]*(expr)
        
        eq = sym.Eq(sym.Symbol("x'_{0}".format(i)), expr)
        
        eqs.append(eq)
        
    return eqs

def displayParamEvolution(params):
    plt.clf()
    count = 0
    for param_set in params:
        
        compiled = [[] for j in params]
        
        for i in range(len(param_set)):
            for j in range(len(param_set[i])):
                compiled[j].append(param_set[i][j])
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        for i in range(len(compiled)):
            ax.plot(compiled[i], label="Population {0}".format(i))
        
        ax.set_title("Evolution of Parameters: Population {0}".format(count))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Parameters")
        #ax.set_ylim((-1,1))
        ax.legend()
        
        count += 1
        
    plt.show()
    
    return compiled

def displayRScoreEvolution(r_scores):
    plt.clf()
    count = 0
    for r_set in r_scores:
        
        compiled = [[] for j in r_scores]
        
        for i in range(len(r_set)):
            for j in range(len(r_set[i])):
                compiled[j].append(r_set[i][j])
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        for i in range(len(compiled)):
            ax.plot(compiled[i], label="Population {0}".format(i))
        
        ax.set_title("Evolution of Relationship Scores: Population {0}".format(count))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("R-Scores")
        #ax.set_ylim((-1,1))
        ax.legend()
        
        count += 1
        
    plt.show()
    
    return compiled

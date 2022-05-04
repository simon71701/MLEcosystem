# -*- coding: utf-8 -*-
"""
Last updated on Wed May 4 18:35:30 2022

@author: simon71701
"""

from utils import *


def main():
    torch.set_default_dtype(torch.float)

    num_nets = 5
    epochs_total = 3000
    gens = 100
    
    x = torch.zeros(num_nets) + 1
    x0 = x
    initial_conditions = x
    lr = 5e-5
    nets= []
    opts = []
    losses = []
    errors = [[] for i in range(num_nets)]
    pops = [[] for i in range(num_nets)]
    pops_0 = [[] for i in range(num_nets)]
    params = [[] for i in range(num_nets)]
    r_scores_all = [[] for i in range(num_nets)]

    step = 3e-5
    
    stop = False
    
    for i in range(len(pops)):
        pops[i].append(list(x.detach().numpy())[i]/sum(x))
        pops_0[i].append(list(x0.detach().numpy())[i]/sum(x0))
        
    layers = [1]
    nodes = [64]
    
    
    plt.clf()
    fig, ax = plt.subplots()
    
    
    for i in range(num_nets):
        # Didn't quite have time to implement different roles.
        # Current alternative role is Equalist, which causes the network to
        ## strive for equality
        
        #role = random.choice(roles)
        role='Conquerer'
    
        net = Network(num_nets,random.choice(layers),random.choice(nodes), role=role)
        net.alive = True
        nets.append(net)
        opts.append(torch.optim.Adam(net.parameters(), lr=lr))
        losses.append(Criterion(num_nets,i,role=role))
    
    epochs = tqdm(range(epochs_total))
    
    r_scores = np.zeros((len(nets),len(nets)))
    r_scores_0 = np.zeros((len(nets),len(nets)))
    
    for i in range(len(nets)):
        r_scores_0[i] = nets[i](x).detach().numpy()
    
    parameters_0 = compileParameters(r_scores_0)
    
    for epoch in epochs:
        
        for gen in range(gens):
            for i in range(len(nets)):
                if nets[i].alive:
                    opts[i].zero_grad()
                    r_scores[i] = nets[i](x).detach().numpy()
            
            
            if np.isnan(np.sum(r_scores)):
                stop = True
                print('Nan in r_scores. Try again')
                break
                
            parameters = compileParameters(r_scores)
            
            x = torch.tensor(PredPreyStep(x.detach().numpy(), parameters, nets, step),requires_grad=True)
            
            if np.isnan(np.sum(x.detach().numpy())):
                stop = True
                print('nan in pops')
                break
            
            for i in range(len(nets)):

                error = losses[i](x)
                errors[i].append(float(error))
                error.backward()
                opts[i].step()
    
            if stop == True:
                break
    
        total = sum(x)

        for i in range(len(nets)):
            pops[i].append(list(x.detach().numpy())[i]/total)
            params[i].append(list(parameters)[i])
            r_scores_all[i].append(list(r_scores[i]))
                
        
        if stop == True:
            break
        
        
        
    for i in range(len(x)):
        print()
        print('Net {0}: {1} ({2},{3})'.format(i, nets[i].role, nets[i].num_hidden, nets[i].nodes_per_hidden))
        print('Initial Proportion:', float(initial_conditions[i].detach()/sum(initial_conditions)))
        print('Final Proportion:', float(x[i].detach()/total))
        print('Final Parameters:',parameters[i])
        print()
    
    
    for i in range(len(pops)):
        ax.plot(pops[i], label="Population {0}".format(i))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Population Proportion")
    ax.set_title("Evolution of Neural Network Ecosystem")
    ax.legend()
    plt.show()
    
    compiled = displayParamEvolution(params)
    compiled = displayRScoreEvolution(r_scores_all)
    
    print('Beginning Static Ecosystem')
    
    fig0, ax0 = plt.subplots()
    
    total_0 = sum(x0)
    
    stop = False
    
    epochs = tqdm(range(epochs_total))
    
    false_nets = []
    
    for i in range(num_nets):
        net = Network(num_nets,random.choice(layers),random.choice(nodes), role=role)
        net.alive = True
        false_nets.append(net)
        
    for epoch in epochs:
        for gen in range(gens):
            
            x0 = torch.tensor(PredPreyStep(x0.detach().numpy(), parameters_0, nets, step),requires_grad=False)
            #print(x0)
            
            if np.isnan(np.sum(x.detach().numpy())):
                stop = True
                
                break 
            
        total_0 = sum(x0)
            
        for i in range(num_nets):
            pops_0[i].append(list(x0.detach().numpy())[i]/total_0)
        
        if stop == True:
            break
    
    for i in range(num_nets):
        ax0.plot(pops_0[i], label="Population {0}".format(i))
        
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Population Proportion")
    ax0.set_title("Evolution of Neural Network Ecosystem without Parameter Changes")
    ax0.legend()
    plt.show()
    
main()

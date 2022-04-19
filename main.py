# -*- coding: utf-8 -*-
"""
Last updated on Mon Apr 18 20:14:30 2022

@author: simon71701
"""

from utils import *


# Note: Make a graph showing how the ecosystem would have evolved if the parameters stayed constant
def main():
    torch.set_default_dtype(torch.float)

    num_nets = 3
    epochs = 2000
    gens = 100
    
    x = torch.zeros(num_nets) + 1
    initial_conditions = x
    lr = 5e-5
    nets= []
    opts = []
    losses = []
    errors = [[] for i in range(num_nets)]
    pops = [[] for i in range(num_nets)]
    params = [[] for i in range(num_nets)]
    r_scores_all = [[] for i in range(num_nets)]

    step = 2e-5
    
    stop = False
    
    for i in range(len(pops)):
        pops[i].append(list(x.detach().numpy())[i]/sum(x))
    
    layers = [1]
    nodes = [64]
    
    
    plt.clf()
    fig, ax = plt.subplots()
    
    
    for i in range(num_nets):
        #role = random.choice(roles)
        role='Conquerer'
        #print(role)
        net = Network(num_nets,random.choice(layers),random.choice(nodes), role=role)
        nets.append(net)
        opts.append(torch.optim.Adam(net.parameters(), lr=lr))
        losses.append(Criterion(num_nets,i,role=role))
    
    epochs = tqdm(range(epochs))
    #epochs = range(epochs)
    
    r_scores = np.zeros((len(nets),len(nets)))
    
    epoch_list = [0]
    
    for epoch in epochs:
        epoch_list.append(epoch+1)
        
        for gen in range(gens):
            for i in range(len(nets)):
                opts[i].zero_grad()
                r_scores[i] = nets[i](x).detach().numpy()
            
            if np.isnan(np.sum(r_scores)):
                stop = True
                break
                
            parameters = compileParameters(r_scores)
            
            x = torch.tensor(PredPreyStep(x.detach().numpy(), parameters, step),requires_grad=True)
            
            if np.isnan(np.sum(x.detach().numpy())):
                stop = True
                break
            
            for i in range(len(nets)):
                for j in range(len(x.detach().numpy())):
                    if x[j] == 0:
                        stop = True
    
                if stop == True:
                    break
    
                error = losses[i](x)
                errors[i].append(float(error))
                error.backward()
                opts[i].step()
    
            if stop == True:
                break
    
        total = sum(x)
        #print(total)
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
    
    print(pops)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Population Proportion")
    ax.set_title("Evolution of Neural Network Ecosystem")
    ax.legend()
    plt.show()
    
    compiled = displayParamEvolution(params)
    compiled = displayRScoreEvolution(r_scores_all)

main()

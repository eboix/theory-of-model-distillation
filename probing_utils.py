import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda")

def batch_logistic_probe_adam(rep, fns,valrep=None,valfns=None, eta=0.0001,num_iters=100, norm_bound=math.inf,iter_log=-1):
    # rep is B x m
    # fns is B x n
    B = rep.shape[0]
    m = rep.shape[1]
    n = fns.shape[1]
    # Want to find W \in \R^{m \times n} such that BW \approx fns, and columns of W are bounded
    
    model = nn.Linear(m, n).to(device)
    torch.nn.init.normal_(model.bias.data,0,0)
    torch.nn.init.normal_(model.weight.data,0,0)
    optimizer = optim.Adam(model.parameters(), lr=eta)
    loss_fn = nn.BCEWithLogitsLoss()

    # Run training
    for i in tqdm(range(0, num_iters)):
                
        optimizer.zero_grad()
        
        rownorms = torch.sqrt(torch.sum(model.weight.data**2,dim=1))
        outsideball = rownorms > norm_bound
        model.weight.data[outsideball,:] = model.weight.data[outsideball,:] / rownorms[outsideball].view(-1,1) * norm_bound
        if iter_log > -1 and i % iter_log == 0:
            with torch.no_grad():
                train_acc = torch.sum(torch.sign(model(rep)) == torch.sign(fns),dim=0) / B
                print('Train',train_acc)
                if valrep is not None:
                    test_acc = torch.sum(torch.sign(model(valrep)) == torch.sign(valfns),dim=0) / valfns.shape[0]
                    print('Test',test_acc)
        
        predictions = model(rep)
        loss = loss_fn(predictions, (fns+1)/2)
        loss.backward()
        optimizer.step()
    
    train_acc = torch.sum(torch.square(model(rep) - fns),dim=0) / B
    if valrep is not None:
        test_acc = torch.sum(torch.sign(model(valrep)) == torch.sign(valfns),dim=0) / valfns.shape[0]
        return train_acc, test_acc
    else:
        return train_acc
    
    
def batch_linear_probe_adam(rep, fns,valrep=None,valfns=None, eta=0.0001,num_iters=100,norm_bound=math.inf,iter_log=-1):
    # rep is B x m
    # fns is B x n
    B = rep.shape[0]
    m = rep.shape[1]
    n = fns.shape[1]
    # Want to find W \in \R^{m \times n} such that BW \approx fns, and columns of W are bounded
    
    model = nn.Linear(m, n).to(device)
    torch.nn.init.normal_(model.bias.data,0,0)
    torch.nn.init.normal_(model.weight.data,0,0)
    optimizer = optim.Adam(model.parameters(), lr=eta)
    loss_fn = nn.MSELoss()

    # Run training
    # niter = 10
    for i in tqdm(range(0, num_iters)):
                
        optimizer.zero_grad()
        
        rownorms = torch.sqrt(torch.sum(model.weight.data**2,dim=1))
        outsideball = rownorms > norm_bound
        model.weight.data[outsideball,:] = model.weight.data[outsideball,:] / rownorms[outsideball].view(-1,1) * norm_bound
        if iter_log > -1 and i % iter_log == 0:
            with torch.no_grad():
                mse_loss = torch.sum(torch.square(fns - model(rep)),dim=0) / B
                print('Train',mse_loss)
                print('Train', min(mse_loss).item(), max(mse_loss).item())
                if valrep is not None: 
                    mse_loss = torch.sum(torch.square(valfns - model(valrep)),dim=0) / valfns.shape[0]
                    print('Test',mse_loss)
                    print('Test', min(mse_loss).item(), max(mse_loss).item())
        
        predictions = model(rep)
        loss = loss_fn(predictions, fns)
        loss.backward()
        optimizer.step()
    if valrep is not None:
        test_mse = torch.sum(torch.square(valfns - model(valrep)),dim=0) / valfns.shape[0]
        train_mse =  torch.sum(torch.square(model(rep) - fns),dim=0) / B
        return train_mse, test_mse
    else:
        return torch.sum(torch.square(model(rep) - fns),dim=0) / B
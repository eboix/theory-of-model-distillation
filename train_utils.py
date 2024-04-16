import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda")

####### DATASET definitions

class DatasetFromFuncBinary(IterableDataset):

    def __init__(self, n, eval_fn):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn

    def __iter__(self):
        while True:
            x = [random.randint(0,1)*2-1 for i in range(self.n)]
            x = torch.FloatTensor(x).to(device)
            
            curreval = self.eval_fn(x)
            if type(curreval) == torch.Tensor:
                yield x, curreval
            else:
                y = torch.FloatTensor([curreval])
                yield x, y
            
    def __len__(self):
        assert(False)
        
def get_random_data_unif_binary(d,num_samples):
    x = torch.sign(torch.randn(num_samples,d))
    return x

class ERMDatasetFromFuncBinary():

    def __init__(self, n, eval_fn, erm_num_samples,batch=1):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn


        self.erm_num_samples = erm_num_samples
        self.counter = 0
        self.xs = []
        self.ys = []

        if batch == 1:
            fn_dataset = DatasetFromFuncBinary(n, eval_fn)
            dataloader = DataLoader(fn_dataset, batch_size=erm_num_samples, num_workers=0)
            dataloaderiter = iter(dataloader)

            data = next(dataloaderiter)
            self.xs, self.ys = data
            self.xs.to(device)
            self.ys.to(device)
        elif batch > 1:
            tot_samples = 0
            while tot_samples < erm_num_samples:
                new_samples = min(batch,erm_num_samples - tot_samples)
                tot_samples += new_samples
                newxs = get_random_data_unif_binary(self.n, new_samples).to(device)
                newys = eval_fn(newxs)
                self.xs.append(newxs.to('cpu'))
                self.ys.append(newys.to('cpu'))
            self.xs = torch.cat(self.xs)
            self.ys = torch.cat(self.ys)
            if len(self.ys.shape) == 1:
                self.ys = self.ys.view(-1,1)
            # print(self.xs.shape)
            # print(self.ys.shape)
        else:
            assert(False)
        
    def __getitem__(self, index):
        return self.xs[index,:], self.ys[index,:]
        
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.erm_num_samples
    
class DatasetFromFuncGaussian(IterableDataset):

    def __init__(self, n, eval_fn):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn

    def __iter__(self):
        while True:
            x = torch.randn(self.n)
            yield x, torch.FloatTensor([self.eval_fn(x)])
    
class ERMDatasetFromFuncGaussian():

    def __init__(self, n, eval_fn, erm_num_samples):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn


        self.erm_num_samples = erm_num_samples
        self.counter = 0
        self.xs = []
        self.ys = []

        fn_dataset = DatasetFromFuncGaussian(n, eval_fn)
        dataloader = DataLoader(fn_dataset, batch_size=erm_num_samples, num_workers=0)
        dataloaderiter = iter(dataloader)

        data = next(dataloaderiter)
        self.xs, self.ys = data
        
    def __getitem__(self, index):
        return self.xs[index,:], self.ys[index,:]
        
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.erm_num_samples
    

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
    
       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
       
       """
    val_offset = int(len(dataset)*(1-val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset)-val_offset)



######### TRAIN/TEST methods

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch {}: Accuracy {:.1f}% ({}/{}). Loss [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,correct/total*100,correct,total,  batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output,target)
            test_loss += loss.item()  # sum up batch loss
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy {:.1f}% ({:d}/{:d}), Average loss: {:.4f}\n'.format(correct/total*100,correct,total,test_loss))

    
def train_mse(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break
                
def train_mse_online(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batch_idx = 0
    num_batches = args['log_interval']
    for (data, target) in train_loader:
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == num_batches:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_batches * len(data),
                100. * batch_idx / num_batches, loss.item()))
            if args['dry_run']:
                break
            return
        
def train_class_erm(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = (torch.sign(target)+1)/2
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break
                
        
def train_class_online(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batch_idx = 0
    num_batches = args['log_interval']
    for (data, target) in train_loader:
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        target = (torch.sign(target)+1)/2
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == num_batches:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_batches * len(data),
                100. * batch_idx / num_batches, loss.item()))
            if args['dry_run']:
                break
            return
        
        

def test_class(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = (torch.sign(target)+1)/2
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            predicted = (torch.sign(output.data)+1)/2
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy {:.1f}% ({:d}/{:d}), Average loss: {:.4f}\n'.format(correct/total*100,correct,total,test_loss))
    return test_loss

def test_class_acc(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = (torch.sign(target)+1)/2
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            predicted = (torch.sign(output.data)+1)/2
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy {:.1f}% ({:d}/{:d}), Average loss: {:.4f}\n'.format(correct/total*100,correct,total,test_loss))
    return test_loss, correct/total*100

def test_mse(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss
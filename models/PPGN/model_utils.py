import random
import torch_geometric as tg
import torch
import torch.nn.functional as F
from random import shuffle
from utils import cross_val_generator
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_parameters = [0.00005, 0.0001, 0.0005, 0.001]
decay_parameters = [0.5, 1]

def epoch_train(model, train_set, optimizer):
    model.train()
    loader = tg.data.DataLoader(train_set, batch_size=1, shuffle=True)
    loss_sum = 0
    count = 0
    for X, y in loader:

        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        out = model(X)
        loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()

        optimizer.step()
        loss_sum += loss.item()
        count += 1
    #return loss normalised for number of batches
    return (loss_sum/count)

def train(model, train_set, num_epochs, optimizer, verbose=False):
    for epoch in range(num_epochs):
        epoch_loss = epoch_train(model, train_set, optimizer)
        if verbose:
            print("Epoch: " +str(epoch) + " Loss: " + str(epoch_loss))

def parameter_search(model, num_epochs, dataset, verbose=False):
    #split data into training and validation sets:
    shuffled_dataset = dataset
    shuffle(shuffled_dataset)
    split_point = len(dataset) // 9
    train_set = shuffled_dataset[:split_point]
    validation_set = shuffled_dataset[split_point:]

    #results stored in a dictionary indexed by parameters
    results_dict = dict()

    #train & validate model on each combination of parameters
    for lr in lr_parameters:
        results_dict[lr] = dict()
        for decay in decay_parameters:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay, step_size=20, last_epoch=-1, verbose=False)
            #train model on those params
            print("Training: start - lr: " + str(lr) + ' decay: '+str(decay))
            for epoch in range(num_epochs):
                epoch_loss = epoch_train(model, train_set, optimizer)
                if verbose:
                    print("Epoch: " + str(epoch) + " Loss: " + str(epoch_loss))
            print("End-training")

            #validate model on those params
            accuracy = test(model, validation_set)
            results_dict[lr][decay] = accuracy
    return results_dict

def test(model, test_set):
    correct = 0
    total = 0
    #basic testing: check if out matches y label
    for X, y in test_set:
        X = X.to(device)
        out = model(torch.unsqueeze(X, 0))
        #if ((data.y[0].item() == 1 and out[0].item() > 0.0)
        #    or (data.y[0].item() == -1 and out[0].item() <= 0.0)):
        if int(torch.argmax(out, 1)) == int(y):
            correct += 1
        total += 1
    return(correct/total)

def CV_10(model, dataset, config):
    #Partition dataset into 10 sets/chunks for Cross-Validation
    num_epochs = config['epochs']
    lr = config['lr']
    print_freq = config['print_freq']
    num_parts = 10
    accuracy_sum = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    #For each partition:
    for test_idx, (train_chunks, test_chunk) in enumerate(cross_val_generator(dataset, num_parts)):
        #Train Model
        print(f'\nTraining using test chunk {test_idx+1}/{num_parts}')
        for epoch in range(1, num_epochs + 1):
            print(f'epoch: {epoch}/{num_epochs}')
            loss = epoch_train(model, train_chunks,  optimizer)
            #display results as the model is training
            if epoch % print_freq == 0:
                print('accuracy:', test(model, test_chunk))
                print('loss:', loss)

        #Test Model
        accuracy_sum += test(model, test_chunk)
    return(accuracy_sum / 10)

"""def partition(dataset, num_parts):
    N = len(dataset)
    part_size = N // num_parts
    mod = N % num_parts
    partitions = []
    count = 0
    part_start = 0
    while count < mod:
        part_end = part_start + part_size + 1
        partitions.append(dataset[part_start:part_end])
        part_start = part_end
        count += 1
    while count < num_parts:
        part_end = part_start + part_size
        partitions.append(dataset[part_start:part_end])
        part_start = part_end
        count += 1
    
    return partitions"""
    

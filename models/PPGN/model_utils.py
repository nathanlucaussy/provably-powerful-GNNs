import torch_geometric as tg
import torch
import torch.nn.functional as F
from random import sample
from utils import cross_val_generator
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_parameters = [0.00005, 0.0001, 0.0005, 0.001]
decay_parameters = [0.5, 1]

def epoch_train(model, train_loader, optimizer, scheduler):
    model.train()
    loss_sum = 0
    count = 0
    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        out = model(X)
        loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()

        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
        count += 1
    #return loss normalised for number of batches
    return (loss_sum/count)

def validate(model, val_set):
    model.eval()
    

def train(model, train_set, num_epochs, optimizer, verbose=False):
    for epoch in range(num_epochs):
        epoch_loss = epoch_train(model, train_set, optimizer)
        if verbose:
            print("Epoch: " +str(epoch) + " Loss: " + str(epoch_loss))

def param_search(model_class, dataset, config):
    model = model_class(config).to(device)
    #split data into training and validation sets:
    shuffled_dataset = sample(list(dataset), len(dataset))
    #shuffle(shuffled_dataset)
    split_point = len(dataset) // 9
    train_set = shuffled_dataset[:split_point]
    val_set = shuffled_dataset[split_point:]
    train_loader = tg.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    best_params = (None, None)
    best_acc = 0

    #train & validate model on each combination of parameters
    for lr in lr_parameters:
        for decay in decay_parameters:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.decay, step_size=20)
            #train model on those params
            print(f'Training: start - lr: {lr}, decay: {decay}')
            for epoch in range(config.epochs):
                epoch_loss = epoch_train(model, train_loader, optimizer, scheduler)                
                if config.verbose:
                    print(f'Epoch: {epoch}, Loss: {epoch_loss}')

            #validate model on those params
            accuracy = test(model, val_set)
            print(f'Achieved test accuracy of {accuracy} with lr={lr}, decay={decay}')
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = (lr, decay)
    return *best_params, best_acc

def test(model, test_set):
    with torch.no_grad():
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

def CV_10(model_class, dataset, config):
    #Partition dataset into 10 sets/chunks for Cross-Validation
    num_epochs = config.epochs
    print_freq = config.print_freq
    num_parts = 10
    accuracy_sum = 0
    model = model_class(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.decay, step_size=20)

    #For each partition:
    for test_idx, (train_chunks, test_chunk) in enumerate(cross_val_generator(dataset, num_parts)):
        #Train Model
        train_loader = tg.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        print(f'\nTraining using test chunk {test_idx+1}/{num_parts}')
        for epoch in range(1, num_epochs + 1):
            print(f'epoch: {epoch}/{num_epochs}')
            loss = epoch_train(model, train_chunks, optimizer, scheduler)
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
    

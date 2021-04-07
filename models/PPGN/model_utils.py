import torch_geometric as tg
import torch
import torch.nn.functional as F
from utils import cross_val_generator, mean_and_std
from .helper import get_batches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_parameters = [0.00005, 0.0001, 0.0005, 0.001]
decay_parameters = [0.5, 1]

def epoch_train(model, train_batches, optimizer, scheduler, regression=False, mean=None, std=None):
    model.train()
    loss_sum = 0
    count = 0
    for X, y in train_batches:

        X = torch.stack(X).to(device)

        # normalize y if mean and std were given
        if regression:
            y = torch.cat(y)
        else:
            y = torch.tensor(y)
        if mean is not None and std is not None:
            y = (y - mean) / std
        y = y.to(device)
        optimizer.zero_grad()
        
        out = model(X)
        if regression:
            differences = (out-y).abs().sum(dim=0)
            loss = differences.sum()
        else:
            loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()

        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
        count = count + len(y)
    #return loss normalised for number of graphs
    return (loss_sum/count)

def param_search(model_class, dataset, config):
    model = model_class(config).to(device)
    #split data into training and validation sets:
    dataset.shuffle()
    split_point = len(dataset) // 9
    train_set = dataset[split_point:]
    val_set = dataset[:split_point]
    train_batches = get_batches(train_set, config.batch_size)
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
                epoch_loss = epoch_train(model, train_batches, optimizer, scheduler)                
                if config.verbose:
                    print(f'Epoch: {epoch}, Loss: {epoch_loss}')

            #validate model on those params
            accuracy = test(model, val_set)
            print(f'Achieved test accuracy of {accuracy} with lr={lr}, decay={decay}')
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = (lr, decay)
    return *best_params, best_acc

def test(model, test_set, regression=False, mean=None, std=None):
    with torch.no_grad():
        correct = 0
        total = 0
        errors = 0
        #basic testing: check if out matches y label
        for X, y in test_set:
            X = X.to(device)
    
            out = model(torch.unsqueeze(X, 0)).cpu()
            if mean is not None and std is not None:
                out = out * std + mean
    
            #if ((data.y[0].item() == 1 and out[0].item() > 0.0)
            #    or (data.y[0].item() == -1 and out[0].item() <= 0.0)):
            if regression:
                errors += (out-y).abs().sum(dim=0).detach().numpy()
            else:
                if int(torch.argmax(out, 1)) == int(y):
                    correct += 1
            total += 1
                
        if regression:
            return (errors / total)
        else:
            return(correct / total)

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
        train_batches = get_batches(train_chunks, config.batch_size)
        #train_loader = tg.data.DataLoader(train_chunks, batch_size=config.batch_size, shuffle=True)
        print(f'\nTraining using test chunk {test_idx+1}/{num_parts}')
        for epoch in range(1, num_epochs + 1):
            print(f'epoch: {epoch}/{num_epochs}')
            loss = epoch_train(model, train_batches, optimizer, scheduler, regression=config.qm9)
            #display results as the model is training
            if epoch % print_freq == 0:
                print('accuracy:', test(model, test_chunk, config.qm9))
                print('loss:', loss)

        #Test Model
        accuracy_sum += test(model, test_chunk, config.qm9)
    return(accuracy_sum / 10)


def CV_regression(model_class, dataset, config):
    num_epochs = config.epochs
    lr = config.lr
    print_freq = config.print_freq
    model = model_class(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.decay, step_size=20)

    #Partition dataset into train / test set
    dataset.shuffle()
    len_test_set = int(len(dataset) / 10)
    test_set = dataset[:len_test_set]
    train_set = dataset[len_test_set:]
    
    print('Calculating mean and std of train data')
    train_labels_mean, train_labels_std = mean_and_std(train_set)
    print('Sorting and grouping data into batches')
    train_batches = get_batches(train_set, config.batch_size)
                            
    for epoch in range(1, num_epochs + 1):
        print(f'epoch: {epoch}/{num_epochs}')
        loss = epoch_train(model, train_batches, optimizer, scheduler,
                           regression=True, mean=train_labels_mean, std=train_labels_std)
        #display results as the model is training
        if epoch % print_freq == 0:
            print('error:', test(model, test_set, regression=True, 
                                 mean=train_labels_mean, std=train_labels_std))
            print('loss:', loss)

    #Test Model
    test_error = test(model, test_set, regression=True, 
                      mean=train_labels_mean, std=train_labels_std)
    return test_error


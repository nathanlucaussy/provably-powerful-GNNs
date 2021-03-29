import random
import torch_geometric as tg
import torch
from random import shuffle
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_parameters = [0.00005, 0.0001, 0.0005, 0.001]
decay_parameters = [0.5, 1]

def epoch_train(model, train_set, optimizer):
    model.train()
    loader = tg.data.DataLoader(train_set, 1)
    loss_sum = 0
    count = 0
    for data in loader:

        x = data.x

        data = data.to(device)
        optimizer.zero_grad()

        loss_func = torch.nn.BCELoss()

        out = model(data)
        loss = loss_func(out, data.y)

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
    for data in test_set:
        out = model(data)
        if ((data.y[0].item() == 1 and out[0].item() > 0.0)
            or (data.y[0].item() == -1 and out[0].item() <= 0.0)):
            correct += 1
        total += 1
    return(correct/total)

def CV_10(model, dataset, num_epochs):
    #Partition dataset into 10 sets/chunks for Cross-Validation
    increment = len(dataset) // 10
    CV_chunks = [dataset[i*increment: (i*increment)+increment] for i in range(9)]
    CV_chunks += dataset[9*increment:]
    accuracy_sum = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    #For each partition:
    for test_chunk_index in range(len(CV_chunks)):
        #build up the training and testing sets for CV (train_set is all sets)
        train_chunks = []
        for index in range(len(CV_chunks)):
            if index == test_chunk_index:
                test_chunk = CV_chunks[index]
            else:
                train_chunks += CV_chunks[index]
        random.shuffle(train_chunks)

        #Train Model
        for epoch in range(num_epochs):
            print('epoch', epoch)
            loss = epoch_train(model, train_chunks,  optimizer)
            #display results as the model is training
            if epoch % (num_epochs // 30) == 0:
                print('accuracy', test(model, test_chunk))
                print('loss', loss)

        #Test Model
        accuracy_sum += test(model, test_chunk)
    return(accuracy_sum/5)

import torch.utils.data
import model
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import torch

def final_cosine_score(G, P):
    return 1 - cosine(G, P)

def train(dataset_train, dataset_test, sentiment_value_test, word_vectors):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=512)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=len(sentiment_value_test))

    # MODEL INITIALIZATION #
    model1 = model.SentimentAnalysisModel(300, 128, word_vectors).to(device)

    # OPTIMIZER #
    optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4)

    # LOSS-FUNCTION #
    criterion = torch.nn.MSELoss() 

    # TRAINING #
    losses = []
    train_results = []
    test_results = []
    for epoch in range(1,101):
        epoch_losses = []
        for i, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            x, y = batch
            y = torch.tensor(y, dtype=torch.float).to(device)
            #print("i=",i)
            #print(batch)
            #print(len(batch))
            y_pred = model1(x)
            
            loss = criterion(y_pred.view(-1) , y).to(device) # to scale sigmoid output from [0,1] to [-1,1] alt: 2*y_pred -1
    #         plt.plot(y_pred.view(-1).detach().numpy(),y.view(-1).numpy(),'r.')
            #score = final_cosine_score(y.view(-1).numpy(), y_pred.view(-1).detach().numpy())
            score = final_cosine_score(y.view(-1).cpu().numpy(), y_pred.view(-1).detach().cpu().numpy())
            epoch_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        #train_results.append(score)
        #for i, batch in enumerate(dataloader_test):
            #y_pred = model(batch[0])
            #test_results.append(final_cosine_score(batch[1].view(-1),y_pred.view(-1).detach().numpy()))
        train_results.append(score)
        #epoch_losses = [epoch_loss.cpu().detach().numpy() for epoch_loss in epoch_losses]
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        for i, batch in enumerate(dataloader_test):
            batch[0] = batch[0].to(device)
            y_pred = model1(batch[0])
            test_results.append(final_cosine_score(batch[1].view(-1),y_pred.view(-1).detach().cpu().numpy()))
        if epoch%5==0:
            print('Epoch {}, loss {:.4f}, score_train {:.4f}, score_train {:.4f}'.format(epoch, epoch_loss, score, test_results[-1]))
    plt.plot(losses,'b', label ="MSE training loss")
    plt.plot(train_results,'g', label ="training score")
    plt.plot(losses,'r', label ="Cosine training loss")
    plt.legend()
    plt.show()
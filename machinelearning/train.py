from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        hasError = True
        while hasError:
            enteredNest = False
            for i in dataloader:
                curr = (model.get_prediction(model(i.get('x'))), i.get('label'))
                if not curr[0] == curr[1]:
                    enteredNest = True
                    model.update_weights(curr[0], i.get('x'))
            if not enteredNest:
                hasError = False
            

def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #print(len(dataset)) #200 examples
    dataloader = DataLoader(dataset, batch_size=model.batchSize, shuffle=True)

    currLoss = float("inf")

    while currLoss > model.goalLoss:
        batchCont = 0
        totalLoss = 0
        for batch in dataloader:
            x = batch.get("x")
            y = batch.get("label")

            #used recommended website: https://pytorch.org/docs/stable/optim.html
            optimizer.zero_grad()
            predicted_y = model(x)
            lossTensor = regression_loss(predicted_y, y)
            lossTensor.backward()
            optimizer.step()

            totalLoss += lossTensor.data
            batchCont += 1

        currLoss = totalLoss / batchCont


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    currLoss = float("inf")

    while currLoss > model.goalLoss:
        dataloader = DataLoader(dataset, batch_size=model.batchSize, shuffle=True)
        total_loss = 0.0
        batchCount = 0

        for batch in dataloader:
            x = batch.get("x")
            y = batch.get("label")

            optimizer.zero_grad()
            score = model(x)
            loss = digitclassifier_loss(score, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batchCount += 1

        currLoss = total_loss / batchCount


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """

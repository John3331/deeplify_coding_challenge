
# %%
from datetime import datetime
from typing import Iterator
from tqdm import tqdm
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from src.utils import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the paths to the train and test data folders
DATA_DIR = r'data'


def train(model: nn.Module, 
          params_to_train: Iterator[torch.nn.parameter.Parameter], 
          epochs: int, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          learning_rate:float =0.001):
    """Takes an initialized model and does the whole training process. 
    It logs train and validation metrics with tensorboard and saves the model in each epoch.
    The training loop is purposfully not split into subfunction to have a better overview over what 
    is happening in the core of the script. 

    Args:
        model (nn.Module): Model with initialized weights ready to train
        params_to_train (nn.Module): parameters of the model that should be optimized
        epochs (int): number of epochs to train
    """

    model.to(device)
    pos_weight = calc_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)

    run_name = f'{model.name},lr={round(learning_rate, 5)},EP={epochs},BS={train_loader.batch_size},' + \
               f'{datetime.now().strftime("%b%d_%H-%M-%S")}'
    run_folder = 'runs/' + run_name
    writer = SummaryWriter(run_folder)

    accuracy_metric = BinaryAccuracy().to(device)
    f1_score_metric = BinaryF1Score().to(device)

    best_val_f1 = 0

    for epoch in range(epochs):  
        model.train()
        running_loss = 0.0
        
        f1_score_metric.reset() 
        accuracy_metric.reset()

        print("---------------------")
        print('epoch', epoch)

        for i, data in enumerate(tqdm(train_loader)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if epoch == 0 and i == 0: writer.add_graph(model, inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bin_outputs = torch.round(torch.sigmoid(outputs))

            accuracy_metric.update(bin_outputs, labels)
            f1_score_metric.update(bin_outputs, labels)

        
        train_loss = running_loss/samples_per_epoch(train_loader)
        train_f1 = f1_score_metric.compute()
        accuracy = accuracy_metric.compute()
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/F1-Score', train_f1, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        print(train_loss, train_f1.item(), accuracy.item())

        if val_loader:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                epoch_labels, epoch_predictions = [], []
                
                f1_score_metric.reset() 
                accuracy_metric.reset()

                for data in tqdm(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs).reshape(-1)
                    loss = criterion(outputs, labels.float())
                    outputs = torch.sigmoid(outputs)
                    
                    val_loss += loss.item()
                    bin_outputs = torch.round(outputs)

                    accuracy_metric.update(bin_outputs, labels)
                    f1_score_metric.update(bin_outputs, labels)
                    epoch_labels.append(labels)
                    epoch_predictions.append(outputs)

                val_len = samples_per_epoch(val_loader)
                val_f1 = cal_and_log_val_metrics(writer, val_loss, val_len, f1_score_metric, 
                                                 accuracy_metric, epoch, epoch_predictions, epoch_labels)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                
            # save the model in each epoch to choose the best generalizing model in the end
            save_model(model, folder_path='models/' + run_name, file_name=f'/epoch{epoch}.pth')
        # Log hyperparameters to TensorBoard
        if val_loader:
            hparams = {
                'model': model.name,
                'lr': learning_rate,
                'epochs': epochs,
                'batch_size': train_loader.batch_size
                }
            writer.add_hparams(hparams, {'best_val_f1': best_val_f1})

    writer.close()



# Training routines -------------------


def simple_training_run():
    efficientnet = create_efficientnet()

    
    train_loader, val_loader, _ = create_dataloader(batch_size=16, data_dir=DATA_DIR)

    train(efficientnet, 
          efficientnet.classifier.fc.parameters(), 
          epochs=4, 
          train_loader=train_loader, 
          val_loader=val_loader, 
          learning_rate=0.001)


def final_train_run_on_all_data():
    efficientnet = create_efficientnet()
    train_loader, _, _ = create_dataloader(batch_size=32, data_dir=DATA_DIR, validation=False)

    train(efficientnet,
          efficientnet.classifier.fc.parameters(),
          epochs=20,
          train_loader=train_loader,
          val_loader=None,
          learning_rate=0.00085468)


def random_search(runs=15):
    search_config = {
        "lr": tune.loguniform(7e-5, 4e-3),
        "batch_size": tune.choice([32, 64]),
    }
    
    for _ in range(runs):
        lr = search_config['lr'].sample()
        batch_size = search_config['batch_size'].sample()

        resnet = create_resnet()
        resnet_params_to_train = resnet.fc.parameters()
        efficientnet = create_efficientnet()
        efficientnet_params_to_train = efficientnet.classifier.fc.parameters()
        for model, params_to_train in zip([resnet, efficientnet], [resnet_params_to_train, efficientnet_params_to_train]):
            train_loader, val_loader, _ = create_dataloader(batch_size=batch_size, data_dir=DATA_DIR)
            train(model, 
                  params_to_train, 
                  epochs=25, 
                  train_loader=train_loader, 
                  val_loader=val_loader, 
                  learning_rate=lr)


def random_search2(runs=8):
    search_config = {
        "lr": tune.loguniform(5e-4, 1e-2),
        "batch_size": tune.choice([32, 64]),
    }
    
    for _ in range(runs):
        lr = search_config['lr'].sample()
        batch_size = search_config['batch_size'].sample()

        efficientnet = create_efficientnet()
        efficientnet_params_to_train = efficientnet.classifier.fc.parameters()
        
        train_loader, val_loader, _ = create_dataloader(batch_size=batch_size, data_dir=DATA_DIR)
        train(efficientnet, 
              efficientnet_params_to_train, 
              epochs=35,
              train_loader=train_loader, 
              val_loader=val_loader, 
              learning_rate=lr)
        

def selected_runs():
    search_config = {
        "lr": [0.003, 0.003],
        "batch_size": [32, 64],
    }
    
    for lr, batch_size in zip(search_config['lr'], search_config['batch_size']):

        efficientnet = create_efficientnet()
        efficientnet_params_to_train = efficientnet.classifier.fc.parameters()
        
        train_loader, val_loader, _ = create_dataloader(batch_size=batch_size, data_dir=DATA_DIR)
        train(efficientnet, 
              efficientnet_params_to_train, 
              epochs=30,
              train_loader=train_loader, 
              val_loader=val_loader, 
              learning_rate=lr)


if __name__ == '__main__':
    # random_search()    
    # final_train_run_on_all_data()
    # random_search2()
    selected_runs()



# %%

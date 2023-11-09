import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import transforms, datasets
from torchmetrics.utilities.data import dim_zero_cat


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def create_transforms():
    # Define a transform to preprocess the images (resize and normalize without standardization)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize the images to a fixed size
        transforms.RandomRotation((-10, 10)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the images (mean and std for grayscale images)
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize the images to a fixed size
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the images (mean and std for grayscale images)
    ])

    return train_transform, val_test_transform


def create_dataloader(batch_size, data_dir, validation=True):
    train_transform, val_test_transform = create_transforms()

    # Create datasets for train and test
    train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=data_dir + '/test', transform=val_test_transform)

    train_indices, val_indices = train_test_split(torch.arange(len(train_dataset)), test_size=0.15, random_state=42)
    if validation: train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_resnet():
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    resnet.name = 'resnet18'

    # change the first layer to grayscale instead of RGB
    state_dict2 = resnet.state_dict()
    state_dict2['conv1.weight'] = state_dict2['conv1.weight'].sum(dim=1, keepdim=True)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.load_state_dict(state_dict2)

    # Freeze all the pre-trained layers
    for param in resnet.parameters():
        param.requires_grad = False

    # add a new classification head
    resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=128, out_features=1),
        # nn.Sigmoid(), included in the loss
    )

    return resnet


def create_efficientnet():
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    efficientnet.name = 'effnet'

    # change the first layer to grayscale instead of RGB
    effnet_state_dict = efficientnet.state_dict()
    effnet_state_dict['stem.conv.weight'] = effnet_state_dict['stem.conv.weight'].sum(dim=1, keepdim=True)
    efficientnet.stem.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    efficientnet.load_state_dict(effnet_state_dict)

    # Freeze all the pre-trained layers
    for param in efficientnet.parameters():
        param.requires_grad = False

    # add new classification head
    efficientnet.classifier.fc = nn.Sequential(
        nn.Linear(in_features=1280, out_features=256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=128, out_features=1),
        # nn.Sigmoid(), included in the loss
    )

    return efficientnet


def create_confusion_matrix_plot(tp, fp, tn, fn):
    df_cm = pd.DataFrame(torch.tensor([[tp, fp], [fn, tn]]), index=['normal', 'pneumonia'],
                         columns=['normal', 'pneumonia'])
    plt.figure(figsize=(12, 7))    
    return sns.heatmap(df_cm, annot=True).get_figure()


def calc_pos_weight(train_loader):
    # calculates the correct pos_weight parameter to give the majority class a lower weight
    if type(train_loader.dataset) == torch.utils.data.dataset.Subset:
        targets = torch.tensor(train_loader.dataset.dataset.targets)
        targets = targets[train_loader.dataset.indices]
    else:
        targets = torch.tensor(train_loader.dataset.targets)
    
    num_pos = targets.sum()
    num_neg = - (targets-1).sum()
    return num_neg / num_pos


def cal_and_log_val_metrics(writer, val_loss, len_val, f1_score_metric, accuracy_metric, epoch, epoch_predictions, epoch_labels):
    # calculate validation metrcs
    val_loss = val_loss / len_val
    tp, fp, tn, fn = f1_score_metric.tp, f1_score_metric.fp, f1_score_metric.tn, f1_score_metric.fn
    val_f1 = f1_score_metric.compute()
    val_acc = accuracy_metric.compute()

    # Log validation metrics to TensorBoard
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    writer.add_scalar('Validation/F1-Score', val_f1, epoch)
    writer.add_scalar('Validation/Accuracy', val_acc, epoch)
    writer.add_pr_curve('pr_curve', dim_zero_cat(epoch_labels), dim_zero_cat(epoch_predictions), epoch)
    writer.add_figure("Confusion matrix", create_confusion_matrix_plot(tp, fp, tn, fn), epoch)

    return val_f1


def samples_per_epoch(data_loader: torch.utils.data.DataLoader):
    # calculates the samples per epoch of a dataloader

    if type(data_loader.dataset) == torch.utils.data.dataset.Subset:
        num_samples = len(data_loader.dataset.indices)
    else:
        num_samples = len(data_loader.dataset)
    return num_samples


def save_model(model, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(model, folder_path + file_name)
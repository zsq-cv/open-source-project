# import packages
import copy
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from Multi_Network_77_5_67_5 import Net
from Multi_Get_Dataset import MyDataset

# set constants
FILE_DIR = {'train': '../Dataset/train/', 'val': '../Dataset/val/'}
ANNO_FILE = {'train': 'Multi_train_annotation.csv', 'val': 'Multi_val_annotation.csv'}
NORM_PARAMS = 'Multi_Norm_Params.csv'
CLASSES = ['birds', 'mammals']
SPECIES = ['chickens', 'rabbits', 'rats']


#################################### Load data ####################################
# Data augmentation and normalization for training, just normalization for validation
norm_params = pd.read_csv(NORM_PARAMS, index_col=0)

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(500),
        transforms.Resize((500, 500)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(eval(norm_params['train']['means']),
        #                     eval(norm_params['train']['stds']))
    ]),
    'val': transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        #transforms.Normalize(eval(norm_params['val']['means']),
        #                     eval(norm_params['val']['stds']))
    ])
}

datasets = {x: MyDataset(file_dir=FILE_DIR[x],
                         anno_file=ANNO_FILE[x],
                         transform=data_transforms[x])
            for x in ['train', 'val']}

train_loader = DataLoader(dataset=datasets['train'], batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=datasets['val'])
data_loaders = {'train': train_loader, 'val': test_loader}

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device:', device)

# define function show_random_data
def show_random_data():
    idx = random.randint(0, len(datasets['train']))
    sample = train_loader.dataset[idx]
    print('The ' + str(idx+1) + 'th sample: size is', sample['image'].shape,
           ', class is ' + CLASSES[sample['classes']] + ', species is ' + SPECIES[sample['species']] + '.')
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

# call function to visualize some data
show_random_data()


################################# Train the model #################################
# define function train_model
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    # initialize variables
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_list = {'train': [], 'val': []}
    acc_list_classes = {'train': [], 'val': []}
    acc_list_species = {'train': [], 'val': []}

    # loop train process for num_epoches times
    for epoch in range(num_epochs):
        print('=' * 28 + ' Epoch {}/{} '.format(epoch + 1, num_epochs) + '=' * 28)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # initialize variables
            running_loss = 0.0
            corrects_classes = 0
            corrects_species = 0

            # set model for different mode in different phases
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            # iterate over data batch
            for data in data_loaders[phase]:
                # get data
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                labels_species = data['species'].to(device)

                # clear the gradients in optimizer
                optimizer.zero_grad()

                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    # predict labels
                    x_classes, x_species = model(inputs)
                    _, preds_classes = torch.max(x_classes.view(-1, 2), 1)
                    _, preds_species = torch.max(x_species.view(-1, 3), 1)

                    # calculate loss for this batch
                    loss_classes = criterion(x_classes, labels_classes)
                    loss_species = criterion(x_species, labels_species)
                    loss = 0.6 * loss_classes + 0.4 * loss_species

                    # backward propagation and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # add loss and correct num of this batch to epoch variables
                running_loss += loss.item() * inputs.size(0)
                corrects_classes += torch.sum(preds_classes == labels_classes)
                corrects_species += torch.sum(preds_species == labels_species)

            # calculate loss and accurate for current epoch
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)
            epoch_acc = 0.5 * epoch_acc_classes + 0.5 * epoch_acc_species
            print('{} loss: {:.4f}, classes accurate: {:.2%}, species accurate: {:.2%}. '.
                    format(phase, epoch_loss, epoch_acc_classes, epoch_acc_species))

            # record loss and accurate in list
            loss_list[phase].append(epoch_loss)
            acc_list_classes[phase].append(100 * epoch_acc_classes)
            acc_list_species[phase].append(100 * epoch_acc_species)

            # scheduler move a step in training phase
            #if phase == 'train':
            #    scheduler.step()

            # evaluate model in validation phase
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                print('* Current best model accurate is {:.2%}. '.format(best_acc))

    # training end, save outcomes and return
    print('Best val accurate is {:.2%}. '.format(best_acc))
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best model saved as best_model.pt! ')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s. '.format(
        time_elapsed // 60, time_elapsed % 60))
    return model, loss_list, acc_list_classes, acc_list_species


# set params and call train_model
network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
num_epochs = 50
model, loss_list, acc_list_classes, acc_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs)

# draw training plots
x = [i for i in range(num_epochs)]
## draw loss plot
y1 = loss_list["val"]
y2 = loss_list["train"]
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all')
## draw accurate plot
y3 = acc_list_classes["train"]
y4 = acc_list_classes["val"]
y5 = acc_list_species["train"]
y6 = acc_list_species["val"]
plt.plot(x, y3, color="r", linestyle="-", marker=".", linewidth=1, label="classes_train")
plt.plot(x, y4, color="b", linestyle="-", marker=".", linewidth=1, label="classes_val")
plt.plot(x, y5, color="y", linestyle="-", marker=".", linewidth=1, label="species_train")
plt.plot(x, y6, color="g", linestyle="-", marker=".", linewidth=1, label="species_val")
plt.legend()
plt.title('train and val acc vs. epoches')
plt.xlabel('epochs')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val acc vs epoches.jpg")
plt.close('all')


########################## Visualize the model predictions ########################
# define function pred_via_model
def pred_via_model(model):
    # set model to evaluate mode
    model.eval()
    # forward propagation
    with torch.no_grad():
        # iterate over data batch
        for data in data_loaders['val']:
            # get data
            inputs = data['image'].to(device)
            labels_classes = data['classes'].to(device)
            labels_species = data['species'].to(device)
            # predict
            x_classes, x_species = model(inputs)
            _, preds_classes = torch.max(x_classes.view(-1, 2), 1)
            _, preds_species = torch.max(x_species.view(-1, 3), 1)
            # show results
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted class: {}, species: {} \n ground-truth class: {},species: {}'
                      .format(CLASSES[preds_classes], SPECIES[preds_species], CLASSES[labels_classes], SPECIES[labels_species]))
            plt.show()
    return


# call function to visualize model predictions
pred_via_model(model)

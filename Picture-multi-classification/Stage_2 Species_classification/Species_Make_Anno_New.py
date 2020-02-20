# import packages
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
# set constant values: ROOTS, PHASE, CLASSES, SPECIES
ROOTS = '../Dataset/'
PHASES = ['train', 'val']
CLASSES = ['birds', 'mammals']
SPECIES = ['chickens', 'rabbits', 'rats']
# make annotation
# # make data info dictionary
data_info = {'train': {'path': [], 'species': []},
             'val': {'path': [], 'species': []}}
norm_params = {'train': {'means': torch.tensor([0.0, 0.0, 0.0]), 'stds': torch.tensor([0.0, 0.0, 0.0])},
             'val': {'means': torch.tensor([0.0, 0.0, 0.0]), 'stds': torch.tensor([0.0, 0.0, 0.0])}}
transform = transforms.Compose([transforms.Resize(500), transforms.ToTensor()])
# # get file name
for p in PHASES:
    num = 0
    for s in SPECIES:
        path = ROOTS + p + '/' + s + '/'
        files = os.listdir(path)
# # for every file, if it can be opened, update info in data info dic
        for f in files:
            try:
                img = Image.open(path + f).convert('RGB')
                tensor = transform(img)
                for i in range(3):
                   norm_params[p]['means'][i] += tensor[i, :, :].mean()
                   norm_params[p]['stds'][i] += tensor[i, :, :].std()
            except OSError:
                pass
            else:
                num += 1
                data_info[p]['path'].append(path + f)
                if s == 'chickens':
                    data_info[p]['species'].append(0)
                elif s == 'rabbits':
                    data_info[p]['species'].append(1)
                else:
                    data_info[p]['species'].append(2)
# # trans data info into pandas data frame
    ANNOTATION = pd.DataFrame(data_info[p])
# # write data frame in csv file
    ANNOTATION.to_csv('Species_%s_annotation.csv' % p)
# # print message
    print('Annotation file Species_%s_annotation.csv is saved.' % p)
    norm_params[p]['means'] = norm_params[p]['means'] / num
    norm_params[p]['stds'] = norm_params[p]['stds'] / num
    norm_params[p]['means'] = norm_params[p]['means'].tolist()
    norm_params[p]['stds'] = norm_params[p]['stds'].tolist()
NORM_PARAMS = pd.DataFrame(norm_params)
NORM_PARAMS.to_csv('Species_Norm_Params.csv')
print('Normalization parameters file Species_Norm_Params.csv is saved.')

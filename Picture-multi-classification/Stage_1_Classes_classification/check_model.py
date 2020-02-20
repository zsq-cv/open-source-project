import torch

model = torch.load('best_model.pt')
for param_tensor in model:
    print(param_tensor, "\t", model[param_tensor].size())

state = {'net': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'exp_lr_scheduler': exp_lr_scheduler.state_dict(),
         'epoch': epoch}
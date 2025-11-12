from component.wR2 import wR2
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from load_data import *

numClasses = 4
imgSize = (480, 480)
resume_file = 'rpnet/result/wR2/wR2_0.pth0'
device_id = torch.device("cuda:1")
model_conv = wR2(numClasses)
# model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load(resume_file))
model_conv = model_conv.to(device=device_id)
val_dataset = ChaLocDataLoader('data/CCPD2019', imgSize, split='val')
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

def evaluate_model(model, valloader):
    model.eval()
    lossAver = []
    with torch.no_grad():
        for i, (XI, YI) in enumerate(tqdm(valloader, desc=f"Validation")):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            x = Variable(XI.to(device_id))
            y = Variable(torch.FloatTensor(YI).to(device_id), requires_grad=False)
            y_pred = model(x)
            loss = 0.0
            loss += 0.8 * nn.L1Loss().to(device_id)(y_pred[:, :2], y[:, :2])
            loss += 0.2 * nn.L1Loss().to(device_id)(y_pred[:, 2:], y[:, 2:])
            lossAver.append(loss.item())
    loss_avg = sum(lossAver) / len(lossAver)
    return loss_avg

loss_avg = evaluate_model(model_conv, valloader)
print(f'Validation Loss: {loss_avg}')
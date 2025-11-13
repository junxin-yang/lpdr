# from component.wR2 import wR2
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np
# from tqdm import tqdm
# from load_data import *

# numClasses = 4
# imgSize = (480, 480)
# resume_file = 'rpnet/result/wR2/wR2_0.pth0'
# device_id = torch.device("cuda:1")
# model_conv = wR2(numClasses)
# # model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
# model_conv.load_state_dict(torch.load(resume_file))
# model_conv = model_conv.to(device=device_id)
# val_dataset = ChaLocDataLoader('data/CCPD2019', imgSize, split='val')
# valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# def evaluate_model(model, valloader):
#     model.eval()
#     lossAver = []
#     with torch.no_grad():
#         for i, (XI, YI) in enumerate(tqdm(valloader, desc=f"Validation")):
#             # print('%s/%s %s' % (i, times, time()-start))
#             YI = np.array([el.numpy() for el in YI]).T
#             x = Variable(XI.to(device_id))
#             y = Variable(torch.FloatTensor(YI).to(device_id), requires_grad=False)
#             y_pred = model(x)
#             loss = 0.0
#             loss += 0.8 * nn.L1Loss().to(device_id)(y_pred[:, :2], y[:, :2])
#             loss += 0.2 * nn.L1Loss().to(device_id)(y_pred[:, 2:], y[:, 2:])
#             lossAver.append(loss.item())
#     loss_avg = sum(lossAver) / len(lossAver)
#     return loss_avg

# loss_avg = evaluate_model(model_conv, valloader)
# print(f'Validation Loss: {loss_avg}')


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from component.fh02 import fh02
from load_data import labelFpsDataLoader, labelTestDataLoader

device_id = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

imgDirs = 'data/CCPD2019'
numClasses = 7
numPoints = 4
classifyNum = 35
imgSize = (480, 480)
provNum, alphaNum, adNum = 38, 25, 35
val_dataset = labelTestDataLoader(imgDirs, imgSize, split='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
model_conv = fh02(num_points=numPoints, num_classes=numClasses, provNum=provNum, alphaNum=alphaNum, adNum=adNum, device_id=device_id)
# model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load("rpnet/result/fh02/fh02_0.pth"))
model_conv = model_conv.to(device=device_id)

def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare)

def eval(model, val_loader, epoch):
    model.eval()
    count, error, correct = 0, 0, 0
    with torch.no_grad():
        for i, (XI, labels, ims) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            count += 1
            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            x = Variable(XI.to(device_id))

            fps_pred, y_pred = model(x)

            outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
            labelPred = [t[0].index(max(t[0])) for t in outputY]

            #   compare YI, outputY
            try:
                if isEqual(labelPred, YI[0]) == 7:
                    correct += 1
                else:
                    pass
            except:
                error += 1
    return count, correct, error, float(correct) / count

count, correct, error, precision = eval(model_conv, val_loader, epoch=0)
print(f'Validation Precision: {precision} ({correct}/{count}), Errors: {error}')

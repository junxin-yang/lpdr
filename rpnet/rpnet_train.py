# Compared to fh0.py
# fh02.py remove the redundant ims in model input
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from torch.optim import lr_scheduler
from component.fh02 import fh02
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to the input file")
    ap.add_argument("-n", "--epochs", default=10000,
                    help="epochs for train")
    ap.add_argument("-b", "--batchsize", default=5,
                    help="batch size for train")
    ap.add_argument("-se", "--start_epoch", required=True,
                    help="start epoch for train")
    ap.add_argument("-r", "--resume", default='111',
                    help="file for re-train")
    ap.add_argument("-f", "--folder", required=True,
                    help="folder to store model")
    ap.add_argument("-w", "--writeFile", default='result/fh02.out',
                    help="file for output")
    ap.add_argument("-l", "--logDir", default='rpnet/result/runs/fh02',
                        help="directory for tensorboard logs")
    ap.add_argument("-p", "--wR2Path", default='rpnet/result/wR2/wR2.pth',
                        help="directory for tensorboard logs")
    return ap.parse_args()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


args = vars(get_args())

writer = SummaryWriter(log_dir=args["logDir"]) 

wR2Path = args["wR2Path"]
use_gpu = torch.cuda.is_available()
print(use_gpu)
torch.backends.cudnn.benchmark = True
if use_gpu:
    device_id = torch.device("cuda:1")
else:
    device_id = torch.device("cpu")

numClasses = 7
numPoints = 4
classifyNum = 35
imgSize = (480, 480)
# lpSize = (128, 64)
provNum, alphaNum, adNum = 38, 25, 35
batchSize = int(args["batchsize"]) if use_gpu else 2
imgDirs = args["images"]
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
storeName = modelFolder + 'fh02.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)

epochs = int(args["epochs"])
#   initialize the output file
if not os.path.isfile(args['writeFile']):
    with open(args['writeFile'], 'wb') as outF:
        pass


epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])
if not resume_file == '111':
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = fh02(num_points=numPoints, num_classes=numClasses, provNum=38, alphaNum=25, adNum=35, device_id='cpu')
    # model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.to(device=device_id)
else:
    model_conv = fh02(num_points=numPoints, num_classes=numClasses, wrPath=wR2Path, provNum=38, alphaNum=25, adNum=35, device_id=device_id)
    if use_gpu:
        # model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.to(device=device_id)

print(model_conv)
print(get_n_params(model_conv))

criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

train_dataset = labelFpsDataLoader(imgDirs, imgSize, split='train')
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=8)
val_dataset = labelTestDataLoader(imgDirs, imgSize, split='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_dataset = labelTestDataLoader(imgDirs, imgSize, split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)


def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


def eval(model, val_loader, epoch):
    model.eval()
    count, error, correct = 0, 0, 0
    start = time()
    with torch.no_grad():
        for i, (XI, labels, ims) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            count += 1
            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            if use_gpu:
                x = Variable(XI.to(device_id))
            else:
                x = Variable(XI)
            # Forward pass: Compute predicted y by passing x to the model

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
        writer.add_scalar('eval/precision', float(correct) / count, epoch)
    return count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            if not len(XI) == batchSize:
                continue

            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            Y = np.array([el.numpy() for el in Y]).T
            if use_gpu:
                x = Variable(XI.to(device_id))
                y = Variable(torch.FloatTensor(Y).to(device_id), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(Y), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            try:
                fps_pred, y_pred = model(x)
            except:
                continue

            # Compute and print loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().to(device_id)(fps_pred[:, :2], y[:, :2])
            loss += 0.2 * nn.L1Loss().to(device_id)(fps_pred[:, 2:], y[:, 2:])
            for j in range(7):
                l = Variable(torch.LongTensor([el[j] for el in YI]).to(device_id))
                loss += criterion(y_pred[j], l)

            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                lossAver.append(loss.data[0])
            except:
                pass

            if i % 50 == 1:
                with open(args['writeFile'], 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                torch.save(model.state_dict(), storeName)
        writer.add_scalar('train/lr', optimizer_conv.param_groups[0]['lr'], epoch)
        lrScheduler.step()
        writer.add_scalar('train/epoch_loss', sum(lossAver) / len(lossAver), epoch)
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        count, correct, error, precision, avgTime = eval(model, val_loader, epoch)
        with open(args['writeFile'], 'a') as outF:
            outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        torch.save(model.state_dict(), storeName + str(epoch))
    return model


if __name__ == '__main__':
    model_conv = train_model(model_conv, train_loader, val_loader, criterion, optimizer_conv, num_epochs=epochs)
    count, correct, error, precision, avgTime = eval(model_conv, val_loader)
    with open(args['writeFile'], 'a') as outF:
        outF.write('Final val: total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
# Code in cnn_fn_pytorch.py
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from component.wR2 import wR2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to the input file")
    ap.add_argument("-n", "--epochs", default=25,
                    help="epochs for train")
    ap.add_argument("-b", "--batchsize", default=4,
                    help="batch size for train")
    ap.add_argument("-r", "--resume", default='111',
                    help="file for re-train")
    ap.add_argument("-w", "--writeFile", default='rpnet/result/wR2.out',
                    help="file for output")
    ap.add_argument("-l", "--logDir", default='rpnet/result/runs/wR2',
                    help="directory for tensorboard logs")
    ap.add_argument("-m", "--modelFolder", default='rpnet/result/wR2',
                    help="folder to store model")
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

use_gpu = torch.cuda.is_available()
print(use_gpu)
torch.backends.cudnn.benchmark = True

if use_gpu:
    torch.cuda.set_device(1)
    device_id = torch.device("cuda:1")
else:
    device_id = torch.device("cpu")

numClasses = 4
imgSize = (480, 480)
batchSize = int(args["batchsize"]) if use_gpu else 8
modelFolder = args["modelFolder"]
os.makedirs(modelFolder, exist_ok=True)

epochs = int(args["epochs"])
#   initialize the output file
with open(args['writeFile'], 'wb') as outF:
    pass


epoch_start = 0
resume_file = str(args["resume"])
if not resume_file == '111':
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = wR2(numClasses)
    # model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file), map_location='cpu')
    model_conv = model_conv.to(device=device_id)
else:
    model_conv = wR2(numClasses)
    if use_gpu:
        # model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.to(device=device_id)

print(model_conv)
print(get_n_params(model_conv))

criterion = nn.MSELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# dst = LocDataLoader([args["images"]], imgSize)
train_dataset = ChaLocDataLoader(args["images"], imgSize, split='train')
val_dataset = ChaLocDataLoader(args["images"], imgSize, split='val')
trainloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
valloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=4)


def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=25):
    min_val_loss = 1e10
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        start = time()
        storeName = modelFolder + '/wR2_' + str(epoch) + '.pth'

        for i, (XI, YI) in enumerate(tqdm(trainloader, desc=f"Train Epoch {epoch}")):
            YI = np.array([el.numpy() for el in YI]).T
            if use_gpu:
                x = Variable(XI.to(device_id))
                y = Variable(torch.FloatTensor(YI).to(device_id), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss += 0.8 * nn.L1Loss().to(device_id)(y_pred[:, :2], y[:, :2])
                loss += 0.2 * nn.L1Loss().to(device_id)(y_pred[:, 2:], y[:, 2:])
                lossAver.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), epoch * len(trainloader) + i)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i % 50 == 1:
                with open(args['writeFile'], 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
        writer.add_scalar('train/lr', optimizer_conv.param_groups[0]['lr'], epoch)
        lrScheduler.step()
        writer.add_scalar('train/epoch_loss', sum(lossAver) / len(lossAver), epoch)
        evl_loss = evaluate_model(model, valloader, epoch)
        if evl_loss < min_val_loss:
            min_val_loss = evl_loss
            torch.save(model.state_dict(), modelFolder + '/wR2_best.pth')
        with open(args['writeFile'], 'a') as outF:
            outF.write('Epoch: %s, train_loss: %s, val_loss: %s, time:%s\n' % (epoch, sum(lossAver) / len(lossAver), evl_loss, time()-start))
        torch.save(model.state_dict(), storeName)
    writer.close()
    return model


def evaluate_model(model, valloader, epoch):
    model.eval()
    lossAver = []
    with torch.no_grad():
        for i, (XI, YI) in enumerate(tqdm(valloader, desc=f"Validation Epoch {epoch}")):
            YI = np.array([el.numpy() for el in YI]).T
            if use_gpu:
                x = Variable(XI.to(device_id))
                y = Variable(torch.FloatTensor(YI).to(device_id), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
            loss = 0.0
            loss += 0.8 * nn.L1Loss().to(device_id)(y_pred[:, :2], y[:, :2])
            loss += 0.2 * nn.L1Loss().to(device_id)(y_pred[:, 2:], y[:, 2:])
            lossAver.append(loss.item())
    loss_avg = sum(lossAver) / len(lossAver)
    writer.add_scalar('test/loss', loss_avg, epoch)
    return loss_avg

    
if __name__ == '__main__':

    model_conv = train_model(model_conv, trainloader, valloader, criterion, optimizer_conv, num_epochs=epochs)
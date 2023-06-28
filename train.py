from models import Classifier
from models import Generator3DLUT
import numpy as np
import os   
import itertools
import sys
import time
import argparse
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = 'cuda' if cuda else 'cpu'



criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT(initialtype='identity')
LUT1 = Generator3DLUT(initialtype='zero')
LUT2 = Generator3DLUT(initialtype='zero')
classifier = Classifier()
if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise=criterion_pixelwise.cuda()
opt_func = torch.optim.Adam

def generator_train(img,imgT):

    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A0 = LUT0(imgT)
    gen_A1 = LUT1(imgT)
    gen_A2 = LUT2(imgT)

    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b,:,:,:] = pred[b,0] * gen_A0[b,:,:,:] + pred[b,1] * gen_A1[b,:,:,:] + pred[b,2] * gen_A2[b,:,:,:] #+ pred[b,3] * gen_A3[b,:,:,:] + pred[b,4] * gen_A4[b,:,:,:]

    return combine_A, weights_norm
def generate_img(img,imgT):
    classifier.eval()
    pred = classifier(inputA).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A0 = LUT0(imgT)
    gen_A1 = LUT1(imgT)
    gen_A2 = LUT2(imgT)
    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b,:,:,:] = pred[b,0] * gen_A0[b,:,:,:] + pred[b,1] * gen_A1[b,:,:,:] + pred[b,2] * gen_A2[b,:,:,:]
    return combine_A

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batchsize",type = int ,default = 1)
parser.add_argument("--batchview",type = int,default = 50)
parser.add_argument("--dataset",type=str,default='fiveK')
parser.add_argument("--lr",type = float,default= 0.00001)
opt = parser.parse_args()

global dataloader

def train():
    print(len(dataloader))
    Pretime = time.time()
    optimizer = opt_func(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()),lr=opt.lr)
    for epoch in range(opt.epoch):
        classifier.train()
        pretime = time.time()
        for i,batch in enumerate(dataloader):
            input_A,input_T,real_B,_= batch
            input_A = input_A.to(device)
            input_T = input_T.to(device)
            real_B = real_B.to(device)
            optimizer.zero_grad()
            pred_B,weihgts_norm = generator_train(input_A,input_T)
            loss = criterion_pixelwise(real_B,pred_B)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i>0) and (i % opt.batchview == 0):
                nowtime = time.time()
                print("loss:{:.5f} time:{:.4f}".format(loss,nowtime-pretime))
                pretime = nowtime
        nowtime = time.time()
        print("epoch:{} time:{:.4f}".format(epoch,nowtime-Pretime))
        Pretime = nowtime
    LUTs = {"0": LUT0.state_dict(),"1": LUT1.state_dict(),"2": LUT2.state_dict()}
    torch.save(LUTs, "saved_models/LUTs.pth")
    torch.save(classifier.state_dict(), "saved_models/classifier.pth")

def transform_img(img_input,img_exptC,filename,mode='train'):

    if (mode == 'test'):
        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img_input = (img_input-0.5)*2
        return (img_input, img_input.transpose(0,2).transpose(0,1),img_exptC,filename)
    
    ratio_H = np.random.uniform(0.6,1.0)
    ratio_W = np.random.uniform(0.6,1.0)
    W,H = img_input._size
    crop_h = round(H*ratio_H)
    crop_w = round(W*ratio_W)
    i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
    img_input = TF.crop(img_input, i, j, h, w)
    img_exptC = TF.crop(img_exptC, i, j, h, w)

    if np.random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_exptC = TF.hflip(img_exptC)

    a = np.random.uniform(0.8,1.2)
    img_input = TF.adjust_brightness(img_input,a)

    a = np.random.uniform(0.8,1.2)
    img_input = TF.adjust_saturation(img_input,a)
    #img_input = (img_input-0.5)*2
    img_input = TF.to_tensor(img_input)
    img_input = (img_input-0.5)*2.0
    img_exptC = TF.to_tensor(img_exptC)
    #input = img_inut[0]
    return (img_input, img_input.transpose(0,2).transpose(0,1),img_exptC,filename)

def getdataset(root, mode="train", unpaird_data="fiveK", combined=True):
    file = open(os.path.join(root,'train_input.txt'),'r')
    input_files = sorted(file.readlines())
    set1_images = list()
    #set1_expert_files = list()
    for i in range(len(input_files)):
        input_file = Image.open(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
        expect_file = Image.open(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))
        set1_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='train'))
        if (i>100): break
        #set1_input_files.append(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
        #set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))

    file = open(os.path.join(root,'train_label.txt'),'r')
    input_files = sorted(file.readlines())
    set2_images = list()
    for i in range(len(input_files)):
        input_file = Image.open(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
        expect_file = Image.open(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))
        set2_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='train'))
        if (i>100): break

    file = open(os.path.join(root,'test.txt'),'r')
    input_files = sorted(file.readlines())
    test_images = list()
    for i in range(len(input_files)):
        input_file = Image.open(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
        expect_file = Image.open(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))
        test_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='test'))

    if combined:
        set1_images = set1_images + set2_images
    
    return set1_images,test_images

if __name__ == '__main__':
    
    traindata,testdata = getdataset("data/%s"%opt.dataset,mode='train')
    print(len(traindata))
    print('hahaha')
    print(cuda)
    dataloader = DataLoader(traindata,batch_size=opt.batchsize,shuffle=True,num_workers=2)
    train()
    for (i,imgs) in enumerate(testdata):
        if (i==0):
            inputA,imgT,expctC,filename = imgs
            inputA = inputA.to(device)
            imgT = imgT.to(device)
            inputA = inputA.unsqueeze(0)
            imgT = imgT.unsqueeze(0)
            preB = generate_img(inputA,imgT).squeeze(0)
            #preB = torch.round(preB*255)
            save_image(preB,'%s.jpg'%filename)
            break
from models import Classifier
from models import Generator3DLUT
from models import MergeWeight
import numpy as np
import os   
import itertools
import sys
import time
import math
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

class Regularization(torch.nn.Module):
    def __init__(self, dim=33):
        super(Regularization,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn
    
reg = Regularization()

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT(initialtype='identity')
LUT1 = Generator3DLUT(initialtype='zero')
LUT2 = Generator3DLUT(initialtype='zero')
classifier = Classifier()
mergedWeight = MergeWeight()
if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise=criterion_pixelwise.cuda()
    reg.cuda()
    reg.weight_b = reg.weight_b.type(Tensor)
    reg.weight_r = reg.weight_r.type(Tensor)
    reg.weight_g = reg.weight_g.type(Tensor)
opt_func = torch.optim.Adam

def generator_train(img,imgT):
    _,h,w = img.shape
    pred = classifier(img).squeeze()
    pred_1 = classifier(img[:,0:(h//2),0:(w//2)])
    pred_2 = classifier(img[:,(h//2) + 1 :h,0:(w//2)])
    pred_3 = classifier(img[:,(h//2) + 1 :h,(w//2) + 1 : w])
    pred_4 = classifier(img[:,0:(h//2),(w//2) + 1 : w])
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    if len(pred_1.shape) == 1:
        pred_1 = pred_1.unsqueeze(0)
    if len(pred_2.shape) == 1:
        pred_2 = pred_2.unsqueeze(0)
    if len(pred_3.shape) == 1:
        pred_3 = pred_3.unsqueeze(0)
    if len(pred_4.shape) == 1:
        pred_4 = pred_4.unsqueeze(0)
    weights = mergedWeight(img)
    gen_A0 = LUT0(imgT)
    gen_A1 = LUT1(imgT)
    gen_A2 = LUT2(imgT)

    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        x,y = b//w, b - b//w
        combine_A[b,:,:,:] = weights[0] * (pred[b,0] * gen_A0[b,:,:,:] + pred[b,1] * gen_A1[b,:,:,:] + pred[b,2] * gen_A2[b,:,:,:]) #+ pred[b,3] * gen_A3[b,:,:,:] + pred[b,4] * gen_A4[b,:,:,:]
        if x <= h//2 and y <= w // 2:
            combine_A[b,:,:,:] += weights[1] * (pred_1[b,0] * gen_A0[b,:,:,:] + pred_1[b,1] * gen_A1[b,:,:,:] + pred_1[b,2] * gen_A2[b,:,:,:])
        elif x > h//2 and y <= w//2:
            combine_A[b,:,:,:] += weights[1] * (pred_2[b,0] * gen_A0[b,:,:,:] + pred_2[b,1] * gen_A1[b,:,:,:] + pred_2[b,2] * gen_A2[b,:,:,:])
        elif x > h//2 and y > w//2:
            combine_A[b,:,:,:] += weights[1] * (pred_3[b,0] * gen_A0[b,:,:,:] + pred_3[b,1] * gen_A1[b,:,:,:] + pred_3[b,2] * gen_A2[b,:,:,:])
        else:
            combine_A[b,:,:,:] += weights[1] * (pred_4[b,0] * gen_A0[b,:,:,:] + pred_4[b,1] * gen_A1[b,:,:,:] + pred_4[b,2] * gen_A2[b,:,:,:])
    return combine_A, weights_norm

def generate_img(img,imgT):
    _,h,w = img.shape
    classifier.eval()
    pred = classifier(img).squeeze()
    pred_1 = classifier(img[:,0:(h//2),0:(w//2)])
    pred_2 = classifier(img[:,(h//2) + 1 :h,0:(w//2)])
    pred_3 = classifier(img[:,(h//2) + 1 :h,(w//2) + 1 : w])
    pred_4 = classifier(img[:,0:(h//2),(w//2) + 1 : w])
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    if len(pred_1.shape) == 1:
        pred_1 = pred_1.unsqueeze(0)
    if len(pred_2.shape) == 1:
        pred_2 = pred_2.unsqueeze(0)
    if len(pred_3.shape) == 1:
        pred_3 = pred_3.unsqueeze(0)
    if len(pred_4.shape) == 1:
        pred_4 = pred_4.unsqueeze(0)
    weights = mergedWeight(img)
    gen_A0 = LUT0(imgT)
    gen_A1 = LUT1(imgT)
    gen_A2 = LUT2(imgT)
    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        x,y = b//w, b - b//w
        combine_A[b,:,:,:] = weights[0] * (pred[b,0] * gen_A0[b,:,:,:] + pred[b,1] * gen_A1[b,:,:,:] + pred[b,2] * gen_A2[b,:,:,:]) #+ pred[b,3] * gen_A3[b,:,:,:] + pred[b,4] * gen_A4[b,:,:,:]
        if x <= h//2 and y <= w // 2:
            combine_A[b,:,:,:] += weights[1] * (pred_1[b,0] * gen_A0[b,:,:,:] + pred_1[b,1] * gen_A1[b,:,:,:] + pred_1[b,2] * gen_A2[b,:,:,:])
        elif x > h//2 and y <= w//2:
            combine_A[b,:,:,:] += weights[1] * (pred_2[b,0] * gen_A0[b,:,:,:] + pred_2[b,1] * gen_A1[b,:,:,:] + pred_2[b,2] * gen_A2[b,:,:,:])
        elif x > h//2 and y > w//2:
            combine_A[b,:,:,:] += weights[1] * (pred_3[b,0] * gen_A0[b,:,:,:] + pred_3[b,1] * gen_A1[b,:,:,:] + pred_3[b,2] * gen_A2[b,:,:,:])
        else:
            combine_A[b,:,:,:] += weights[1] * (pred_4[b,0] * gen_A0[b,:,:,:] + pred_4[b,1] * gen_A1[b,:,:,:] + pred_4[b,2] * gen_A2[b,:,:,:])
    return combine_A

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batchsize",type = int ,default = 1)
parser.add_argument("--batchview",type = int,default = 50)
parser.add_argument("--dataset",type=str,default='Teal&Orange')
parser.add_argument("--lr",type = float,default= 0.00001)
opt = parser.parse_args()

global dataloader

def train(R_s = True, R_m = True, lambda_s = 1e-3, lambda_m = 1):
    print(len(dataloader))
    Pretime = time.time()
    optimizer = opt_func(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters(), MergeWeight.parameters()),lr=opt.lr)
    
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
            tv0, mn0 = reg(LUT0)
            tv1, mn1 = reg(LUT1)
            tv2, mn2 = reg(LUT2)
            tv = tv0 + tv1 + tv2
            mn = mn0 + mn1 + mn2
            loss = criterion_pixelwise(real_B,pred_B)
            if (R_s):
                loss = loss + lambda_s * (weihgts_norm + tv)
            if (R_m):
                loss = loss + lambda_m * (mn)
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
    torch.save(LUTs, "LUTs.pth")
    torch.save(classifier.state_dict(), "classifier.pth")

def transform_img(img_input,img_exptC,filename,mode='train'):
    #print(filename)
    if (mode == 'test'):
        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        T = img_input.clone()
        T = T*255
        N = img_input.size(1)*img_input.size(2)
        N/=32.0
        L = [0 for i in range(256)]
        R = [32 for i in range(256)]
        V = [0.0 for i in range(33)]
        def trans(x,*y):
            l = V[L[round(x)]]
            r = V[R[round(x)]]
            return (L[round(x)]*(255.0/32.0) + (x-l)/(r-l)*255.0/32.0)
        for color in range(img_input.size(0)):
            V[0]=0
            V[32]=255
            preC = 0
            T1 = torch.flatten(T[color])
            for i in range(1,32):
                C = (T1.kthvalue(round(N*i))).values.int().item()
                oriC = i*255.0/32.0
                C = (oriC+oriC+C)/3.0
                V[i]=C
            for i in range(32):
                for j in range(math.ceil(V[i]),int(V[i+1])+1):
                    L[j]=i
                    R[j]=i+1
            T[color].map_(T[color],trans)
        T = ((T/255)-0.5)*2.0
        img_input = (img_input-0.5)*2
        return (img_input, T.transpose(0,2).transpose(0,1),img_exptC,filename)
    
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
    img_input = TF.to_tensor(img_input)
    T = img_input.clone()
    T = T*255
    N = img_input.size(1)*img_input.size(2)
    N/=32
    L = [0 for i in range(256)]
    R = [32 for i in range(256)]
    V = [0.0 for i in range(33)]
    def trans(x,*y):
        l = V[L[round(x)]]
        r = V[R[round(x)]]
        return (L[round(x)]*(255.0/32.0) + (x-l)/(r-l)*255.0/32.0)
    for color in range(img_input.size(0)):
        V[0]=0
        V[32]=255
        T1 = torch.flatten(T[color])
        for i in range(1,32):
            C = (T1.kthvalue(round(N*i))).values.int().item()
            oriC = i*255.0/32.0
            C = (oriC+oriC+C)/3.0 #稍微根据颜色的分布偏离均匀点
            V[i]=C
        for i in range(32):
            for j in range(math.ceil(V[i]),int(V[i+1])+1):
                L[j]=i
                R[j]=i+1
        T[color].map_(T[color],trans)
    T = ((T/255)-0.5)*2.0

    img_input = (img_input-0.5)*2.0
    img_exptC = TF.to_tensor(img_exptC)
    #input = img_inut[0]
    return (img_input, T.transpose(0,2).transpose(0,1),img_exptC,filename)

def getdataset(root, mode="train", unpaird_data="fiveK", combined=True):
    file = open(os.path.join(root,'train_img.txt'),'r')
    input_files = sorted(file.readlines())
    set1_images = list()
    #set1_expert_files = list()
    for i in range(len(input_files)):
        input_file = Image.open(os.path.join(root,"origin_jpgs",input_files[i][:-1]))
        expect_file = Image.open(os.path.join(root,"new_jpgs",input_files[i][:-1]))
        set1_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='train'))
        #set1_input_files.append(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
        #set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))

    # file = open(os.path.join(root,'train_label.txt'),'r')
    # input_files = sorted(file.readlines())
    # set2_images = list()
    # for i in range(len(input_files)):
    #     input_file = Image.open(os.path.join(root,"input","JPG/480p",input_files[i][:-1] + ".jpg"))
    #     expect_file = Image.open(os.path.join(root,"expertC","JPG/480p",input_files[i][:-1] + ".jpg"))
    #     set2_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='train'))
    #     if (i>500): break

    file = open(os.path.join(root,'test_img.txt'),'r')
    input_files = sorted(file.readlines())
    test_images = list()
    for i in range(len(input_files)):
        input_file = Image.open(os.path.join(root,"origin_jpgs",input_files[i][:-1]))
        expect_file = Image.open(os.path.join(root,"new_jpgs",input_files[i][:-1]))
        test_images.append(transform_img(input_file,expect_file,input_files[i][:-1],mode='test'))

    # if combined:
    #     set1_images = set1_images + set2_images
    
    return set1_images,test_images

if __name__ == '__main__':
    
    traindata,testdata = getdataset("data/%s"%opt.dataset,mode='train')
    dataloader = DataLoader(traindata,batch_size=opt.batchsize,shuffle=True,num_workers=2)
    train()
    for (i,imgs) in enumerate(testdata):
        if (i>=0):
            inputA,imgT,expctC,filename = imgs
            inputA = inputA.to(device)
            imgT = imgT.to(device)
            inputA = inputA.unsqueeze(0)
            subimages = []
            
            imgT = imgT.unsqueeze(0)
            preB = generate_img(inputA,imgT).squeeze(0)
            #preB = torch.round(preB*255)
            save_image(preB,'%s.jpg'%filename)
            break
import torch
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
from torch.optim import SGD
import random
import os
import cv2
import numpy as np

cuda_avail = torch.cuda.is_available()

pathDir = os.listdir('./1080pTest/')


class LaserNet(nn.Module):
    def __init__(self):

        super(LaserNet, self).__init__()
        #128,3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1,bias=False)
        self.relu1 = nn.ReLU()
        #64,4
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1,bias=False)
        self.relu2 = nn.ReLU()
        #32,8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1,bias=False)
        self.relu3 = nn.ReLU()
        #16,8
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu4 = nn.ReLU()
        #16,8

        self.up4 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        #32,8

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu5 = nn.ReLU()
        #32,4

        self.up5 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        #64,4

        self.conv6 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu6 = nn.ReLU()
        #64,1

        self.up6 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        #128,1


    def forward(self, input):
        output1 = self.conv1(input)
        output1 = self.relu1(output1)

        #output1 = torch.cat((input, output1), 1)

        output2 = self.conv2(output1)
        output2 = self.relu2(output2)

        #output2 = torch.cat((output1, output2), 1)

        output3 = self.conv3(output2)
        output3 = self.relu3(output3)

        #output3 = torch.cat((output2, output3), 1)

        output4 = self.conv4(output3)
        output4 = self.relu4(output4)

        output4 = self.up4(output4)

        output4 = torch.cat((output2, output4), 1)

        output5 = self.conv5(output4)
        output5 = self.relu5(output5)

        output5 = self.up5(output5)

        output5 = torch.cat((output1, output5), 1)

        output6 = self.conv6(output5)
        output6 = self.relu6(output6)

        output6 = self.up6(output6)

        return output6


model = LaserNet()

checkpoint = torch.load('./50_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.cuda()

summary(model.eval(), input_size=(3, 128, 128))

for i in range(len(pathDir)):

    image_name  = pathDir[i]

    img = cv2.imread('./1080pTest/'+ image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img / 255.0

    chech_img = img
    chech_img = chech_img.unsqueeze(0)
    chech_img = chech_img.cuda()
    out = model(chech_img)

    out_image = out.cpu()

    out_image = torch.cat((out_image,out_image,out_image), 1).squeeze(0)
    out_image = torch.cat((out_image,img), 1)
    print(out_image.shape)

    out_image = out_image.permute(1, 2, 0).detach().numpy()
    out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)

    name = './testOutPut/'+ image_name
    cv2.imwrite(name,out_image*255)

    cv2.imshow('out', out_image)
    cv2.waitKey(1)


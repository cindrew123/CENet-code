import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
affine_par = True


class SoftDiceLoss(nn.Module):
# dice_loss1 = dice_criterion(output_train[0,...], gt_onehot1).item()
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation='sigmoid', reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        #print y_pred.shape
        #print y_true.shape
    
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):


    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    #print pred
    #print gt.shape
    #H = pred.size(0)

    N = gt.size(0)
    #print N
    #print pred.shape
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat.cuda() * gt_flat.cuda()).sum(1)
    unionset = (pred_flat.cuda()).sum(1) + (gt_flat.cuda()).sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):


    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N









def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        ##
        # self.conv3 = nn.Conv2d(2048,256,kernel_size=1,stride=1,bias=False)
        ##
        self.conv1=nn.Conv2d(2048,256,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(256,affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True),
                                                  nn.BatchNorm2d(256,affine=affine_par),
                                                  nn.ReLU(inplace=True)))
        self.conv2=nn.Conv2d(2048,256,kernel_size=1,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(256, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv_last=nn.Conv2d(256*5,256,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(256, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.classifier=nn.Conv2d(256,num_classes,kernel_size=1,stride=1,bias=False)
        #self.g_style = nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, bias=False)
        # self.glo_avr_pool = nn.functional.adaptive_avg_pool2d(a, (1,1))
        self.sigmoid = nn.Sigmoid()
        self.Multiply = np.multiply
        self.Add = np.add
        self.Dot = np.dot
        self.Transpose = np.transpose
        self.downchannel = nn.Conv2d(512,256,kernel_size=1,stride=1,bias=False)
        self.downchannel_1 = nn.Conv2d(256,1,kernel_size=1,stride=1,bias=False)
        self.downchannel_2 = nn.Conv2d(256,16,kernel_size=1,stride=1,bias=False)
        self.downchannel_3 = nn.Conv2d(16,1,kernel_size=1,stride=1,bias=False)
        self.downchannel_4 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1, bias=False)
        
        self.upchannel = nn.Conv2d(16,256,kernel_size=1,stride=1,bias=False)
        self.l2 = nn.Conv2d(64, 64,kernel_size = 3, stride=4, bias=False)
        self.l4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.l6 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.l8 = nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=1,bias=False)
        #self.l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False)
        self.lo = nn.Conv2d(512, 256, kernel_size=3, stride=1,padding=1, bias=False)


    def forward(self, x, l2, l4, l6, l8):
        ## outcam1 final shape is 256 chanel
        # out 256 64 64
        # before jinzita
        # adaptive_avg_pool2d makes outcam1 256 1 1
        # multiply makes outcam1 256 64 64
        # outcam1 final shape is 256 64 64
        # outcam1 is jingguo cam , before jinzita
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = out
        outcam = out
        nn.functional.adaptive_avg_pool2d(outcam, (1, 1))
        outcam1 = self.relu(outcam)
        outcam1 = self.sigmoid(outcam1)
        # -->16*64*64
        outcam1 = self.downchannel_2(outcam1)
        # -->16*(64*64)
        outcam1 = outcam1.data.cpu().numpy()
        if outcam1.shape[0] ==  2:
            outcam1a = outcam1[0,...]
            outcam1b = outcam1[1,...]
            outcam1a = outcam1a.reshape(16,4096)
            outcam1b = outcam1b.reshape(16,4096)
            # transpose --> (64*64)*16
            outcam3a = self.Transpose(outcam1a)
            outcam3b = self.Transpose(outcam1b)

            # multiply [16*(64*64)] x [(64*64)*16] --> 16*16
            # outcam1 16*(64*64), outcam3 (64*64)*16
            # outcam4 is A
            outcam4a = self.Dot(outcam1a, outcam3a)
            outcam4b = self.Dot(outcam1b, outcam3b)

            # multiply [16*16] * [16*(64*64)] --> 16*(64*64), outcam4 * outcam1
            outcam5a = self.Dot(outcam4a, outcam1a)
            outcam5b = self.Dot(outcam4b, outcam1b)

            # 16*(64*64) --> 16*64*64
            # outcam5 is B
            outcam5a = outcam5a.reshape(16,64,64)
            outcam5b = outcam5b.reshape(16,64,64)

            # 16*64*64 --> 1*64*64
            outcam5 = outcam1
            outcam5[0,...] = torch.from_numpy(outcam5a).cuda()
            outcam5[1,...] = torch.from_numpy(outcam5b).cuda()
            outcam5 =  torch.from_numpy(outcam5).cuda()
            
        else:
            outcam1 = outcam1.reshape(16,4096)
            outcam3 = self.Transpose(outcam1)
            outcam4 = self.Dot(outcam1, outcam3)
            outcam5 = self.Dot(outcam4, outcam1)
            outcam5 = outcam5.reshape(16,64,64)
            outcam5 = torch.from_numpy(outcam5).cuda()
            outcam5 = outcam5.unsqueeze(0)
            
        
        
        ##
        outcam1 = self.upchannel(outcam5)
        outcam1 = torch.cat((outcam,outcam1),1)
        # 512*64*64 --> 256 *64*64
        outcam1 = self.downchannel(outcam1)
        # jinzita kongdong
        for i in range(len(self.conv2d_list)):
            out = torch.cat((out,self.conv2d_list[i](x)),1)
        
        global_avg = nn.AvgPool2d(kernel_size=x.size()[-1]/16,stride=1)(x)
        img_feature = nn.Upsample(size=x.size()[2:],mode='bilinear')(global_avg)
        img_feature = self.conv2(img_feature)
        img_feature = self.bn2(img_feature)
        out = torch.cat((out,img_feature),1)
        out = self.relu(out)
        out = self.conv_last(out)
        out = self.bn3(out)
        out = self.relu(out)   
        # concat --> 512 *64*64
        out = torch.cat((out,outcam1),1)
        # 512 *64*64 --> 256 *64*64
        out_last = self.downchannel(out)
        
        # junyun
        l2 = self.l2(l2)
        l4 = self.l4(l4)
        l6 = self.l6(l6)
        l8 = self.l8(l8)
        l = torch.cat([l2,l4,l6,l8], dim=1).type(torch.FloatTensor)
        # 256 *64*64 --> 16 *64*64
        l = self.downchannel_4(l.cuda())
        # 16 *64*64 --> 256 *64*64
        l = self.upchannel(l)
        lo = torch.cat([l.cuda(),out_last], dim=1).type(torch.FloatTensor)
        lo = self.lo(lo.cuda())
        lo = self.sigmoid(lo)
        out = self.classifier(lo)

        return out_last,out

class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg

class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer_multi_grid(block, 512, layers[3], stride=1, dilation=4,multi_grid=[1,2,4])
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18],[6,12,18],num_classes)
        #self.conv_downfeature = nn.Conv2d(2048, 256, kernel_size=1, stride=1,bias=False)
        #self.bn_downfeature = nn.BatchNorm2d(256, affine = affine_par)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False



    def _make_layer_multi_grid(self, block, planes, blocks, stride=1, dilation=1,multi_grid=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        if multi_grid ==None:
            multi_grid=np.ones(blocks,dtype=int)

        assert len(multi_grid)==blocks
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation*multi_grid[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation*multi_grid[i]))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        l2 = x
        x = self.relu(x)
        x = self.maxpool(x)
        l4 = x
        x = self.layer1(x)
        x = self.layer2(x)
        l6 = x
        x = self.layer3(x)
        x = self.layer4(x)
        l8 = x
        #x4 = self.conv_downfeature(x)
        #x4 = self.bn_downfeature(x4)
        #x4 = self.relu(x4)
        x = self.layer5(x,l2,l4,l6,l8)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.channels = [512,256,128,3]
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(256,self.channels[0],3,1,1,bias=False),
                                    nn.BatchNorm2d(self.channels[0]),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(self.channels[0], self.channels[1], 4,2, 1, bias=False),
                                     nn.BatchNorm2d(self.channels[1]),
                                     nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(self.channels[1], self.channels[2], 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(self.channels[2]),
                                     nn.ReLU(inplace=True))

        self.layer4 = nn.ConvTranspose2d(self.channels[2], self.channels[3], 4, 2, 1, bias=False)

        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out= self.tanh(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AttenLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttenLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Softmax(dim=1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        atten_map = x*y
        return self.gamma * atten_map+x

class Self_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self,in_dim):
        super(Self_Attn, self).__init__()

        self.in_dim = in_dim
        self.f_ = nn.Conv2d(in_dim, int(in_dim / 8), 1)
        self.g_ = nn.Conv2d(in_dim, int(in_dim / 8), 1)
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pixel_wise=True):
        if pixel_wise:
            b_size = x.size(0)
            f_size = x.size(-1)

            f_x = self.f_(x)  # batch x in_dim/8*f*f x f_size x f_size
            g_x = self.g_(x)  # batch x in_dim/8*f*f x f_size x f_size
            h_x = self.h_(x)  # batch x in_dim x f_size x f_size

            attn_dist = torch.mul(f_x, g_x).sum(dim=1).contiguous().view(-1, f_size ** 2)  # batch*f*f x f_size1*f_size2
            attn_soft = F.softmax(attn_dist, dim=1).contiguous().view(b_size, f_size , f_size )  # batch x f*f x f*f
            attn_score = attn_soft.unsqueeze(1)  # batch x 1 x f*f x f*f

            self_attn_map = torch.mul(h_x, attn_score)  # batch x in_dim x f*f

            self_attn_map = self.gamma * self_attn_map + x

        return self_attn_map

class Self_Attn_2(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self,in_dim):
        super(Self_Attn_2, self).__init__()

        self.in_dim = in_dim
        self.f_ = nn.Conv2d(in_dim, int(in_dim / 8), 1)
        self.g_ = nn.Conv2d(in_dim, int(in_dim / 8), 1)
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pixel_wise=True):
        if pixel_wise:

            f_x = self.f_(x)  # batch x in_dim/8*f*f x f_size x f_size
            g_x = self.g_(x)  # batch x in_dim/8*f*f x f_size x f_size
            h_x = self.h_(x)  # batch x in_dim x f_size x f_size

            size_f_g = f_x.size()
            size_h = h_x.size()

            f_x = f_x.transpose(1,0).contiguous().view(size_f_g[1],-1)
            g_x = g_x.transpose(1, 0).contiguous().view(size_f_g[1], -1)
            h_x = h_x.transpose(1, 0).contiguous().view(size_h[1], -1)

            attn_dist = torch.matmul(f_x.t(),g_x)
            attn_soft = F.softmax(attn_dist,dim=1)
            self_attn_map = torch.matmul(h_x,attn_soft).contiguous().view(size_h[1],size_h[0],size_h[2],size_h[3])
            self_attn_map = self_attn_map.transpose(1,0)


            self_attn_map = self.gamma * self_attn_map + x

        return self_attn_map

class Discriminator(nn.Module):
    """Discriminator model."""

    def __init__(self,num_class):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.num_class = num_class
        self.channels = [512, 1024, 2048, 1]
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer1 = nn.Sequential(nn.Conv2d(256, self.channels[0], 1, 1, bias=False),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(self.channels[0], self.channels[1], 3,1, 1, bias=False),
                                     nn.InstanceNorm2d(self.channels[1]),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(nn.Conv2d(self.channels[1], self.channels[2], 3,1, 1, bias=False),
                                     nn.InstanceNorm2d(self.channels[2]),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.cls = nn.Conv2d(self.channels[2],self.num_class,3,1,padding=2,dilation=2,bias=False)

        self.layer4 = nn.Conv2d(self.channels[2]*2, self.channels[3], 3,1,1,bias=False)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.class_label = nn.Conv2d(self.channels[2],self.num_class,kernel_size=4,stride=1,padding=2, dilation=2, bias=False)

    def forward_onece(self,x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        return out

    def forward(self, x1,x2,domain = 'S'):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1,out2),1)
        d = self.layer4(d)
        d = self.sigmoid(d)
        if domain == 'S':
            c1 = self.cls(out1)
            c2 = self.cls(out2)
            return d,c1,c2
        else:
            return d


class Discriminator_Triplet(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_Triplet, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        #self.g_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(self.channels[3],self.channels[4])
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out

    def forward(self, x):
        """Forward the discriminator."""

        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        out = self.leaky_relu(out)
        #out = self.g_avg_pool(out)
        #out = out.view(-1, self.channels[3])
        #out = self.fc(out)
        out = self.layer5(out)
        #out = self.sigmoid(out)
        out = out.mean().view(-1,1)
        #out = F.normalize(out,dim=1)
        return out

class Discriminator_Triplet_Split(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class,split=2):
        super(Discriminator_Triplet_Split, self).__init__()
        self.num_class = num_class
        self.split = split
        self.channels = [64, 128, 256, 512, 128]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.g_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.channels[3],self.channels[4])
        #self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def Split_Enum(self,x1):
        enum_x1 = []
        w1,h1 = x1.size()[2:]
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)

        xx1 = F.adaptive_max_pool2d(x1,(w,h))
        enum_x1.append(xx1)

        d = enum_x1[0]
        for i in range(len(enum_x1)):
            if i == 0 :
                continue
            else:
                d = torch.cat((d, enum_x1[i]), 0)

        return d

    def forward(self, x):
        """Forward the discriminator."""
        out = self.Split_Enum(x)

        out = self.layer1(out)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        out = self.leaky_relu(out)
        out = self.g_avg_pool(out)
        out = out.view(-1, self.channels[3])
        out = self.fc(out)
        #out = self.layer5(out)
        #out = self.sigmoid(out)
        out = out.view(-1,self.channels[4])
        #out = F.normalize(out,dim=1)
        return out

class Discriminator_Down(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""
    
    def __init__(self,num_class):
        super(Discriminator_Down,self).__init__()
        self.num_class = num_class
        self.channels = [512,1024,2048,1]
        self.layer1 = nn.Conv2d(256,self.channels[0],kernel_size=4,stride=2,padding=1)
        self.layer2 = nn.Conv2d(self.channels[0],self.channels[1],kernel_size=4,stride=2,padding=1)
        self.layer3 = nn.Conv2d(self.channels[1],self.channels[2],kernel_size=4,stride=2,padding=1)
        self.layer4 = nn.Conv2d(self.channels[2]*2,self.channels[3],kernel_size=4,stride=2,padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward_onece(self,x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out
    
    def forward(self,x1,x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1,out2),1)
        d = self.layer4(d)
        d = self.sigmoid(d)
        return d


class Discriminator_ONES_ONET(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_ONES_ONET, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        out = self.leaky_relu(out)
        out = self.layer5(out)
        out = self.sigmoid(out)
        return out

class Discriminator_NumClass_Layer0(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer0, self).__init__()
        self.num_class = num_class
        self.channels = [64, 256, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0]*2, self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer1(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer1, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1]*2, self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d


class Discriminator_NumClass_Layer2(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer2, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer0_DOutPutOne(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer0_DOutPutOne, self).__init__()
        self.num_class = num_class
        self.channels = [64, 256, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0]*2, self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.leaky_relu(d)
        d = self.globalAvg(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer1_DOutPutOne(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer1_DOutPutOne, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1]*2, self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.leaky_relu(d)
        d = self.globalAvg(d)
        d = self.sigmoid(d)
        return d


class Discriminator_NumClass_Layer2_DOutPutOne(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer2_DOutPutOne, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.leaky_relu(d)
        d = self.globalAvg(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""
    
    def __init__(self,num_class):
        super(Discriminator_NumClass,self).__init__()
        self.num_class = num_class
        self.channels = [64,128,256,1024,1]
        self.layer1 = nn.Conv2d(self.num_class,self.channels[0],kernel_size=4,stride=2,padding=1)
        self.layer2 = nn.Conv2d(self.channels[0],self.channels[1],kernel_size=4,stride=2,padding=1)
        self.layer3 = nn.Conv2d(self.channels[1],self.channels[2],kernel_size=4,stride=2,padding=1)
        self.layer4 = nn.Conv2d(self.channels[2]*2,self.channels[3],kernel_size=4,stride=2,padding=1)
        self.layer5 = nn.Conv2d(self.channels[3],self.channels[4],kernel_size=4,stride=1,padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward_onece(self,x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out
    
    def forward(self,x1,x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1,out2),1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_2, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        #print out.shape
        out = self.leaky_relu(out)
        # if input is 1 img, out is (512,32,32)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        # if input is 2 img, out is (1, 1, 16, 16 )

        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2_DOutPutOne(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_2_DOutPutOne, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=4, stride=2, padding=1)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        out = self.leaky_relu(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)

        d = self.layer5(d)
        d = self.leaky_relu(d)
        d = self.globalAvg(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2_FullConv(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_2_FullConv, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        out = self.leaky_relu(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)

        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2_Entropy_Weight (nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_2_Entropy_Weight, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        return out


    def entropy_y_x(self,p_logit):
        #p = F.softmax(p_logit,1)
        entropy = -F.softmax(p_logit,1) * F.log_softmax(p_logit,1)
        entr_weight = entropy.sum(dim=1)/np.log(self.num_class)
        return entr_weight*p_logit

    def forward(self, x1, x2):
        """Forward the discriminator."""
        x1 = self.entropy_y_x(x1)
        x2 = self.entropy_y_x(x2)
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2_Entropy_Weight_Inverse (nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_2_Entropy_Weight_Inverse, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        return out


    def entropy_y_x(self,p_logit):
        #p = F.softmax(p_logit,1)
        entropy = -F.softmax(p_logit,1) * F.log_softmax(p_logit,1)
        entr_weight = entropy.sum(dim=1)/np.log(self.num_class)
        return (1-entr_weight)*p_logit

    def forward(self, x1, x2):
        """Forward the discriminator."""
        x1 = self.entropy_y_x(x1)
        x2 = self.entropy_y_x(x2)
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_FullConv(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col_FullConv, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """Forward the discriminator."""

        d = torch.cat((x1, x2), 1)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer0_FullConv(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer0_FullConv, self).__init__()
        self.num_class = num_class
        self.channels = [64, 256, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(self.channels[0]*2, self.channels[1], kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer1_FullConv(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer1_FullConv, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 512, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(self.channels[1]*2, self.channels[2], kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Layer2_FullConv(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Layer2_FullConv, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(self.channels[2]*2, self.channels[3], kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_2_4Layer(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_2_4Layer, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2]*2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)

        d = self.layer4(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Entropy (nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Entropy, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out

    def entropy_y_x(self,p_logit):
        p = F.softmax(p_logit,1)
        entropy = - torch.sum(p * torch.log(p),1).unsqueeze(1)
        return entropy*p_logit + p_logit

    def forward(self, x1, x2):
        """Forward the discriminator."""
        x1 = self.entropy_y_x(x1)
        x2 = self.entropy_y_x(x2)
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        #d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Channel_Weight (nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Channel_Weight, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out

    def channel_weight(self,p_logit):
        channel_avg = Variable(torch.zeros(self.num_class)).cuda()
        p = F.softmax(p_logit,1)
        pred_max , pred_arg = torch.max(p,1)
        for i in range(self.num_class):
            channel_max = pred_max[pred_arg==i]
            if channel_max.data.nelement() !=0:
                avg = torch.mean(channel_max)
                channel_avg[i] = avg
        channel_avg = channel_avg.view(1,-1,1,1)
        return channel_avg*p_logit + p_logit

    def forward(self, x1, x2):
        """Forward the discriminator."""
        x1 = self.channel_weight(x1)
        x2 = self.channel_weight(x2)
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Channel_Weight_2 (nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Channel_Weight_2, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out

    def channel_weight(self,p_logit):
        channel_avg = Variable(torch.zeros(self.num_class)).cuda()
        p = F.softmax(p_logit,1)
        pred_max , pred_arg = torch.max(p,1)
        for i in range(self.num_class):
            channel_max = pred_max[pred_arg==i]
            if channel_max.data.nelement() !=0:
                avg = torch.mean(channel_max)
                channel_avg[i] = avg
        channel_avg = 2.0/(1+channel_avg)-1.0
        channel_avg = channel_avg.view(1,-1,1,1)
        return channel_avg*p_logit + p_logit

    def forward(self, x1, x2):
        """Forward the discriminator."""
        x1 = self.channel_weight(x1)
        x2 = self.channel_weight(x2)
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_AttenLayer(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_AttenLayer, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.atten_layer = AttenLayer(self.num_class,reduction=4)
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.atten_layer(x)
        out = self.layer1(out)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d


class Discriminator_NumClass_Self_Atten(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Self_Atten, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.attn1 = Self_Attn(self.channels[2])
        self.attn2 = Self_Attn(self.channels[3])
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.attn1(out)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.attn2(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Self_Atten_2(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Self_Atten_2, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1024, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] * 2, self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.attn1 = Self_Attn(self.channels[0])
        #self.attn2 = Self_Attn(self.channels[3])
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.attn1(out)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)

        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        #d = self.attn2(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out


    def forward(self, x1, x2):
        """Forward the discriminator."""
        d = torch.cat((x1,x2),1)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_OutPutOne(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col_OutPutOne, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out


    def forward(self, x1, x2):
        """Forward the discriminator."""
        d = torch.cat((x1,x2),1)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.leaky_relu(d)
        d = self.globalAvg(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_4layerD(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col_4layerD, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        #self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out


    def forward(self, x1, x2):
        """Forward the discriminator."""
        d = torch.cat((x1,x2),1)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        #d = self.leaky_relu(d)
        #d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Self_Atten(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col_Self_Atten, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attn1 = Self_Attn(self.channels[2])
        self.attn2 = Self_Attn(self.channels[3])
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out


    def forward(self, x1, x2):
        """Forward the discriminator."""
        d = torch.cat((x1,x2),1)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.attn1(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.attn2(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        #d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Self_Atten_2(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class):
        super(Discriminator_NumClass_Single_Col_Self_Atten_2, self).__init__()
        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attn1 = Self_Attn(self.channels[0])
        #self.attn2 = Self_Attn(self.channels[3])
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out


    def forward(self, x1, x2):
        """Forward the discriminator."""
        d = torch.cat((x1,x2),1)
        d = self.layer1(d)
        d = self.attn1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)

        d = self.leaky_relu(d)
        d = self.layer4(d)
        #d = self.attn2(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Split_16(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None):
        super(Discriminator_NumClass_Single_Col_Split_16, self).__init__()
        self.num_class = num_class
        self.split = split
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        xx1 = F.adaptive_max_pool2d(x1,(w,h))
        xx2 = F.adaptive_max_pool2d(x2,(w,h))
        enum_x1.append(xx1)
        enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(len(enum_x1)):
            for j in range(len(enum_x2)):
                if i == 0 and j == 0:
                    continue
                else:
                    temp_d = torch.cat((enum_x1[i], enum_x2[j]), 1)
                    d = torch.cat((d, temp_d), 0)
        return d

    def forward(self, x1, x2):
        """Forward the discriminator."""
        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Split_SW(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None):
        super(Discriminator_NumClass_Single_Col_Split_SW, self).__init__()
        self.num_class = num_class
        self.split = split
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        xx1 = F.adaptive_max_pool2d(x1,(w,h))
        xx2 = F.adaptive_max_pool2d(x2,(w,h))
        enum_x1.append(xx1)
        enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(1,len(enum_x1)):
            temp_d = torch.cat((enum_x1[i], enum_x2[i]), 1)
            d = torch.cat((d, temp_d), 0)
        return d

    def forward(self, x1, x2):
        """Forward the discriminator."""
        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Split_16_BN(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None):
        super(Discriminator_NumClass_Single_Col_Split_16_BN, self).__init__()
        self.num_class = num_class
        self.split = split
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels[0],affine=affine_par)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channels[1], affine=affine_par)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channels[2], affine=affine_par)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channels[3], affine=affine_par)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        xx1 = F.adaptive_max_pool2d(x1,(w,h))
        xx2 = F.adaptive_max_pool2d(x2,(w,h))
        enum_x1.append(xx1)
        enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(len(enum_x1)):
            for j in range(len(enum_x2)):
                if i == 0 and j == 0:
                    continue
                else:
                    temp_d = torch.cat((enum_x1[i], enum_x2[j]), 1)
                    d = torch.cat((d, temp_d), 0)
        return d

    def forward(self, x1, x2):
        """Forward the discriminator."""
        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.bn1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.bn2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.bn3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.bn4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None):
        super(Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool, self).__init__()
        self.num_class = num_class
        self.split = split
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        # xx1 = F.adaptive_max_pool2d(x1,(w,h))
        # xx2 = F.adaptive_max_pool2d(x2,(w,h))
        # enum_x1.append(xx1)
        # enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(len(enum_x1)):
            for j in range(len(enum_x2)):
                if i == 0 and j == 0:
                    continue
                else:
                    temp_d = torch.cat((enum_x1[i], enum_x2[j]), 1)
                    d = torch.cat((d, temp_d), 0)
        return d

    def forward(self, x1, x2):
        """Forward the discriminator."""
        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d



class Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool_SENET(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None,reduction = 3):
        super(Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool_SENET, self).__init__()
        self.num_class = num_class
        self.split = split

        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.senet1 = SELayer(self.channels[0])
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.senet2 = SELayer(self.channels[1])
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.senet3 = SELayer(self.channels[2])
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.senet4 = SELayer(self.channels[3])
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward_onece(self, x):
    #     out = self.layer1(x)
    #     out = self.leaky_relu(out)
    #     out = self.layer2(out)
    #     out = self.leaky_relu(out)
    #     out = self.layer3(out)
    #     return out

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        # xx1 = F.adaptive_max_pool2d(x1,(w,h))
        # xx2 = F.adaptive_max_pool2d(x2,(w,h))
        # enum_x1.append(xx1)
        # enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(len(enum_x1)):
            for j in range(len(enum_x2)):
                if i == 0 and j == 0:
                    continue
                else:
                    temp_d = torch.cat((enum_x1[i], enum_x2[j]), 1)
                    d = torch.cat((d, temp_d), 0)
        return d

    def forward(self, x1, x2):
        """Forward the discriminator."""

        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.senet1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.senet2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.senet3(d)
        d = self.leaky_relu(d)
        d = self.layer4(d)
        d = self.senet4(d)
        d = self.leaky_relu(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool_Self_Atten(nn.Module):
    """Discriminator model with the input size will down and without auxiliary classifier."""

    def __init__(self, num_class, split = None,batch_size = 3,imsize = 512):
        super(Discriminator_NumClass_Single_Col_Split_16_NoAdaptivePool_Self_Atten, self).__init__()
        self.num_class = num_class
        self.split = split
        self.imsize = imsize
        self.batch_size = batch_size
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class*2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2] , self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attn1 = Self_Attn(self.channels[2])
        self.attn2 = Self_Attn(self.channels[3])
        self.sigmoid = nn.Sigmoid()

    def Pair_Enum(self,x1,x2):
        enum_x1 = []
        enum_x2 = []
        w1,h1 = x1.size()[2:]
        w2,h2 = x2.size()[2:]
        assert w1 == w2 and h1 == h2
        assert self.split>0


        w = w1/self.split + 1
        h = h1/self.split + 1
        for i in range(self.split):
            l1 = w*i
            l2 = w*(i+1)
            if l2>w1:
                l1 = w1 - w
                l2 = w1
            for j in range(self.split):
                k1 = h * j
                k2 = h * (j + 1)
                if k2 > h1:
                    k1 = h1 - h
                    k2 = h1
                xx1 = x1[:,:,l1:l2,k1:k2]
                xx2 = x2[:,:,l1:l2,k1:k2]
                enum_x1.append(xx1)
                enum_x2.append(xx2)

        # xx1 = F.adaptive_max_pool2d(x1,(w,h))
        # xx2 = F.adaptive_max_pool2d(x2,(w,h))
        # enum_x1.append(xx1)
        # enum_x2.append(xx2)
        d = torch.cat((enum_x1[0], enum_x2[0]), 1)
        for i in range(len(enum_x1)):
            for j in range(len(enum_x2)):
                if i == 0 and j == 0:
                    continue
                else:
                    temp_d = torch.cat((enum_x1[i], enum_x2[j]), 1)
                    d = torch.cat((d, temp_d), 0)
        return d


    def forward(self, x1, x2):
        """Forward the discriminator."""

        if self.split == None:
            d = torch.cat((x1,x2),1)
        else:
            d = self.Pair_Enum(x1, x2)
        d = self.layer1(d)
        d = self.leaky_relu(d)
        d = self.layer2(d)
        d = self.leaky_relu(d)
        d = self.layer3(d)
        d = self.leaky_relu(d)
        d = self.attn1(d)
        d = self.layer4(d)
        d = self.leaky_relu(d)
        d = self.attn2(d)
        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class Discrimer(nn.Module):
    def __init__(self, numclass=5):
        super(Discrimer, self).__init__()
        self.numclass = numclass
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(256, 512, 1,1, bias=False)
        #self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(512, 1024,3,1,1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(1024, 2048,3,1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(2048)

        #self.downsimple = nn.Conv2d(256,1024,1,1,bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(2048, 1, 3, 1,1, bias=False)

        #self.conv5 = nn.Conv2d(1024,self.numclass,1,1,bias=False)
        #self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        #self.model = nn.Sequential()
        #self.model.add_module('conv1', nn.Conv2d(256, 512, 1, 1, bias=False))
        #self.model.add_module('relu1', nn.LeakyReLU(0.2, inplace=True))
        #self.model.add_module('dropout1',nn.Dropout2d(p=0.5))

        #self.model.add_module('conv2', nn.Conv2d(512, 512, 1, 1, bias=False))
        #self.model.add_module('bnorm2', nn.BatchNorm2d(512))
        #self.model.add_module('relu2', nn.LeakyReLU(0.2, inplace=True))
        #self.model.add_module('dropout2', nn.Dropout2d(p=0.5))

        #self.model.add_module('conv3', nn.Conv2d(512, 1024, 1, 1, bias=False))
        #self.model.add_module('bnorm3', nn.BatchNorm2d(1024))
        #self.model.add_module('relu3', nn.LeakyReLU(0.2, inplace=True))
        #self.model.add_module('dropout3', nn.Dropout2d(p=0.5))

        #self.model.add_module('conv4', nn.Conv2d(64 * 4, 64*8, 3, 1, 1, bias=False))
        #self.model.add_module('bnorm4', nn.BatchNorm2d(64 * 8))
        #self.model.add_module('relu4', nn.LeakyReLU(0.2, inplace=True))

        #self.model.add_module('conv5', nn.Conv2d(1024, 1, 1, 1, bias=False))
        #self.model.add_module('relu5', nn.LeakyReLU(0.2, inplace=True))
        #self.model.add_module('sigmoid', nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, S):
        out = self.conv1(S)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout(out)

        out_real_fake = self.conv4(out)
        out_real_fake = self.sigmoid(out_real_fake)

        #out_label = self.conv5(out)
        return out_real_fake#,out_label



class MS_Deeplab(nn.Module):
    def __init__(self,block,num_classes):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],num_classes)   #changed to fix #4

    def forward(self,x):
        output = self.Scale(x) # for original scale
        output_size = output.size()[2]
        input_size = x.size()[2]

        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(output_size, output_size), mode='bilinear')

        x75 = self.interp1(x)
        output75 = self.interp3(self.Scale(x75)) # for 0.75x scale

        x5 = self.interp2(x)
        output5 = self.interp3(self.Scale(x5))	# for 0.5x scale

        out_max = torch.max(torch.max(output, output75), output5)
        return [output, output75, output5, out_max]

def Res_Ms_Deeplab(num_classes=21):
    model = MS_Deeplab(Bottleneck, num_classes)
    return model

def Res_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

def Get_Discrimer(input_size=(512,512), hidden_dims=512, output_dims=1):
    out_size=outS(input_size[0])
    input_dims = out_size*out_size*256
    model = Discriminator(input_dims=input_dims,hidden_dims=hidden_dims,output_dims=output_dims)
    return model
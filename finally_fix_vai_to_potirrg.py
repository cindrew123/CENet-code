import argparse
import os
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable, grad
import torch.optim as optim
# import scipy.misc
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# xianshi

plt.switch_backend('agg')

import pickle
import cv2
import sys

from deeplab.finally_fix_model import diceCoeffv2, SoftDiceLoss, Res_Deeplab, Discriminator, Discriminator_Down, Discriminator_NumClass_2
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet, VOCDataTestSet
from deeplab.remote_beijing import RemoteBeiJing
from deeplab.isprs import ISPRS
from deeplab.isprs_vai import ISPRS_VAI
from deeplab.isprs_pot_irrg import ISPRS_POT_IRRG
from deeplab.isprs_pot_rgb import ISPRS_POT_RGB

import random
import timeit

start = timeit.default_timer()

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN_BEIJING = np.array((82.90118227, 99.76638055, 96.33826206), dtype=np.float32)
IMG_MEAN_ISPRS = np.array((79.78101224, 81.65397782, 120.85911836), dtype=np.float32)

BATCH_SIZE = 2
DATA_DIRECTORY = '../../datasets/remote_beijing'
DATA_LIST_PATH = '../../datasets/remote_beijing/ImageSets/train.txt'
VAL_DATA_LIST_PATH = '../../datasets/remote_beijing/ImageSets/val.txt'
NUM_CLASSES = 6

# DATA_DIRECTORY = '../../datasets/VOCdevkit/VOC2012'
# DATA_LIST_PATH = '../../datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
# NUM_CLASSES = 21
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9

NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = '../../datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'
RESTORE_FROM = '../../datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--val-data-list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-d", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training epoches.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1, and K is the number of segmented class.
    eg:
    palette:[[0],[1],[2],[3],[4],[5]]
    """
    semantic_map = []
    for colour in palette:
        #print mask
        mask2 = mask.data.cpu().numpy()
        equality = np.equal(mask2, colour)
        #print equality
        #class_map = np.all(equality, axis=-1)
        #print class_map
        semantic_map.append(equality)
        #print sum(equality)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    #print semantic_map
    return semantic_map

# imgt from target , imgs from source
# xiugai  butong
def loss_global_style(imgtrain, imgtest):
    # style_loss = loss_slobal_style(output_train_img, output_test_img)
    # output_train_img ge poti map ,output_test_img 1 VAI
    imgtrain1 = imgtrain[0, ...]
    # imgtrain1 (256L, 64L, 64L),256
    imgtest = imgtest[0, ...]
    loss_style = 0
    # imgtest (256L, 64L, 64L)

    for i in range(6):
        imgtr = (imgtrain1[i, ...]).cpu().detach().numpy()
        imgte = (imgtest[i, ...]).cpu().detach().numpy()
        meantr = np.mean(imgtr)
        meante = np.mean(imgte)
        vartr = np.std(imgtr)
        varte = np.std(imgte)
        meansty = float(meante - meantr)
        varsty = float(varte - vartr)

        loss_style = loss_style + abs(meansty) + abs(varsty)

    loss_style = loss_style/6
    loss_style = torch.tensor(loss_style, requires_grad = True).cuda()
    return loss_style


def loss_calc_G(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()

    return criterion(pred, label)


def loss_calc_D(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()
    criterion = torch.nn.BCELoss().cuda()

    return criterion(pred, label)


def res_loss_calc(real, fake):
    criterion = nn.L1Loss().cuda()

    return criterion(real, fake)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    # b.append(model.layer5)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def critic_params(model):
    b = []
    b.append(model.parameters())
    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels, size, name in data_loader:
            yield (images, labels, size, name)


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def calc_gradient_penalty(D, real_data, fake_data):
    """Calculatge gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_data.size())
    # alpha = alpha.expand(args.batch_size, real_data.nelement() / args.batch_size).contiguous().view(real_data.size())
    alpha = alpha.cuda()

    interpolates = make_variable(alpha * real_data + ((1 - alpha) * fake_data))
    interpolates.requires_grad = True

    disc_interpolates, _ = D(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=make_cuda(
                         torch.ones(disc_interpolates.size())),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = 10 * \
                       ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def loss_plot_sigle(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']
    root = path
    for name, value in hist.iteritems():
        if name == 'steps' or name == 'mIOUs':
            continue
        else:
            y = value
            plt.plot(x, y, label=name)
            plt.xlabel('Iter')
            plt.ylabel(name)
            plt.legend(loc=4)
            plt.grid(True)
            plt.tight_layout()

            loss_path = os.path.join(root, model_name + '_' + name + '.png')

            plt.savefig(loss_path)

            plt.close()


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric2 import ConfusionMatrix
    #print 1
    #print data_list

    ConfM = ConfusionMatrix(class_num)
    
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()
    for m in m_list:
        ConfM.addM(m)
    aveJ, j_list, M, class_iou = ConfM.jaccard()
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
            f.write(str(class_iou) + '\n')

    return aveJ, class_iou


def test(model, step, testloader):
    model.eval()
    interp = nn.Upsample(size=(512, 512), mode='bilinear')
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print 'The''iter of ', index, 'processed'
        image, label, size, name = batch
        size = size[0].numpy()
        _, output = model(Variable(image, volatile=True).cuda())
        output = interp(output).cpu().data[0].numpy()

        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
        gt = gt.flatten()
        output = output.flatten()
        class_F1 = f1_score(gt, output, average=None)
        mean_F1 = f1_score(gt, output, average='weighted')
        #print F1
        #print f1
        #bug

        # if index % 100 == 0 and index !=0:
        #    show_all(gt, output,name[0])
        data_list.append([gt.flatten(), output.flatten()])

    miou,class_iou = get_iou(data_list, args.num_classes,
                   './result/target_potirrg_to_vai_change_numclass_2_softmax_bce_{}.txt'.format(step))
    return miou,class_iou,mean_F1,class_F1
    



def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu
    train_hist = {}
    train_hist['D_loss_d'] = []
    train_hist['D_loss_f'] = []
    train_hist['f_d_loss'] = []
    train_hist['label_loss'] = []
    train_hist['mIOUs'] = []
    train_hist['steps'] = []
    train_hist['global_style_loss'] = []
    train_hist['dice_loss'] = []
    
    evalu = {}
    evalu['mean_F1'] = []
    evalu['class_F1'] = []
    evalu['class_iou'] = []
    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes)
    # model_src = Res_Deeplab(num_classes=args.num_classes)
    # critic = Get_Discrimer(input_size)
    critic = Discriminator_NumClass_2(num_class=args.num_classes)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    # model.load_state_dict(torch.load('../../datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'))


    # model.load_state_dict(torch.load(args.restore_from))
    # model_src.load_state_dict(torch.load(args.restore_from))

    # model.float()
    # model.eval() # use_global_stats = True
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 6 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    model.cuda()
    # model_src.cuda()
    critic.cuda()

    cudnn.benchmark = True

    trainloader = data.DataLoader(ISPRS_VAI('fine', 'train', IMG_MEAN_BEIJING),
                                  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = data.DataLoader(ISPRS_POT_IRRG('fine', 'train', IMG_MEAN_ISPRS),
                                 batch_size=args.batch_size / 2, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    iters_of_epoch = len(testloader)
    target_testloader = data.DataLoader(
        ISPRS_POT_IRRG('fine', 'val', IMG_MEAN_ISPRS),
        batch_size=1, shuffle=False, pin_memory=True
    )

    train_loader = get_inf_iterator(trainloader)
    test_loader = get_inf_iterator(testloader)

    optimizer_f = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                             {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                            lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(critic_params(critic),
                             lr=args.learning_rate_d, betas=(0.9, 0.99))

    interp = nn.Upsample(size=input_size, mode='bilinear')

    # interp = nn.Upsample(size=input_size, mode='bilinear')

    # num_images = len(trainloader)

    max_miou = 0
    c_iou = []
    m_F1 = 0
    c_F1 = []
    best_iter = 0

    num_images = len(testloader)
    criterion = nn.CrossEntropyLoss()
    lamda = 0.1

    writer = SummaryWriter(log_dir='./logs/target_potirrg_to_vai_change_numclass_2_softmax_bce')

    for step in range(args.num_steps):
        model.train()
        critic.train()

        # train F

        # don't accumulate grads in D
        for param in critic.parameters():
            param.requires_grad = False

        # adjust learning rate
        adjust_learning_rate(optimizer_f, step)
        adjust_learning_rate(optimizer_d, step)
        optimizer_f.zero_grad()
        optimizer_d.zero_grad()

        batch_train = next(train_loader)
        batch_test = next(test_loader)
        images_train, labels_train, _, _ = batch_train
        images_test, labels_test, _, _ = batch_test
        images_train = Variable(images_train, requires_grad=False).cuda(args.gpu)
        images_test = Variable(images_test, requires_grad=False).cuda(args.gpu)
        labels_train = Variable(labels_train.long()).cuda(args.gpu)
        out_predict_train, output_train = model(images_train)
        out_last_test, output_test = model(images_test)





        

        output_train = interp(output_train)
        
        gt_onehot1 = mask_to_onehot(labels_train[0,...],[[0],[1],[2],[3],[4],[255]])
        gt_onehot2 = mask_to_onehot(labels_train[1,...],[[0],[1],[2],[3],[4],[255]])
        dice_criterion = SoftDiceLoss(num_classes=args.num_classes, activation='sigmoid').cuda()
        gt_onehot1 = torch.tensor(gt_onehot1)
        gt_onehot2 = torch.tensor(gt_onehot2)
        dice_loss1 = dice_criterion(output_train[0,...], gt_onehot1).item()
        dice_loss2 = dice_criterion(output_train[1,...], gt_onehot2).item()
        dice_loss = dice_loss1 + dice_loss2
        dice_loss = torch.tensor(dice_loss, requires_grad = True).cuda()
        #print dice_loss
        
        
        output_test = interp(output_test)
        ##F1 = fscore(labels_test,output_test,target_testloader,step)
        #labels_test.shape [1,512,512]
        # ALL F1
        #F1 = f1_score(labels_test.detach().numpy(), output_test.cpu().detach().numpy(), average='weighted')
        #print F1
        # each F1
        ##f1 = f1_score(labels_test.detach().numpy(), output_test.cpu().detach().numpy(), average=None)
        ##print f1
        ##bug

        
        
        
        
        D_test_score = critic(F.softmax(output_train[0, ...].unsqueeze(0)), F.softmax(output_test))
        # D_test_score.shape is  1,1,16,16
        D_same_domain_label = Variable(torch.ones(D_test_score.size()[0], 1,
                                                  D_test_score.size()[2],
                                                  D_test_score.size()[3]).cuda())
        # D_same_domain_label.shape is 1,1,16,16

        out_source_loss = loss_calc_G(output_train, labels_train)
        # output_train.shape is (2,6,512,512) 
        # labels_train.shape is (2,512,512)
        #xiugai
        loss_style = loss_global_style(output_train, output_test)
        #loss_style = torch.FloatTensor(loss_style_float)
        
        D_test_loss = loss_calc_D(D_test_score, D_same_domain_label)
        D_loss_f = D_test_loss


        f_d_loss = out_source_loss  + D_loss_f * 0.01 + loss_style * 0.01 + dice_loss * 0.01
        f_d_loss.backward()

        for param in critic.parameters():
            param.requires_grad = True

        output_train1 = output_train[0, ...].unsqueeze(0).detach()
        output_train2 = output_train[1, ...].unsqueeze(0).detach()
        output_test = output_test.detach()

        D_train_score = critic(F.softmax(output_train1), F.softmax(output_train2))
        D_test_score = critic(F.softmax(output_train1), F.softmax(output_test))
        D_train_label = Variable(torch.ones(D_train_score.size()[0], 1,
                                            D_train_score.size()[2],
                                            D_train_score.size()[3]).cuda())
        D_test_label = Variable(torch.zeros(D_test_score.size()[0], 1,
                                            D_test_score.size()[2],
                                            D_test_score.size()[3]).cuda())

        D_train_loss = loss_calc_D(D_train_score, D_train_label)
        D_test_loss = loss_calc_D(D_test_score, D_test_label)
        D_loss_d = (D_train_loss + D_test_loss ) * 0.5

        D_loss_d.backward()

        # update parameters
        optimizer_f.step()
        optimizer_d.step()

        train_hist['D_loss_d'].append(D_loss_d.data.cpu().numpy())
        train_hist['D_loss_f'].append(D_loss_f.data.cpu().numpy())
        train_hist['f_d_loss'].append(f_d_loss.data.cpu().numpy())
        train_hist['label_loss'].append(out_source_loss.data.cpu().numpy())
        train_hist['steps'].append(step)
        train_hist['dice_loss'].append(dice_loss.data.cpu().numpy())
        train_hist['global_style_loss'].append(loss_style.data.cpu().numpy())

        print 'The ', step, 'iter ', 'of' '/', args.num_steps, \
            'completed, D_loss_d = ', D_loss_d.data.cpu().numpy(), \
            'completed, D_loss_f = ', D_loss_f.data.cpu().numpy(), \
            'completed, f_d_loss = ', f_d_loss.data.cpu().numpy(), \
            'completed, label_loss=', out_source_loss.data.cpu().numpy(), \
            'completed, dice_loss=', dice_loss.data.cpu().numpy(), \
            'completed, global_style_loss= ', loss_style.data.cpu().numpy()

        In_fo = {
            'D_loss_d': D_loss_d.data.cpu().numpy(),
            'D_loss_f': D_loss_f.data.cpu().numpy(),
            'f_d_loss': f_d_loss.data.cpu().numpy(),
            'label_loss': out_source_loss.data.cpu().numpy(),
            'dice_loss': dice_loss.data.cpu().numpy(),
            'global_style_loss': loss_style.data.cpu().numpy()
        }

        for tag, value in In_fo.items():
            writer.add_scalar(tag, value, step + 1)

            # save model and count the best iou
        if (step + 1) % args.save_pred_every == 0:
            miou,class_iou,mean_F1,class_F1 = test(model, step + 1, target_testloader)
            evalu['mean_F1'].append(mean_F1)
            evalu['class_F1'].append(class_F1)
            evalu['class_iou'].append(class_iou)
            train_hist['mIOUs'].append(miou)
            if max_miou < miou:
                max_miou = miou
                c_iou = class_iou
                m_F1 = mean_F1
                c_F1 = class_F1
                best_iter = step + 1

            writer.add_scalar('mIOUs', miou, step + 1)
            print "max_miou is : ", max_miou, " best_iter is : ", best_iter, "class_iou is : ", c_iou
            print "mean_F1 is : ", m_F1, "class_F1 is :", c_F1
            # print 'taking sources snapshot ...'
            # torch.save(model_src.state_dict(), osp.join(args.snapshot_dir,
            #                                        'sources_isprs_scenes_v5_' + str(
            #                                            step + 1) + '.pth'))
            print 'taking model snapshot ...'
            torch.save(model.state_dict(), osp.join(args.snapshot_dir,
                                                    'target_potirrg_to_vai_change_numclass_2_softmax_bce_' + str(
                                                        step + 1) + '.pth'))
            print 'taking critic snapshot ...'
            torch.save(critic.state_dict(), osp.join(args.snapshot_dir,
                                                     'potirrg_to_vai_critic_change_numclass_2_softmax_bce_' + str(
                                                         step + 1) + '.pth'))

    np.savetxt('./target_result/target_potirrg_to_vai_change_numclass_2_softmax_bce_mIOUs.txt', train_hist['mIOUs'])
    np.savetxt('./target_result/mean_F1.txt', evalu['mean_F1'])
    np.savetxt('./target_result/class_F1.txt', evalu['class_F1'])
    np.savetxt('./target_result/class_iou.txt', evalu['class_iou'])
    
    loss_plot_sigle(train_hist, './loss_plot_dice_junyun', 'target_potirrg_to_vai_change_numclass_2_softmax_bce')

    end = timeit.default_timer()
    print end - start, 'seconds'
    writer.close()


if __name__ == '__main__':
    main()

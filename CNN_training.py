import sys
#print(sys.version)

# imports and stuff
import os
import numpy as np
from skimage import io
from glob import glob
#from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from gdal_dataloader import RoadDataset
#os.chdir("E:\\SeanK_Work\\Roads\\Road_detection\\Final\\CNN_01_noscl\\")
# os.chdir("C:\\Daisy\\Sean_tiles\\")


# Parameters
WINDOW_SIZE = (64, 64)
STRIDE = 16 # Stride for testing
BASE_LR = 0.001 # Base learn rate
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
MAIN_FOLDER = "./Sean_tiles/"
BATCH_SIZE = 16 # Number of samples in a mini-batch

LABELS = ["non-road", "road"]  # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing - use for equal weighting
CACHE = True # Store the dataset in-memory

DATA_FOLDER = MAIN_FOLDER + 'CNN_inputs/rgb/tile_{}.png'
LABEL_FOLDER = MAIN_FOLDER + 'CNN_inputs/labels/tile_{}.png'


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data, a=0.1)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data, a=0.1)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data)

    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x):
        # print(x.shape)
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D(x)
        softmax = F.softmax(x, dim=1)
        argmax = torch.argmax(softmax, dim=1)
        return x, softmax, argmax


if __name__ == '__main__':

    use_cuda=True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    # Utils
    def get_random_pos(img, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        W, H = img.shape[-2:]
        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2


    def CrossEntropy2d(input, target, weight=None, size_average=True):
        """ 2D version of the cross entropy loss """
        dim = input.dim()
        if dim == 2:
            return F.cross_entropy(input, target, weight, size_average)
        elif dim == 4:
            output = input.view(input.size(0), input.size(1), -1)
            output = torch.transpose(output, 1, 2).contiguous()
            output = output.view(-1, output.size(2))
            target = target.view(-1)
            return F.cross_entropy(output, target, weight, size_average)
        else:
            raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


    def accuracy(input, target, label_values):
        #cm = confusion_matrix(target.argmax(axis=1),
        #                      input.argmax(axis=1),
        #                      range(len(label_values)))
        # Compute kappa coefficient
        #total = np.sum(cm)
        #pa = np.trace(cm) / float(total)
        #pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        #kappa = (pa - pe) / (1 - pe)
        #return kappa
        return 100 * float(np.count_nonzero(input == target)) / target.size


    def sliding_window(top, step=10, window_size=(20, 20)):
        """ Slide a window_shape window across the image with a stride of step """
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]


    def count_sliding_window(top, step=10, window_size=(20, 20)):
        """ Count the number of windows in an image """
        c = 0
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                c += 1
        return c


    def grouper(n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk


    def metrics(predictions, gts, id, file_location, label_values=LABELS):
        cm = confusion_matrix(
            gts,
            predictions,
            labels=range(len(label_values)))

        print("Confusion matrix")
        print(cm)

        print("---")

        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("{} pixels processed".format(total))
        print("Total accuracy: {}%".format(accuracy))

        print("---")

        # Compute recall and precision
        recall = np.zeros(len(label_values))
        precision = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                recall[i] = float(cm[i, i]) / float(np.sum(cm[i, :]))
                precision[i] = float(cm[i, i]) / float(np.sum(cm[:, i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("Recall")
        for l_id, score in enumerate(recall):
            print("{}: {}".format(label_values[l_id], score))
        print("Precision")
        for l_id, score in enumerate(precision):
            print("{}: {}".format(label_values[l_id], score))

        print("---")

        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score")
        for l_id, score in enumerate(F1Score):
            print("{}: {}".format(label_values[l_id], score))

        print("---")

        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe)
        print("Kappa: " + str(kappa))
        with open(file_location + 'Accuracy_tile_{}.txt'.format(id), "w") as text_file:
            print("Confusion matrix", file=text_file)
            print(cm, file=text_file)
            print("{} pixels processed".format(total), file=text_file)
            print("Total accuracy: {}%".format(accuracy), file=text_file)
            print("---", file=text_file)
            print("Recall", file=text_file)
            for l_id, score in enumerate(recall):
                print("{}: {}".format(label_values[l_id], score), file=text_file)
            print("Precision", file=text_file)
            for l_id, score in enumerate(precision):
                print("{}: {}".format(label_values[l_id], score), file=text_file)
            print("---", file=text_file)
            print("F1Score", file=text_file)
            for l_id, score in enumerate(F1Score):
                print("{}: {}".format(label_values[l_id], score), file=text_file)
            print("---", file=text_file)
            print("Kappa: " + str(kappa), file=text_file)
        return kappa


    def class_weight(images, classes):
        # images is a list of image paths
        # classes is list of class numbers
        nclasses = len(classes)
        count = [0] * nclasses
        for i in images:
            count_tmp = [0] * nclasses
            item = io.imread(i)
            for n, c in enumerate(classes):
                count_tmp[n] = len(item[np.where(item == c)])
            count = [x + y for x, y in zip(count, count_tmp)]
        weight = [0] * nclasses
        N = float(sum(count))
        for n, c in enumerate(classes):
            weight[n] = 1.0 - float(count[n])/N
        weight = [round(x, 4) for x in weight]
        return torch.tensor(weight)




    #net = SegNet().double()
    net = SegNet().double().to(device)
    net = net.cuda()

    try:
        from urllib.request import URLopener
    except ImportError:
        from urllib import URLopener

    # Download VGG-16 weights from PyTorch
    vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
        weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

    vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
    mapped_weights = {}
    k_segnet_no_num_batches = net.state_dict().keys()
    k_segnet_no_num_batches = [key for key in k_segnet_no_num_batches if not key.endswith('num_batches_tracked')]
    for k_vgg, k_segnet in zip(vgg16_weights.keys(), k_segnet_no_num_batches):
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
            print("Mapping {} to {}".format(k_vgg, k_segnet))

    try:
        net.load_state_dict(mapped_weights)
        print("Loaded VGG-16 weights in SegNet !")
    except:
        # Ignore missing keys
        pass

    # # Download VGG-19 weights from PyTorch
    # vgg_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
    # if not os.path.isfile('./vgg19_bn-c79401a0.pth'):
    #     weights = URLopener().retrieve(vgg_url, './vgg19_bn-c79401a0.pth')

    # vgg19_weights = torch.load('./vgg19_bn-c79401a0.pth')
    # mapped_weights = {}
    # for k_vgg, k_segnet in zip(vgg19_weights.keys(), net.state_dict().keys()):
    #     if "features" in k_vgg:
    #         mapped_weights[k_segnet] = vgg19_weights[k_vgg]
    #         print("Mapping {} to {}".format(k_vgg, k_segnet))

    # try:
    #     net.load_state_dict(mapped_weights)
    #     print("Loaded VGG-19 weights in SegNet !")
    # except:
    #     # Ignore missing keys
    #     pass


    def test(net, test_ids, all=False, stride=STRIDE, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
        # Use the network on the test set
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
        #     eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
        all_preds = []
        all_gts = []

        # Switch the network to inference mode
        net.eval()

        kappa_all = []

        for img, gt, id_ in zip(test_images, test_labels, test_ids):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                # print(coords)
                # Display in progress results
                if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    #fig = plt.figure()
                    #fig.add_subplot(1, 3, 1)
                    #plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    #fig.add_subplot(1, 3, 2)
                    #plt.imshow(_pred, cmap=plt.cm.gray, vmin=0, vmax=1)
                    #fig.add_subplot(1, 3, 3)
                    #plt.imshow(gt, cmap=plt.cm.gray, vmin=0, vmax=1)
                    #clear_output()
                    #plt.show()

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                #image_patches = torch.from_numpy(image_patches).to(dtype=torch.double)
                image_patches = Variable(torch.from_numpy(image_patches).to(dtype=torch.double).cuda(), volatile=True)

                # Do the inference
                net = net
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred_log_softmax = pred
            pred_prob = np.exp(pred)
            #         pred_prob[:,:,1] = 1.0 + pred_prob[:,:,1]
            pred = np.argmax(pred, axis=-1)

            # Save/display the result
            #clear_output()
            fig = plt.figure(figsize=(15,5))
            fig.add_subplot(1, 3, 1)
            plt.imshow(np.asarray(255 * img, dtype='uint8'))
            if not os.path.exists('./results/spectral_img/'):
                os.mkdir('./results/spectral_img/')
            if not os.path.exists('./results/predictions/'):
                os.mkdir('./results/predictions/')
            if not os.path.exists('./results/ground_truths/'):
                os.mkdir('./results/ground_truths')
            if not os.path.exists('./results/summaries/'):
                os.mkdir('./results/summaries')
            io.imsave('./results/spectral_img/original_tile_{}.png'.format(id_),
                      np.asarray(255 * img, dtype='uint8'))
            fig.add_subplot(1, 3, 2)
            plt.imshow(pred, cmap=plt.cm.gray, vmin=0, vmax=1)
            io.imsave('./results/predictions/inference_tile_{}.png'.format(id_),
                      convert_to_color(pred))
            fig.add_subplot(1, 3, 3)
            plt.imshow(gt, cmap=plt.cm.gray, vmin=0, vmax=1)
            io.imsave('./results/ground_truths/ground_tile_{}.png'.format(id_),
                      convert_to_color(gt))
            plt.tight_layout()
            plt.savefig('./results/summaries/ground_tile_{}.png'.format(id_))
            plt.close()

            all_preds.append(pred)
            all_gts.append(gt)

            # Compute accuracy metrics and save
            if not os.path.exists('./results/train_accuracy/'):
                os.mkdir('./results/train_accuracy/')
            kappa = metrics(pred.ravel(), gt.ravel(), id=id_, file_location="./results/train_accuracy/")
            kappa_all.append(kappa)
            # Save softmax probabilities
            if not os.path.exists('./results/probabilities/'):
                os.mkdir('./results/probabilities/')
            with open('./results/probabilities/prob_{}.txt'.format(id_), 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                outfile.write('# Array shape: {0}\n'.format(pred_log_softmax.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                for data_slice in pred_log_softmax:
                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.
                    np.savetxt(outfile, data_slice, fmt='%-7.2f')

                    # Writing out a break to indicate different slices...
                    outfile.write('# New slice\n')
        if all:
            return all_preds, all_gts
        else:
            return np.sum(kappa_all)

    def train(net, device, optimizer, epochs, scheduler=None, weights=None, save_epoch=5):
        losses = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        #weights = weights.to(dtype=torch.double)
        #weights = torch.nn.Parameter(weights.to(dtype=torch.double).cuda())
        weights = weights.to(device, dtype=torch.double)

        criterion = nn.NLLLoss(weight=weights)
        iter_ = 0

        for e in range(1, epochs + 1):
            net.train()
            for batch_idx, (data, target, _) in enumerate(train_loader):
                #data, target = data.to(dtype=torch.double), target.to(dtype=torch.long)
                #data, target = Variable(data.to(dtype=torch.double).cuda()), Variable(target.to(dtype=torch.long).cuda())
                data, target = data.to(device, dtype=torch.double), target.to(device, dtype=torch.long)
                optimizer.zero_grad()
                output = net(data)
                loss = CrossEntropy2d(output, target, weight=weights)
                loss.backward()
                optimizer.step()

                losses[iter_] = loss.data
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

                if iter_ % 100 == 0 and iter_ != 0:
                    if not os.path.exists('./results/segnet_tmp/'):
                        os.mkdir('./results/segnet_tmp/')
                    #clear_output()
                    rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                    pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                    gt = target.data.cpu().numpy()[0]
                    print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                        e, epochs, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt, LABELS)))
                    plt.plot(mean_losses[:iter_])# and plt.show()
                    #fig = plt.figure()
                    #fig.add_subplot(131)
                    #plt.imshow(rgb)
                    #plt.title('RGB')
                    #fig.add_subplot(132)
                    #plt.imshow(gt, cmap=plt.cm.gray, vmin=0, vmax=1)
                    #plt.title('Ground truth')
                    #fig.add_subplot(133)
                    #plt.title('Prediction')
                    #plt.imshow(pred, cmap=plt.cm.gray, vmin=0, vmax=1)
                    #plt.show()
                    #plt.pause(5)
                    plt.savefig('./results/segnet_tmp/loss_{}_{}'.format(e, iter_))
                    plt.close()
                iter_ += 1

                del (data, target, loss)

            if scheduler is not None:
                scheduler.step()
            if e % save_epoch == 0:
                # Validate (use the largest possible stride, 'stride=min(WINDOW_SIZE)' for faster computing
                kappa = test(net, test_ids, all=False, stride=STRIDE)
                if not os.path.exists('./results/segnet_tmp/'):
                    os.mkdir('./results/segnet_tmp/')
                torch.save(net.state_dict(), './results/segnet_tmp/segnet128_epoch{}_{}'.format(e, kappa))
        if not os.path.exists('./results/segnet_final/'):
            os.mkdir('./results/segnet_final/')
        torch.save(net.state_dict(), './results/segnet_final/segnet_final')


    ##############################################################################################
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    print(LABEL_FOLDER.replace('{}', '*'))
    all_ids = [f.split('tile_')[-1].split('.')[0] for f in all_files]

    # Random tile numbers for train/test split
    random.seed(456)
    #train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
    #train_ids = random.sample(all_ids, 9 * len(all_ids) // 10 + 1)
    train_ids = all_ids
    test_ids = list(set(all_ids) - set(train_ids))

    train_files = [f for f in all_files for x in train_ids if "tile_" + x + ".png" in f]
    #WEIGHTS = class_weight(images=train_files, classes=[0, 1])
    WEIGHTS = torch.tensor([0.05, 0.95])

    print("Tiles for training : ", len(train_ids))
    print("Tiles for testing : ", len(test_ids))

    # = {'train':train_ids, 'valiidation':test_ids}

    #train_set = ISPRS_dataset(train_ids, cache=CACHE)
    train_set = RoadDataset([], r"E:\Data\lwiniwar\planet_2020.vrt",
                            r"E:\Data\lwiniwar\planet_2020_label.vrt",
                            r"E:\Data\lwiniwar\Bbox\ECCC_SMCcentral_BBox_3857.shp")
    print(f"Training dataset: {len(train_set)} patches")
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    base_lr = BASE_LR
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params':[value],'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params':[value],'lr': base_lr / 2}]

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 5, 7], gamma=0.1)

    ######################################################################################################



    train(net, device, optimizer, epochs=10, scheduler=scheduler, weights=WEIGHTS, save_epoch=1)
    #net.load_state_dict(torch.load('E:\\SeanK_Work\\Roads\\Road_detection\\Final\\CNN_01_noscl\\results_01_950\\segnet_final\\segnet_final'))
    all_preds, all_gts = test(net, train_ids, all=True, stride=STRIDE)
    #os.rename(os.getcwd() + '\\results_01_950\\predictions\\',
    #          os.getcwd() + '\\results_01_950\\predictions_val\\')
    #all_preds, all_gts = test(net, list(set(all_ids) - set(test_ids)), all=True, stride=STRIDE)
    #os.rename(os.getcwd() + '\\results\\predictions\\',
    #          os.getcwd() + '\\results_01_950\\predictions_cal_val\\')
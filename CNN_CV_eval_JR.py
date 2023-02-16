import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from urllib.request import URLopener
import tqdm
from osgeo import gdal

from gdal_dataloader import SplitRoadDataset
from CNN_training import SegNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_vgg16_weights(path="./vgg16_bn-6c64b313.pth"):
    vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    if not os.path.isfile(path):
        weights = URLopener().retrieve(vgg_url, path)
    return torch.load(path)

def get_vgg19_weights(path="./vgg19_bn-c79401a0.pth"):
    vgg_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
    if not os.path.isfile(path):
        weights = URLopener().retrieve(vgg_url, path)
    return torch.load(path)

def init_net(num_channels=3):
    net = SegNet(num_channels).double().to(device)
    net = net.cuda()

    # Download VGG-16 weights from PyTorch
    vgg16_weights = get_vgg16_weights()
    # vgg16_weights = get_vgg19_weights()
    mapped_weights = {}
    k_segnet_no_num_batches = net.state_dict().keys()
    k_segnet_no_num_batches = [key for key in k_segnet_no_num_batches if not key.endswith('num_batches_tracked')]
    for k_vgg, k_segnet in zip(vgg16_weights.keys(), k_segnet_no_num_batches):
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
            if k_segnet == 'conv1_1.weight' and num_channels > 3:
                mapped_weights[k_segnet] = torch.concat((mapped_weights[k_segnet],
                                                         mapped_weights[k_segnet][:, [0] * (num_channels-3), :, :]), 1)
                # print("Mapping {} to {}".format(k_vgg, k_segnet))
    try:
        net.load_state_dict(mapped_weights, strict=True)
    except Exception as e:
        # print("Some error occurred during weight mapping... continuing anyway.")
        print(e)
    return net

def train(net, train_loader, weights, base_lr, out_folder, epochs):
    params_dict = dict(net.named_parameters())
    params = [{'params': torch.nn.ParameterList()},
              {'params': torch.nn.ParameterList(),
               'lr': base_lr / 2}]
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params[0]['params'].append(value)
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params[1]['params'].append(value)

    optimizer = optim.Adam(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1,2], gamma=0.1)
    weights = weights.to(device, dtype=torch.double)
    net.train()
    losses = []
    mean_losses = []
    for ep in range(epochs):
        for batch_idx, (data, target, _) in enumerate(tqdm.tqdm(train_loader, colour='#4051b5',
                                                                desc=f'Training in progress (Epoch {ep+1:02d}/{epochs})',
                                                      unit_scale=16)):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            data, target = data.to(device, dtype=torch.double), target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output, o_softmax, o_argmax = net(data)
            loss = F.cross_entropy(output, target, weights, reduction='mean')  # use softmax if we have raw MLP output
            # loss = F.nll_loss(output, target, weights, reduction='mean')  # use nll loss if we have log probabilities (log_softmax output)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data.cpu().numpy()))

            # print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
            #    batch_idx, len(train_loader),
            #     100. * batch_idx / len(train_loader), loss.data, sklearn.metrics.accuracy_score(pred, gt)))

            mean_losses.append(np.mean(losses))
            if batch_idx % 100 == 0 and batch_idx > 0:
                plt.plot(losses, label='Loss')
                plt.plot(mean_losses, label='Running mean (loss)')
                plt.legend()
                plt.savefig(out_folder / f'loss_ep{ep}_{batch_idx}.png')
                plt.close()

                rgb = np.transpose(data.data.cpu().numpy()[0], (1, 2, 0))

                pred = o_softmax.data.cpu().numpy()[0, 1, :, :]
                gt = target.data.cpu().numpy()[0]

                fig = plt.figure(figsize=(20, 8))
                fig.add_subplot(131)
                plt.imshow(rgb[..., :3])
                plt.title('RGB')
                fig.add_subplot(132)
                plt.imshow(gt, cmap=plt.cm.gray, vmin=0, vmax=1)
                plt.title('Ground truth')
                fig.add_subplot(133)
                plt.title('Prediction')
                plt.imshow(pred, cmap=plt.cm.gray, vmin=0, vmax=1)
                plt.tight_layout()
                plt.savefig(out_folder / f'comparison_ep{ep}_{batch_idx}.png')
                plt.close()

            del (data, target, loss)

        if scheduler is not None:
            scheduler.step()
        torch.save(net.state_dict(), out_folder / f'segnet128_ep{ep}.pth')

    torch.save(net.state_dict(), out_folder / f'segnet128_final.pth')

def test(net, test_loader, test_set, out_folder, merge_option='center'):
    if not isinstance(merge_option, list):
        merge_option = [merge_option]
    kernels = {}
    for mo in merge_option:
        if mo == 'center':
            kernel = None
        elif mo == 'linear':
            kernel = np.zeros((test_set.size, test_set.size))
            weights = np.indices((test_set.size // 2 , test_set.size // 2))
            weights = np.linalg.norm(weights, axis=0)
            kernel[:test_set.size // 2, :test_set.size // 2] = np.fliplr(np.flipud(weights))
            kernel[test_set.size // 2:, :test_set.size // 2] = np.fliplr(weights)
            kernel[:test_set.size // 2, test_set.size // 2:] = np.flipud(weights)
            kernel[test_set.size // 2:, test_set.size // 2:] = weights
            kernel = (np.max(kernel) - kernel) / np.max(kernel)
        elif mo == 'gaussian':
            from scipy import signal
            gkern1d = signal.gaussian(test_set.size, std=test_set.size // 4).reshape(test_set.size, 1)
            kernel = np.outer(gkern1d, gkern1d)
        elif mo == 'average':
            kernel = np.ones((test_set.size, test_set.size))
        kernels[mo] = kernel

    net.eval()

    out_matrs = {}
    out_probs = {}
    out_weights = {}
    for mo in merge_option:
        out_matrs[mo] = np.zeros((test_set.xsize, test_set.ysize), int)
        out_probs[mo] = np.zeros((test_set.xsize, test_set.ysize), float)
        out_weights[mo] = np.zeros((test_set.xsize, test_set.ysize), float)

    for batch_idx, (data, target, loc) in enumerate(tqdm.tqdm(test_loader, colour='#16a349',
                                                              desc='Test in progress                 ')):
        data = data.to(device, dtype=torch.double)
        res, out_softmax, out_class = net(data)
        out_softmax = out_softmax.detach().cpu().numpy()
        out_class = out_class.detach().cpu().numpy()
        for batch in range(out_class.shape[0]):
            if 'center' in mo:
                diffmatrix = out_class[batch, ...] + 2 * target.cpu().numpy()[batch, ...]
                xstart = loc[0][batch].cpu().numpy() + test_set.size//4
                ystart = loc[1][batch].cpu().numpy() + test_set.size//4
                out_matrs['center'][xstart:xstart + test_set.size//2,
                ystart:ystart + test_set.size//2] = diffmatrix[test_set.size//4:3*test_set.size//4,
                                                    test_set.size//4:3*test_set.size//4].T
                out_probs['center'][xstart:xstart + test_set.size//2,
                ystart:ystart + test_set.size//2] = out_softmax[batch, test_set.size//4:3*test_set.size//4,
                                                    test_set.size//4:3*test_set.size//4].T
            else:
                for mo, kernel in kernels.items():
                    xstart = loc[0][batch].cpu().numpy()
                    ystart = loc[1][batch].cpu().numpy()
                    out_probs[mo][xstart:xstart + test_set.size,
                    ystart:ystart + test_set.size] += out_softmax[batch, 1, ...].T * kernel
                    out_weights[mo][xstart:xstart + test_set.size,
                    ystart:ystart + test_set.size] += kernel
    for mo in merge_option:
        if merge_option != 'center':
            out_probs[mo] = out_probs[mo] / out_weights[mo]
            out_matrs[mo] = (out_probs[mo] > 0.5).astype(int)

            # if False:
            #     fig = plt.figure(figsize=(7, 3.5), constrained_layout=True)
            #     spec = fig.add_gridspec(ncols=2, nrows=1)
            #     ax0 = fig.add_subplot(spec[0, 0])
            #     ax0.imshow(data.cpu().numpy()[batch, ...].transpose(1, 2, 0))
            #     ax1 = fig.add_subplot(spec[0, 1])
            #     im = ax1.imshow(diffmatrix, vmin=0, vmax=3)
            #
            #     colors = [im.cmap(im.norm(value)) for value in [0, 1, 2, 3]]
            #     labels = ["No road (TN)", "Err. of commission (FP)", "Err. of ommission (FN)", "Correct pred. (TP)"]
            #     # create a patch (proxy artist) for every color
            #     patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in
            #                range(len(colors))]
            #     # put those patched as legend-handles into the legend
            #     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #
            #     plt.show()
    if out_folder is not None:
        for mo in merge_option:
            test_set.write_results(out_matrs[mo].T, str(out_folder / fr"test_pred_{mo}.tif"))
            test_set.write_results(out_probs[mo].T, str(out_folder / fr"test_prob_{mo}.tif"), dtype=gdal.GDT_Float32)
    return out_matrs

def run_single_train_test(train_set, test_set, net, HP, out_folder):
    WEIGHTS = torch.tensor([0.05, 0.95])
    # WEIGHTS = torch.tensor([1., 1.])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=HP['batch_size'], shuffle=True, pin_memory=True, num_workers=5)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=HP['batch_size'], shuffle=True, pin_memory=True, num_workers=5)

    base_lr = HP['base_lr']
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params':[value],'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params':[value],'lr': base_lr / 2}]


    train(net, train_loader, weights=WEIGHTS, base_lr=base_lr, out_folder=out_folder, epochs=HP['epochs'])
    test(net, test_loader, test_set, out_folder)

if __name__ == '__main__':
    from sklearn.model_selection import LeavePOut
    import copy

    HP = {
        "epochs": 10,
        "base_lr": 0.0001,
        # "base_lr": 0.01,
        "batch_size": 16
    }

    # train_set = SplitRoadDataset([], [], r"E:\Data\lwiniwar\aoi_rasters\ps-tile.tif", r"E:\Data\lwiniwar\aoi_rasters\ps-tile_label.tif", False,
    #                              cache=False, augmentation=True, overlap=32, k_split_approx=10)
    # train_set = SplitRoadDataset([], [], r"E:\Data\lwiniwar\planet_2020\L15-0319E-1407N.tif", r"E:\Data\lwiniwar\planet_2020_label\L15-0319E-1407N.tif", False,
    #                                cache=False, augmentation=True, overlap=32, k_split_approx=10)
    # train_set = SplitRoadDataset([], [],
    #                              r"E:\Data\lwiniwar\aoi_rasters\ps-2021.tif",
    #                              r"E:\Data\lwiniwar\aoi_rasters\ps-2021_label.tif",
    #                              r"E:\Data\lwiniwar\aoi\central_aoi_north.shp",
    #                                cache=False, augmentation=True, overlap=32, k_split_approx=10, num_channels=3)
    # train_set = SplitRoadDataset([], [],
    #                              r"E:\Data\lwiniwar\aoi_rasters\re-tile.tif",
    #                              r"E:\Data\lwiniwar\aoi_rasters\re-tile_label.tif",
    #                              None,
    #                              # r"E:\Data\lwiniwar\aoi\central_aoi_north.shp",
    #                                cache=False, augmentation=True, overlap=32, k_split_approx=10, num_channels=3, min_road_pixels=-1)


    train_set = SplitRoadDataset([], [],
                                 r"E:\Data\lwiniwar\jr-data\all.vrt",
                                 r"E:\Data\lwiniwar\jr-data\all_label.vrt",
                                 r"E:\Data\lwiniwar\jr-data\train_polygons.shp",
                                   cache=False, augmentation=True, overlap=112, k_split_approx=1, num_channels=7, min_road_pixels=10,
                                 minv=[0, 0, 0, 0, 0, 0, 0], maxv=[36.8852, 0.6, 25.175, 0.375, 0.625, 0.666667, 57.1994])
    test_set = SplitRoadDataset([], [],
                                 r"E:\Data\lwiniwar\jr-data\all.vrt",
                                 r"E:\Data\lwiniwar\jr-data\all_label.vrt", None,
                                 # r"E:\Data\lwiniwar\jr-data\test_polygons.shp",
                                   cache=False, augmentation=False, overlap=112, k_split_approx=1, num_channels=7, min_road_pixels=-1,
                                 minv=[0, 0, 0, 0, 0, 0, 0], maxv=[36.8852, 0.6, 25.175, 0.375, 0.625, 0.666667, 57.1994])

    # test_set = copy.deepcopy(train_set)
    # perc_train = 80
    # kf = LeavePOut(p=int((100-perc_train) / 100 * train_set.num_k))
    # for LpOidx, (train_index, test_index) in enumerate(kf.split(range(train_set.num_k))):

    LpOidx = 0
    # train_index = [1,2,3,4,5,6,8,9]
    # test_index = [0,7]
    train_index = [0]
    test_index = [0]
    # re-init net
    net = init_net(num_channels=7)
    train_set.set_iterate_set(train_index)
    test_set.set_iterate_set(test_index)
    if False:
        run_single_train_test(train_set, test_set, net, HP, out_folder=Path(f'./results_jr_trainmerge_new_aug_adam'))

        # print("Next Leave-P-Out run:", train_index, test_index)

    net.load_state_dict(torch.load(rf"E:\Data\lwiniwar\results_jr_trainmerge_new_aug_adam\segnet128_final.pth"), strict=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=HP['batch_size'], shuffle=False, pin_memory=True, num_workers=5)
    test(net, test_loader, test_set, Path(f'./results_jr_trainmerge_new_aug_adam'),
         merge_option='average')
    test(net, test_loader, test_set, Path(f'./results_jr_trainmerge_new_aug_adam'),
         merge_option='linear')
    test(net, test_loader, test_set, Path(f'./results_jr_trainmerge_new_aug_adam'),
         merge_option='gaussian')
    # test(net, test_loader, test_set, Path(f'./results_jr_trainmerge_aug_adam'),
    #      merge_option='center')


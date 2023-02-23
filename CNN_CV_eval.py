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

from gdal_dataloader import SplitRoadDataset, RoadDataset
from SegNetModule import SegNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_vgg16_weights(path="./vgg16_bn-6c64b313.pth"):
    vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    if not os.path.isfile(path):
        weights = URLopener().retrieve(vgg_url, path)
    return torch.load(path)

def create_net(num_channels=3, load_vgg_weights=True):
    net = SegNet(num_channels).double().to(device)

    if not load_vgg_weights:
        return net

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
            loss = F.cross_entropy(output, target, weights, reduction='mean')  # use cross-entropy if we have raw MLP output
            # loss = F.nll_loss(o_softmax, target, weights, reduction='mean')  # use nll loss if we have log probabilities (log_softmax output)

            loss.backward()
            optimizer.step()
            losses.append(float(loss.data.cpu().numpy()))

            mean_losses.append(np.mean(losses))
            if batch_idx % 10 == 0:
                plt.plot(losses, label='Loss')
                plt.plot(mean_losses, label='Running mean (loss)')
                plt.xlabel('Training batches')
                plt.ylabel('Loss value')
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

def inference(net, test_loader, test_set, out_folder, merge_option='average'):
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
        elif mo == 'flatroof':
            kernel = np.zeros((test_set.size, test_set.size))
            weights = np.indices((test_set.size // 2 , test_set.size // 2))
            weights = np.linalg.norm(weights, axis=0)
            kernel[:test_set.size // 2, :test_set.size // 2] = np.fliplr(np.flipud(weights))
            kernel[test_set.size // 2:, :test_set.size // 2] = np.fliplr(weights)
            kernel[:test_set.size // 2, test_set.size // 2:] = np.flipud(weights)
            kernel[test_set.size // 2:, test_set.size // 2:] = weights
            kernel[kernel < test_set.size / 4] = test_set.size / 4
            kernel = (np.max(kernel) - kernel) / np.max(kernel)
        elif mo == 'gaussian':
            from scipy import signal
            gkern1d = signal.gaussian(test_set.size, std=test_set.size // 4).reshape(test_set.size, 1)
            kernel = np.outer(gkern1d, gkern1d)
        elif mo == 'average':
            kernel = np.ones((test_set.size, test_set.size))
        else:
            raise NotImplementedError(f"Unknown merge option: '{mo}'.")
        kernels[mo] = kernel + 0.0001  # small epsilon to avoid div by zero

    net.eval()

    out_matrs = {}
    out_probs = {}
    out_weights = {}
    for mo in merge_option:
        out_matrs[mo] = np.zeros((test_set.xsize, test_set.ysize), int)
        out_probs[mo] = np.zeros((test_set.xsize, test_set.ysize), float)
        if mo != 'center':
            out_weights[mo] = np.zeros((test_set.xsize, test_set.ysize), float)

    for batch_idx, (data, target, loc) in enumerate(tqdm.tqdm(test_loader, colour='#16a349',
                                                              desc='Test in progress                 ')):
        data = data.to(device, dtype=torch.double)
        res, out_softmax, out_class = net(data)
        out_softmax = out_softmax.detach().cpu().numpy()
        out_class = out_class.detach().cpu().numpy()
        for batch in range(out_class.shape[0]):
            for mo, kernel in kernels.items():
                if mo == 'center':
                    xstart = loc[0][batch].cpu().numpy() + test_set.size//4
                    ystart = loc[1][batch].cpu().numpy() + test_set.size//4
                    out_matrs['center'][xstart:xstart + test_set.size//2,
                    ystart:ystart + test_set.size//2] = out_class[batch, test_set.size//4:3*test_set.size//4,
                                                        test_set.size//4:3*test_set.size//4].T
                    out_probs['center'][xstart:xstart + test_set.size//2,
                    ystart:ystart + test_set.size//2] = out_softmax[batch, 1, test_set.size//4:3*test_set.size//4,
                                                        test_set.size//4:3*test_set.size//4].T
                else:
                    xstart = loc[0][batch].cpu().numpy()
                    ystart = loc[1][batch].cpu().numpy()
                    out_probs[mo][xstart:xstart + test_set.size,
                    ystart:ystart + test_set.size] += out_softmax[batch, 1, ...].T * kernel
                    out_weights[mo][xstart:xstart + test_set.size,
                    ystart:ystart + test_set.size] += kernel

    for mo in merge_option:
        if mo != 'center':
            out_probs[mo] = out_probs[mo] / out_weights[mo]
            out_matrs[mo] = (out_probs[mo] > 0.5).astype(int)

    if out_folder is not None:
        if not out_folder.is_dir():
            out_folder.mkdir()
        for mo in merge_option:
            test_set.write_results(out_matrs[mo].T, str(out_folder / fr"test_pred_{mo}.tif"))
            test_set.write_results(out_probs[mo].T, str(out_folder / fr"test_prob_{mo}.tif"), dtype=gdal.GDT_Float32)
    return out_matrs

def run_single_train_test(train_set, test_set, net, HP, out_folder):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=HP['batch_size'],
                                               shuffle=True, pin_memory=True, num_workers=5)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=HP['batch_size'],
                                              shuffle=False, pin_memory=True, num_workers=5)

    base_lr = HP['base_lr']
    train(net, train_loader, weights=HP['weights'], base_lr=base_lr, out_folder=out_folder, epochs=HP['epochs'])
    inference(net, test_loader, test_set, out_folder)

if __name__ == '__main__':
    HP = {
        "epochs": 3,
        "base_lr": 0.0001,
        "batch_size": 16,
        'weights': torch.tensor([0.05, 0.95])
    }
    work_dir = Path(r"C:\Users\Lukas\Documents\Data\road-cnn-small\demo_inference")
    run_name = 'experiment1'

    TRAIN = False
    TEST = True


    if TRAIN:
        train_set = SplitRoadDataset([1,2,3,4,5,6,8,9],
                                     r"E:\Data\lwiniwar\aoi_rasters\ps-tile.tif",
                                     r"E:\Data\lwiniwar\aoi_rasters\ps_label_notrails.tif",
                                     None,
                                     augmentation=True, overlap=112, k_split_approx=10, num_channels=3,
                                     min_road_pixels=10, dilate_iter=1,
                                     minv=[49, 155, 157, 1017], maxv=[961, 1181, 1316, 4189])


        # re-init net
        net = create_net(num_channels=3)

        print(f"Next Leave-P-Out run: Train on {train_set.iterate_sets}, test on full dataset")

        # uncomment to continue training from a saved checkpoint
        # net.load_state_dict(torch.load(work_dir / rf"{run_name}\segnet128_ep0.pth"), strict=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=HP['batch_size'], shuffle=True,
                                                   pin_memory=True, num_workers=5)
        train(net, train_loader, weights=HP['weights'], base_lr=HP['base_lr'], out_folder=Path(f'./{run_name}'), epochs=HP['epochs'])

    if TEST:
        net = create_net(num_channels=3, load_vgg_weights=False)
        net.load_state_dict(torch.load(work_dir / rf"segnet128_final.pth"), strict=False)

        test_set = RoadDataset(work_dir / r"planet_data.tif",
                               None,  # no reference labels
                               None,  # no AOI
                               augmentation=False, overlap=112, num_channels=3,
                               min_road_pixels=-1,
                               minv=[49, 155, 157, 1017], maxv=[961, 1181, 1316, 4189])

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=HP['batch_size'], shuffle=False, pin_memory=True, num_workers=5)


        inference(net, test_loader, test_set, work_dir / f'{run_name}',
                  merge_option=['flatroof', 'gaussian','average','linear'])
        inference(net, test_loader, test_set, work_dir / f'{run_name}',
                  merge_option=['center'])



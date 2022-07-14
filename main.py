from __future__ import print_function
import argparse
import pandas as pd
import os
import os.path as osp
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataloader import LLP_dataset, ToTensor, categories
from nets.net_audiovisual import MMIL_Net, LabelSmoothingNCELoss
from utils.eval_metrics import segment_level, event_level, print_overall_metric


def get_LLP_dataloader(args):
    train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                video_dir=args.video_dir, st_dir=args.st_dir,
                                transform=transforms.Compose([ToTensor()]),
                                a_smooth=args.a_smooth, v_smooth=args.v_smooth)
    val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir,
                              video_dir=args.video_dir, st_dir=args.st_dir,
                              transform=transforms.Compose([ToTensor()]))
    test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir,
                               video_dir=args.video_dir, st_dir=args.st_dir,
                               transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=5, pin_memory=True, sampler=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True, sampler=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, sampler=None)

    return train_loader, val_loader, test_loader


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_random_state():
    state = {
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'random_rng': random.getstate(),
        'numpy_rng': np.random.get_state()
    }
    return state


def train_noise_estimator(args, model, train_loader, optimizer, criterion, epoch, logger):
    print(f"begin train_noise_estimator.")
    model.train()

    criterion2 = LabelSmoothingNCELoss(classes=10, smoothing=0.1)

    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), \
                                         sample['video_s'].to('cuda'), \
                                         sample['video_st'].to('cuda'), \
                                         sample['label'].type(torch.FloatTensor).to('cuda')
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')

        optimizer.zero_grad()
        output, a_prob, v_prob, frame_prob, sims_after, mask_after = model(audio, video, video_st, with_ca=False)

        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        a_prob = torch.clamp(a_prob, min=1e-7, max=1 - 1e-7)
        v_prob = torch.clamp(v_prob, min=1e-7, max=1 - 1e-7)

        loss1 = criterion(a_prob, Pa)
        loss2 = criterion(v_prob, Pv)
        loss3 = criterion(output, target)
        loss4 = criterion2(sims_after, mask_after)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}\tLoss4: {:.3f}'.
                  format(epoch, batch_idx * len(audio), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss1.item(),
                         loss2.item(), loss3.item(), loss4.item()))
        if logger:
            logger.add_scalar('loss', loss, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_audio', loss1, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_visual', loss2, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_global', loss3, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_nce', loss4, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)


def calculate_noise_ratio(args, model, train_loader):
    print(f"begin calculate_noise_ratio.")
    model.eval()

    datas = []
    a_prob_list = []
    v_prob_list = []
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), \
                                         sample['video_s'].to('cuda'), \
                                         sample['video_st'].to('cuda'), \
                                         sample['label'].type(torch.FloatTensor).to('cuda')
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')
        output, a_prob, v_prob, frame_prob = model(audio, video, video_st, with_ca=False)[:4]
        a_prob_list.append(torch.mean(a_prob, dim=0).detach().cpu().numpy())
        v_prob_list.append(torch.mean(v_prob, dim=0).detach().cpu().numpy())
        da = {
              'a': a_prob.cpu().detach(),
              'v': v_prob.cpu().detach(),
              'label': target.cpu(),
              'Pa': Pa.cpu().detach(),
              'Pv': Pv.cpu().detach()
        }
        datas.append(da)
        if batch_idx % args.log_interval == 0:
            print('Estimate epoch: 1 [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(audio), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))
    a_prob_mean = np.mean(a_prob_list, axis=0)
    v_prob_mean = np.mean(v_prob_list, axis=0)

    noise_num_v = np.zeros(25)
    noise_num_a = np.zeros(25)
    for data in datas:
        a = data['a']
        v = data['v']
        label = data['label']
        Pa = data['Pa']
        Pv = data['Pv']
        a = a * Pa
        v = v * Pv
        for b in range(len(a)):
            for c in range(25):
                if label[b][c] != 0:
                    if v[b][c] / v_prob_mean[c] < args.v_thres:
                        noise_num_v[c] += 1
                    if a[b][c] / a_prob_mean[c] < args.a_thres:
                        noise_num_a[c] += 1

    event_nums = np.zeros(25)
    labels = pd.read_csv("data/AVVP_train.csv", header=0, sep="\t")["event_labels"].values
    id_to_idx = {id: index for index, id in enumerate(categories)}
    for video_id, label in enumerate(labels):
        ls = label.split(',')
        label_id = [id_to_idx[l] for l in ls]
        for id in label_id:
            event_nums[id] += 1
    v_noise_ratio = np.divide(noise_num_v, event_nums)
    a_noise_ratio = np.divide(noise_num_a, event_nums)
    np.savez(args.noise_ratio_file, audio=a_noise_ratio, visual=v_noise_ratio)


def train_label_denoising(args, model, train_loader, optimizer, criterion, epoch, logger):
    print(f"begin train_label_denoising.")
    model.train()
    criterion2 = LabelSmoothingNCELoss(classes=10, smoothing=args.nce_smooth)

    noise_ratios = np.load(args.noise_ratio_file)
    noise_ratios_a_init = torch.from_numpy(noise_ratios['audio']).to('cuda')
    noise_ratios_v_init = torch.from_numpy(noise_ratios['visual']).to('cuda')
    noise_ratios_a = noise_ratios_a_init.clone()
    noise_ratios_v = noise_ratios_v_init.clone()

    iters_per_epoch = len(train_loader)

    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), \
                                         sample['video_s'].to('cuda'), \
                                         sample['video_st'].to('cuda'), \
                                         sample['label'].type(torch.FloatTensor).to('cuda')
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')
        batch = len(audio)

        if args.warm_up_epoch is not None:
            noise_ratios_a = \
                torch.min(
                    torch.cat(
                        (noise_ratios_a.reshape(1, -1),
                         noise_ratios_a.reshape(1, -1) *
                         ((epoch - 1) * iters_per_epoch + batch_idx) / (args.warm_up_epoch * iters_per_epoch)),
                        dim=0),
                    dim=0)[0]
            noise_ratios_v = \
                torch.min(
                    torch.cat(
                        (noise_ratios_v.reshape(1, -1),
                         noise_ratios_v.reshape(1, -1) *
                         ((epoch - 1) * iters_per_epoch + batch_idx) / (args.warm_up_epoch * iters_per_epoch)),
                        dim=0),
                    dim=0)[0]

        with torch.no_grad():
            output, a_prob, v_prob, frame_prob, sims_after, mask_after = model(audio, video, video_st, with_ca=False)

            a_prob = torch.clamp(a_prob, min=args.clamp, max=1 - args.clamp)
            v_prob = torch.clamp(v_prob, min=args.clamp, max=1 - args.clamp)

            tmp_loss_a = nn.BCELoss(reduction='none')(a_prob, Pa)
            tmp_loss_v = nn.BCELoss(reduction='none')(v_prob, Pv)
            _, sort_index_a = torch.sort(tmp_loss_a, dim=0)
            _, sort_index_v = torch.sort(tmp_loss_v, dim=0)

            pos_index_a = Pa > 0.5
            pos_index_v = Pv > 0.5

            for i in range(25):
                pos_num_a = sum(pos_index_a[:, i].type(torch.IntTensor))
                pos_num_v = sum(pos_index_v[:, i].type(torch.IntTensor))
                numbers_a = torch.mul(noise_ratios_a[i], pos_num_a).type(torch.IntTensor)
                numbers_v = torch.mul(noise_ratios_v[i], pos_num_v).type(torch.IntTensor)
                # remove noise labels for visual
                mask_a = torch.zeros(batch).to('cuda')
                mask_v = torch.zeros(batch).to('cuda')
                if numbers_v > 0:
                    mask_a[sort_index_a[pos_index_v[sort_index_a[:, i], i], i][:numbers_v]] = 1
                    mask_v[sort_index_v[pos_index_v[sort_index_v[:, i], i], i][-numbers_v:]] = 1
                mask = torch.nonzero(torch.mul(mask_a, mask_v)).squeeze(-1).type(torch.LongTensor)
                Pv[mask, i] = 0

                # remove noise labels for audio
                mask_a = torch.zeros(batch).to('cuda')
                mask_v = torch.zeros(batch).to('cuda')
                if numbers_a > 0:
                    mask_a[sort_index_a[pos_index_a[sort_index_a[:, i], i], i][-numbers_a:]] = 1
                    mask_v[sort_index_v[pos_index_a[sort_index_v[:, i], i], i][:numbers_a]] = 1
                mask = torch.nonzero(torch.mul(mask_a, mask_v)).squeeze(-1).type(torch.LongTensor)
                Pa[mask, i] = 0

        optimizer.zero_grad()
        output, a_prob, v_prob, frame_prob, sims_after, mask_after = model(audio, video, video_st, with_ca=True)

        output = torch.clamp(output, min=args.clamp, max=1 - args.clamp)
        a_prob = torch.clamp(a_prob, min=args.clamp, max=1 - args.clamp)
        v_prob = torch.clamp(v_prob, min=args.clamp, max=1 - args.clamp)

        loss1 = criterion(a_prob, Pa)
        loss2 = criterion(v_prob, Pv)
        loss3 = criterion(output, target)
        loss4 = criterion2(sims_after, mask_after)

        loss = loss1 * args.audio_weight + loss2 * args.visual_weight + \
               loss3 * args.video_weight + loss4 * args.nce_weight

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}\tLoss4: {:.3f}'.
                  format(epoch, batch_idx * len(audio), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss1.item(),
                         loss2.item(), loss3.item(), loss4.item()))
        if logger:
            logger.add_scalar('loss', loss, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_audio', loss1, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_visual', loss2, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_global', loss3, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)
            logger.add_scalar('loss_nce', loss4, global_step=(epoch - 1) * len(train_loader) + batch_idx + 1)


def eval(args, model, val_loader, set):
    model.eval()
    print("begin evaluate.")
    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), \
                                             sample['video_s'].to('cuda'), \
                                             sample['video_st'].to('cuda'), \
                                             sample['label'].to('cuda')
            output, a_prob, v_prob, frame_prob = model(audio, video, video_st, with_ca=args.with_ca)[:4]
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)

            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level \
        = print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av)
    return audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument("--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument("--video_dir", type=str, default='data/feats/res152/', help="video dir")
    parser.add_argument("--st_dir", type=str, default='data/feats/r2plus1d_18/', help="video dir")
    parser.add_argument("--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--warm_up_epoch', type=float, default=0.9, help='warm-up epochs')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument("--mode", type=str, default='train_noise_estimator',
                        choices=['train_noise_estimator', 'calculate_noise_ratio',
                                 'train_label_denoising', 'test_noise_estimator', 'test_JoMoLD'],
                        help="with mode to use")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--v_thres', type=float, default=1.8)
    parser.add_argument('--a_thres', type=float, default=0.6)
    parser.add_argument('--noise_ratio_file', type=str)
    parser.add_argument('--a_smooth', type=float, default=1.0)
    parser.add_argument('--v_smooth', type=float, default=0.9)
    parser.add_argument('--audio_weight', type=float, default=2.0)
    parser.add_argument('--visual_weight', type=float, default=1.0)
    parser.add_argument('--video_weight', type=float, default=1.0)
    parser.add_argument('--nce_weight', type=float, default=1.0)
    parser.add_argument('--clamp', type=float, default=1e-7)
    parser.add_argument('--nce_smooth', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2, help='feature temperature number')
    parser.add_argument('--log_interval', type=int, default=700, help='how many batches for logging training status')
    parser.add_argument('--log_file', type=str, help="log file path")
    parser.add_argument('--save_model', type=str, choices=["true", "false"], help='whether to save model')
    parser.add_argument("--model_save_dir", type=str, default='ckpt/', help="model save dir")
    parser.add_argument("--checkpoint", type=str, default='MMIL_Net', help="save model name")
    args = parser.parse_args()

    # print parameters
    print('----------------args-----------------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('----------------args-----------------')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')

    set_random_seed(args.seed)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    save_model = args.save_model == 'true'
    os.makedirs(args.model_save_dir, exist_ok=True)

    model = MMIL_Net(args.num_layers, args.temperature).to('cuda')

    start = time.time()

    if args.mode == 'train_noise_estimator':
        logger = SummaryWriter(args.log_file) if args.log_file else None

        args.with_ca = False
        train_loader, val_loader, test_loader = get_LLP_dataloader(args)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        criterion = nn.BCELoss()

        best_F = 0
        best_model = None
        for epoch in range(1, args.epochs + 1):
            train_noise_estimator(args, model, train_loader, optimizer, criterion, epoch=epoch, logger=logger)
            scheduler.step(epoch)
            print("Validation Performance of Epoch {}:".format(epoch))
            audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, \
            audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve \
                = eval(args, model, val_loader, args.label_val)
            if audio_eve >= best_F:
                best_F = audio_eve
                best_model = copy.deepcopy(model)
                if save_model:
                    state_dict = get_random_state()
                    state_dict['model'] = model.state_dict()
                    state_dict['optimizer'] = optimizer.state_dict()
                    state_dict['scheduler'] = scheduler.state_dict()
                    state_dict['epochs'] = args.epochs
                    torch.save(state_dict, osp.join(args.model_save_dir, args.checkpoint))
            if logger:
                logger.add_scalar("audio_seg", audio_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("visual_seg", visual_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("av_seg", av_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_type_seg", avg_type_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_event_seg", avg_event_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("audio_eve", audio_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("visual_eve", visual_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("av_eve", av_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_type_eve", avg_type_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_event_eve", avg_event_eve, global_step=epoch * len(train_loader))
        if logger:
            logger.close()
        optimizer.zero_grad()
        model = best_model
        print("Test the best model:")
        eval(args, model, test_loader, args.label_test)
    elif args.mode == 'calculate_noise_ratio':
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                    video_dir=args.video_dir, st_dir=args.st_dir,
                                    transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=5, pin_memory=True,
                                  sampler=None)
        resume = torch.load(osp.join(args.model_save_dir, args.checkpoint))
        model.load_state_dict(resume['model'])
        calculate_noise_ratio(args, model, train_loader)
    elif args.mode == 'train_label_denoising':
        logger = SummaryWriter(args.log_file) if args.log_file else None
        args.with_ca = True

        train_loader, val_loader, test_loader = get_LLP_dataloader(args)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        criterion = nn.BCELoss()

        best_F = 0
        best_model = None
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_label_denoising(args, model, train_loader, optimizer, criterion, epoch=epoch, logger=logger)
            scheduler.step(epoch)

            print("Validation Performance of Epoch {}:".format(epoch))
            audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, \
            audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve \
                = eval(args, model, val_loader, args.label_val)
            if audio_eve >= best_F:
                best_F = audio_eve
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            if logger:
                logger.add_scalar("audio_seg", audio_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("visual_seg", visual_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("av_seg", av_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_type_seg", avg_type_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_event_seg", avg_event_seg, global_step=epoch * len(train_loader))
                logger.add_scalar("audio_eve", audio_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("visual_eve", visual_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("av_eve", av_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_type_eve", avg_type_eve, global_step=epoch * len(train_loader))
                logger.add_scalar("avg_event_eve", avg_event_eve, global_step=epoch * len(train_loader))
        if logger:
            logger.close()
        optimizer.zero_grad()
        model = best_model
        if save_model:
            state_dict = get_random_state()
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['scheduler'] = scheduler.state_dict()
            state_dict['epochs'] = args.epochs
            torch.save(state_dict, osp.join(args.model_save_dir, args.checkpoint))
        print(f"Test the best epoch {best_epoch} model:")
        eval(args, model, test_loader, args.label_test)
    elif args.mode == 'test_noise_estimator' or args.mode == 'test_JoMoLD':
        dataset = args.label_test
        args.with_ca = True if args.mode == 'test_JoMoLD' else False
        test_dataset = LLP_dataset(label=dataset,
                                   audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
                                   transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        resume = torch.load(osp.join(args.model_save_dir, args.checkpoint))
        model.load_state_dict(resume['model'])
        eval(args, model, test_loader, dataset)

    end = time.time()
    print(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')


if __name__ == '__main__':
    main()

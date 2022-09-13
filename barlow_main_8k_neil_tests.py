from pathlib import Path
from dataloader import AudioDataset, AudioDatasetEval
import argparse
import json
from torch import nn, optim
import torch
from barlow_model8k import Barlow_diarization, window
import time
import math
import os
import numpy as np
from numpy import dot
import libs.aggHC as AHC
from numpy.linalg import norm
import wandb
import shutil
from einops import rearrange, reduce, repeat

parser = argparse.ArgumentParser(description='Barlow Twins Training')
# parser.add_argument('data', type=Path, metavar='DIR',
                    # help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--load-epoch', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--fc-dim', default=1024, type=int, metavar='N', help='fc layer size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=1 / 32, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log', default='log.json', metavar='DIR', help='log file')
parser.add_argument('--embed', dest='embed', default='False', help='Embed data')
# parser.add_argument('--train-data', default="data/barlowTwins/data/Data/fisher1_supervised.json", help='train dataset json')
parser.add_argument('--train-data', default="/yoav_stg/Shua/barlowTwins/fish_swbd_callfr_8k_vad1.json", help='train dataset json')
#### NEIL TEST ####
parser.add_argument('--dataset', default="/yoav_stg/Shua/barlowTwins/callhome_neil_test_8k_vad3.json", help='dataset json')
parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrTest/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--VADdir', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_3_500/r65_8_1/sid00sg1/data/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir', default='shuaDataDir/barlowTwins/results8k_vicreg/callhome_neil_test_results/', type=Path,
                    metavar='DIR', help='path to reference results directory')
#### NEIL VALIDATE ####                    
parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/callhome_neil_valid_8k_vad3.json", help='dataset json')
parser.add_argument('--refdir1', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrValid/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir1', default='shuaDataDir/barlowTwins/results8k_vicreg/callhome_neil_validate_results/', type=Path,
                    metavar='DIR', help='path to reference results directory')
#### CALLHOME unknown speaker TEST ####
parser.add_argument('--dataset2', default="/yoav_stg/Shua/barlowTwins/callhome_fulltest_8k_vad3.json", help='dataset json')
parser.add_argument('--refdir2', default='/data/shua/speakerDiar/callhome/RTTMs/unknownSpeakerTest/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir2', default='shuaDataDir/barlowTwins/results8k_vicreg/callhome_full_test_results/', type=Path,
                    metavar='DIR', help='path to reference results directory')
#### NEIL TEST vad 1####
parser.add_argument('--dataset3', default="/yoav_stg/Shua/barlowTwins/callhome_neil_test_8k_vad1.json", help='dataset json')
parser.add_argument('--refdir3', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrTest/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--VADdir3', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_1_500/r65_8_1/sid00sg1/data/', type=Path,
                    metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir3', default='shuaDataDir/barlowTwins/results8k_vicreg/callhome_neil_test_results/', type=Path,
                    metavar='DIR', help='path to reference results directory')
def exclude_bias_and_norm(p):
    return p.ndim == 1

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def eval(args, wandb=None, epoch=None):
    model = Barlow_diarization(256, args.fc_dim, 1, 0).cuda()
    path_ = str(args.checkpoint_dir) + '/checkpoint_%d.pth' % epoch

    if os.path.exists(path_):
        ckpt = torch.load(path_, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()
    else:
        print("No model found")
        return
    with torch.no_grad():
        print("MODEL LOADED")
        cutoffMinutes = 31
        subpath = "cpc_5554422_fisher_swbd_callfriend_vad1_500_unsupervised_neilEval_512_256channels_min4"
        overlap_frames = True
        receptive_field = int(8000*0.5)
        frame_step = .25
        start_offset = 0.0
        #### NEIL VALIDATE ####    
        if epoch%10 == 0:
            dataset = AudioDatasetEval(args.dataset1, min_duration=0, max_duration=8000*60*cutoffMinutes, rand=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname = str(args.resdir1) + "/%s_rttms/%s/epoch_%d" % (args.dataset1.split('/')[-1].split('.json')[0], subpath, int(epoch))
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                # name = fname[0].split("/")[-3].replace('.', '_')
                name = fname[0].split("/")[-2]
                chunk = int(fname[0].split("/")[-1].split('.')[0].split('-')[1])
                
                # exit(1)
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    # print(fname, y1.shape[0])
                    if y1.shape[0] <= 4000:
                        pad_width = 4000 - y1.shape[0] + 1
                        # print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                        
                    y1 = rearrange(window(y1, receptive_field, int(8000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                
                if len(y1[0]) < 16000:
                    y1 = rearrange(torch.cat((y1[0][:8000], y1[0])), 't -> 1 t')
                    # print("padded ") 
                # import ipdb; ipdb.set_trace()
                # y3 = torch.from_numpy(np.pad(y1.numpy(), ((0,0), (16000,0)), 'symmetric', reflect_type='odd'))
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                if name in names:
                    names[name].append((chunk, embedings_np))
                else:
                    names[name] = [(chunk, embedings_np)]               
            # for threshold in range(0, 250, 10):
            threshold = "gold_standard"
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            print(threshold)
            dists = []
            for name in names:
                # print(name)
                VAD = open(os.path.join(args.VADdir, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = set(gt_spkrs)
                # print(gt_spkrs)
                AHC.n_clusters_vad_segments(names[name], len(gt_spkrs), os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
            cmd = "../../../shuaUtils/dscore/score.py --collar 0.25 -r "
            for fd in os.listdir(args.refdir1):
                if "._" in fd:
                    continue
                cmd += os.path.join(args.refdir1, fd) 
                cmd += ' '
            cmd += " -s "
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                cmd += os.path.join(rttmDirname, str(threshold), fd) 
                cmd += ' '
            cmd += " > %s/%s_derResults.txt" % (str(rttmDirname), str(threshold))
            outScript = open("%s/%s_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            cmd = "sh %s/%s_derResults.sh" % (str(rttmDirname), str(threshold))
            os.system(cmd)
            der = open("%s/%s_derResults.txt" % (str(rttmDirname), str(threshold))).readlines()[-1].split()[3]
            print("callhome_neil_validate DER% = ", der)

            if wandb != None:
                wandb.log({"Gold Standard callhome_neil_validate DER Loss":float(der)})#, "Threshold":float(bestThreshold)})
        
        return 
        
def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    gpu = args.ngpus_per_node
    args.rank = 0
    args.rank += gpu
    print(args.embed)
    if args.embed == "True":
        print("EMBEDDING")
        eval(args, None, args.load_epoch)
        return
    offset = 2
    pred_steps = 1
    minDuration = 4
    encoder_name = "cpc_5554422_fisher_swbd_callfriend_8k_unsupervised_neilEval_512_256_channel_min4"
    config_defaults = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'fc_layer_size': args.fc_dim,
        'offset': offset,
        'pred_steps': pred_steps,
        'frame_size': 1.0,
        'trainset': args.train_data,
        'encoderType': encoder_name,
        'minDuration': minDuration,
    }
    
    # wandb.agent(sweep_id, function=train)
    args.checkpoint_dir = "./data/barlowTwins/checkpoints/checkpoint_" + encoder_name + '/'
    args.logs = "./logs/log_" + encoder_name + '.json'
    wb = True
    # wandb = None
    if wb:
        wandb.init(project='barlowTwins', entity='shua', config=config_defaults, settings=wandb.Settings(start_method='thread'))
        config = wandb.config
    
    # torch.distributed.init_process_group(backend='nccl', world_size=args.ngpus_per_node, rank=args.ngpus_per_node)
    stats_file = open(args.log,'w')
    # device = torch.device('cuda', 2)
    torch.backends.cudnn.benchmark = True
    model = Barlow_diarization(256, args.fc_dim, pred_steps, offset).cuda()
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[2])
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists

    os.makedirs(args.checkpoint_dir)
    # if (args.checkpoint_dir + '/checkpoint.pth').is_file():
        # ckpt = torch.load(args.checkpoint_dir + '/checkpoint.pth',
                          # map_location='cpu')
        # start_epoch = ckpt['epoch']
        # model.load_state_dict(ckpt['model'])
        # optimizer.load_state_dict(ckpt['optimizer'])
    # else:
    start_epoch = 0
    dataset = AudioDataset(args.train_data, min_duration=minDuration*2*4000, n_samples=minDuration*2*4000)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.ngpus_per_node == 0
    per_device_batch_size = args.batch_size // args.ngpus_per_node
    loader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=None)
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    log_loss = []
    for epoch in range(start_epoch, args.epochs):
        path_ = str(args.checkpoint_dir) + '/checkpoint_%d.pth' % epoch
        #sampler.set_epoch(epoch)
        epochLossAcum = 0.0
        steps = 0.0
        print("Epoch: %d" % epoch)
        for step, y1 in enumerate(loader, start=epoch * len(loader)):
            y1 = rearrange(y1, 'b 1 t -> b t')
            # y2 = np.apply_along_axis(window, 1, y1.cpu().numpy(), 8000, 4000)
            # y3 = torch.from_numpy(np.pad(y1.numpy(), ((0,0), (16000,0)), 'symmetric', reflect_type='odd'))
            y1 = y1.cuda(non_blocking=True)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1)
                # loss = model.forward_vicReg(y1)
                epochLossAcum += float(loss.item())
                steps += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step, learning_rate=lr,
                             loss=loss.item(),
                             time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

            if step % 100 == 0:
                print(float(epochLossAcum / float(steps)))
                if wb:
                    wandb.log({"100 step loss":float(epochLossAcum/steps)})

        print("Epoch Loss = %0.2f" % float(epochLossAcum/steps))
        if wb:
            wandb.log({"Epoch Loss":float(epochLossAcum/steps)})
        # save checkpoint
        state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
        path_ = str(args.checkpoint_dir) + '/checkpoint_%d.pth' % epoch
        torch.save(state, path_)
        eval(args, wandb, epoch)
    # save final model
    torch.save(model.cpc_enc.state_dict(), str(args.checkpoint_dir + '/barlow.pth'))


if __name__ == '__main__':
    main()

from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__)) 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from dataloader import AudioDataset, AudioDatasetEval
import argparse
import json
from torch import nn, optim
import torch
from barlow_model import Barlow_diarization, window
import time
import math
import numpy as np
from numpy import dot
import libs.aggHC as AHC
from numpy.linalg import norm
import wandb
import shutil
import copy
import math
import csv

from einops import rearrange, reduce, repeat
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--load-epoch', default=28, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--fc-dim', default=512, type=int, metavar='N', help='fc layer size')
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
parser.add_argument('--checkpoint-dir', default='data/barlowTwins/checkpoints/checkpoint_cpc_5554422_fisher_swbd_callfriend_vad1_500_unsupervised_neilEval', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log', default='log.json', metavar='DIR', help='log file')
parser.add_argument('--embed', dest='embed', default='True', help='Embed data')

parser.add_argument('--train-data', default="data/barlowTwins/data/Data/fisher_swbd_callhome_16k_vad_lvl1_split500.json", help='train dataset json')
#### NEIL VALIDATE ####    
parser.add_argument('--VADdir', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_1_500/r65_8_1/sid00sg1/data/', type=Path, metavar='DIR', help='path to reference rttms directory')              
parser.add_argument('--dataset', default="/yoav_stg/Shua/barlowTwins/Data/callhome_not_neil_test_vad1_500.json", help='dataset json')
# parser.add_argument('--dataset', default="/yoav_stg/Shua/barlowTwins/Data/callhome_neil_valid_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrValid/', type=Path, metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/unknownSpeakerTest/', type=Path, metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir', default='shuaDataDir/barlowTwins/results/callhome_neil_validate_results/', type=Path, metavar='DIR', help='path to reference results directory')
#### disk6  ####    
# parser.add_argument('--VADdir', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_3_500/vad/', type=Path, metavar='DIR', help='path to reference rttms directory')              
# parser.add_argument('--dataset', default="/yoav_stg/Shua/barlowTwins/callhome_disk_6_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrValid/', type=Path, metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--resdir', default='shuaDataDir/barlowTwins/results/callhome_neil_validate_results/', type=Path, metavar='DIR', help='path to reference results directory')

#### disk7  ####    
# parser.add_argument('--VADdir', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_3_500/6_7_vad/', type=Path, metavar='DIR', help='path to reference rttms directory')              
# parser.add_argument('--dataset', default="/yoav_stg/Shua/barlowTwins/callhome_disk_6_7_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrValid/', type=Path, metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--resdir', default='shuaDataDir/barlowTwins/results/callhome_neil_validate_results/', type=Path, metavar='DIR', help='path to reference results directory')

#### UNKNOWN SPK TEST ####
# parser.add_argument('--dataset', default="data/barlowTwins/data/Data/callhome_unknown_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir', default='/data/shua/speakerDiar/callhome/RTTMs/unknownSpeakerTest/', type=Path,
                    # metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--resdir', default='shuaDataDir/barlowTwins/results/callhome_full_test_results/', type=Path,
                    # metavar='DIR', help='path to reference results directory')

# parser.add_argument('--VADdir1', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_3_500/niel_valid/sid00sg1/data/', type=Path, metavar='DIR', help='path to reference rttms directory')              
# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/Data/callhome_neil_valid_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir1', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrValid/', type=Path, metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--resdir1', default='shuaDataDir/barlowTwins/results/callhome_neil_validate_results/', type=Path, metavar='DIR', help='path to reference results directory')


# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/Data/callhome_neil_test_vad1_500.json", help='dataset json')
# parser.add_argument('--refdir1', default='/data/shua/speakerDiar/callhome/RTTMs/neil2spkrTest/', type=Path, metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--VADdir1', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_1_500/r65_8_1/sid00sg1/data/', type=Path, metavar='DIR', help='path to reference rttms directory')
parser.add_argument('--resdir1', default='shuaDataDir/barlowTwins/results/callhome_neil_test_results_500_vad3_/', type=Path, metavar='DIR', help='path to reference results directory')

#FULL TEST
# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/callhome_fulltest_vad1_16k.json", help='dataset json')
parser.add_argument('--refdir1', default='/data/shua/speakerDiar/callhome/RTTMs/unknownSpeakerTest/', type=Path, metavar='DIR', help='path to reference rttms directory')
# GT VAD 
# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/neil_test_full_file.json", help='dataset json')
parser.add_argument('--VADdir1', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_GT_vad/r65_8_1/sid00sg1/data/', type=Path, metavar='DIR', help='path to reference rttms directory')
  
# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/callhome_disk8_fullFile.json", help='dataset json')
parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/callhome_fulltest_GTvad_.json", help='dataset json')
# parser.add_argument('--VADdir1', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_GT_vad/r65_8_1/sid00sg1/data/', type=Path, metavar='DIR', help='path to reference rttms directory')
 
                    
# parser.add_argument('--dataset1', default="/yoav_stg/Shua/barlowTwins/Data/callhome_unknown_vad3_500.json", help='dataset json')
# parser.add_argument('--refdir1', default='/data/shua/speakerDiar/callhome/RTTMs/unknownSpeakerTest/', type=Path, metavar='DIR', help='path to reference rttms directory')
# parser.add_argument('--resdir1', default='shuaDataDir/barlowTwins/results/callhome_full_test_results/', type=Path,
                    # metavar='DIR', help='path to reference results directory')
# parser.add_argument('--VADdir1', default='/x-wing/datasets/public/callhome/nist_recognition_evaluation_vad_3_500/r65_8_1/sid00sg1/data/', type=Path,
                    # metavar='DIR', help='path to reference rttms directory')

def save_pt(gt, segs, xvecs, plda_scores, dir):
    torch_ex_float_tensor = torch.from_numpy(xvecs).float()
    torch.save(torch_ex_float_tensor, dir + 'train_codes.pt')    
    # torch_plda_float_tensor = torch.from_numpy(plda_scores).float()
    # torch.save(torch_plda_float_tensor, dir + 'plda_scores.pt')
    labels = []
    for seg in segs:
        for gt_seg in gt:
            if seg[0] >= float(gt_seg[0]) and seg[0] < float(gt_seg[0]) + float(gt_seg[1]):
                labels.append(gt_seg[2])
                break
    # CHsrt = {'A', 'B1', 'B2', 'B4', 'B', 'A1', 'B3'}
    CHsrt2num=dict((j,i) for i,j in enumerate(set(labels)))
    # num_labels = alphabet_position(labels)
    num_labels = [CHsrt2num[x] for x in labels]

    pt_labels = torch.tensor(num_labels, dtype=torch.int)
    # import ipdb; ipdb.set_trace()
    torch.save(pt_labels, dir + 'train_labels.pt')
    # ipdb.set_trace()
    
    with open("%s/segments.csv" % dir, "w") as f:
        writer = csv.writer(f)
        writer.writerows(segs)
    
                    
def exclude_bias_and_norm(p):
    return p.ndim == 1

def eval_neil_callhome_reg(args, wandb=None, epoch=None):
    model = Barlow_diarization(128, args.fc_dim, 1, 0).cuda()
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
        overlap_frames  = True
        receptive_field = int(16000*.5)
        frame_step      = .25
        start_offset    = 0.25
        threshold = "gold_standard"
        train_plda = False
        score_barlow = False
        dists      = []
        score      = False
        score_pl   = False
        score_plda = True
        train_sup_plda = False
        score_plda_post_vad = False
        # plda_model_dir = "shuaDataDir/barlowTwins/plda_model_nottest_offset_5_train5em_10spk_vad3_512model_threshold_tune_m46_fish_cf_arab_rus"
        # plda_model_dir = "shuaDataDir/barlowTwins/plda_model_nottest_supervised_train5em_10spk_vad3_512model_threshold_tune"
        plda_model_dir = "shuaDataDir/barlowTwins/plda_model_nottest_supervised_train5em_10spk_vad3_512model_threshold_tune_67_032222"
        # plda_model_dir  = "shuaDataDir/barlowTwins/plda_model_disk67_offset_5_train5em_2spk_vad3_512model_threshold_tune"
        plda_score_dir  = "shuaDataDir/plda_score_0302222_gtvad"
        plda_train_data = "shuaDataDir/plda_train_vad3_sup_nottest_032222_vad3/"
        if train_plda:
            dataset = AudioDatasetEval(args.dataset, min_duration=0, max_duration=16000*60*cutoffMinutes, rand=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname = str(args.resdir)+"_test"
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                # name = fname[0].split("/")[-3].replace('.', '_')
                name = fname[0].split("/")[-2]
                chunk = int(fname[0].split("/")[-1].split('.')[0].split('-')[1])
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    if y1.shape[0] <= 8000:
                        pad_width = 8000 - y1.shape[0] + 1
                        print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                    y1 = rearrange(window(y1, receptive_field, int(16000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                if len(y1[0]) < 32000:
                    y1 = rearrange(torch.cat((y1[0][:16000], y1[0])), 't -> 1 t')
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                if name in names:
                    names[name].append((chunk, embedings_np))
                else:
                    names[name] = [(chunk, embedings_np)]
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            print(threshold)
            
            for name in names:
                # print(name)
                VAD = open(os.path.join(args.VADdir, name+'.segments')).readlines()
                # gt_rttm = open(os.path.join(args.refdir, name+'.rttm')).readlines()
                # gt_spkrs = []
                # for line in gt_rttm:
                    # gt_spkrs.append(line.split()[7])
                # gt_spkrs = set(gt_spkrs)
                cluster_count = 10 #len(gt_spkrs)*4
                dendogram = AHC.n_clusters_vad_segments_get_dendogram(names[name], cluster_count, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD, plda_train_data)
            
            # score dendogram
            ref = open('ref.txt', 'w')
            hyp = open('hyp.txt', 'w')
            cmd = "./tools/md-eval.pl -R ref.txt -S hyp.txt -c 0.25 -1" #  
            for fd in os.listdir(args.refdir):
                if "._" in fd:
                    continue
                ref.write(str(os.path.join(args.refdir, fd)) + '\n')
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                hyp.write(str(os.path.join(rttmDirname, str(threshold), fd)) + '\n')
            cmd += " > %s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))
            print(cmd)
            outScript = open("%s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            ref.close()
            hyp.close()
            cmd = "sh %s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold))
            os.system(cmd)
            
            
            os.makedirs(plda_model_dir, exist_ok=True)    
            os.system("cat %s/*.spk2utt > %s/spk2utt" % (plda_train_data, plda_model_dir))
            os.system("cat %s/*.ivecs > %s/ivecs.ark" % (plda_train_data, plda_model_dir))
            os.system("ivector-mean ark:%s/ivecs.ark %s/mean.vec" % (plda_model_dir, plda_model_dir))
            os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_model_dir, plda_model_dir))
            os.system("time ivector-compute-plda --num-em-iters=5 ark:%s/spk2utt ark:%s/ivecs_mean.ark %s/plda_model" % (plda_model_dir, plda_model_dir, plda_model_dir))
           
            # os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_model_dir, plda_model_dir, plda_model_dir))
            # os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_model_dir, plda_model_dir))      
        if train_sup_plda:
            dataset = AudioDatasetEval(args.dataset, min_duration=0, max_duration=16000*60*cutoffMinutes, rand=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname = str(args.resdir)+"_test"
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                # name = fname[0].split("/")[-3].replace('.', '_')
                name = fname[0].split("/")[-2]
                chunk = int(fname[0].split("/")[-1].split('.')[0].split('-')[1])
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    if y1.shape[0] <= 8000:
                        pad_width = 8000 - y1.shape[0] + 1
                        print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                    y1 = rearrange(window(y1, receptive_field, int(16000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                if len(y1[0]) < 32000:
                    y1 = rearrange(torch.cat((y1[0][:16000], y1[0])), 't -> 1 t')
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                if name in names:
                    names[name].append((chunk, embedings_np))
                else:
                    names[name] = [(chunk, embedings_np)]
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            threshold = "SUPERVISED_PLDA"
            print(threshold)
            
            for name in names:
                if 'iaeu' in name:
                    continue
                # print(name)
                
                vad = open(os.path.join(args.VADdir, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir, name+'.rttm')).readlines()
                gt_segs = []
                for seg in gt_rttm:
                    gt_segs.append((seg.split()[7], float(seg.split()[3]), float(seg.split()[3]) + float(seg.split()[4])))
                segments = []
                embeddings = []
                segnames = names[name]
                for segment in vad:
                    segment = segment.split()
                    segments.append([int(segment[0].split("/")[-1].split('.')[0].split('-')[1]), float(segment[2]), float(segment[3])])
                segments = sorted(segments)
                assert (len(segnames) == len(segments)), (name, len(segnames), len(segments))
                frame_counts = []
                for N in sorted(segnames):
                    embeddings.extend(N[1])
                    frame_counts.append(len(N[1]))
                os.makedirs(plda_train_data, exist_ok=True)
                rttmFd = os.path.join(rttmDirname, str(threshold), name+".rttm")
                RTTM = open(rttmFd, 'w')
                ivecs_fd = open(os.path.join(plda_train_data, name+".ivecs"), 'w')
                utt2spk_fd = open(os.path.join(plda_train_data, name+".utt2spk"), 'w')
                i = 0
                while i < len(embeddings):
                    for segment, frame_len in zip(segments, frame_counts):
                        j = 0
                        start = segment[1] + start_offset
                        duration = float(frame_step)
                        while j < frame_len:
                            label = "Z"
                            for seg in gt_segs:
                                if start <= seg[2] and start >= seg[1]:
                                    label = seg[0]
                            ivecs_fd.write(name+"_"+str(i)+" " + str(embeddings[i].tolist()).replace('[', '[ ').replace(']', ' ]').replace(',', '') + '\n')
                            utt2spk_fd.write(name+"_"+str(i)+' '+name+"_"+label+'\n')
                            i += 1
                            j += 1
                            RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(name, start, duration, label))  
                            start = start + duration
                ivecs_fd.close()
                utt2spk_fd.close()    
                    
            # score dendogram
            ref = open('ref.txt', 'w')
            hyp = open('hyp.txt', 'w')
            cmd = "./tools/md-eval.pl -R ref.txt -S hyp.txt -c 0.25 -1" #  
            for fd in os.listdir(args.refdir):
                if "._" in fd:
                    continue
                ref.write(str(os.path.join(args.refdir, fd)) + '\n')
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                hyp.write(str(os.path.join(rttmDirname, str(threshold), fd)) + '\n')
            cmd += " > %s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))
            print(cmd)
            outScript = open("%s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            ref.close()
            hyp.close()
            cmd = "sh %s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold))
            os.system(cmd)
            
            os.makedirs(plda_model_dir, exist_ok=True)    
            os.system("cat %s/*.utt2spk > %s/utt2spk" % (plda_train_data, plda_model_dir))
            os.system("cat %s/*.ivecs > %s/ivecs.ark" % (plda_train_data, plda_model_dir))
            os.system("./k_utils/utt2spk_to_spk2utt.pl %s/utt2spk > %s/spk2utt" % (plda_model_dir, plda_model_dir))
            os.system("ivector-mean ark:%s/ivecs.ark %s/mean.vec" % (plda_model_dir, plda_model_dir))
            os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_model_dir, plda_model_dir))
            os.system("time ivector-compute-plda --num-em-iters=5 ark:%s/spk2utt ark:%s/ivecs_mean.ark %s/plda_model" % (plda_model_dir, plda_model_dir, plda_model_dir))
           

            os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_model_dir, plda_model_dir, plda_model_dir))
            os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_model_dir, plda_model_dir))
             
        if score_barlow:
            start_offset = 0.25
            dataset      = AudioDatasetEval(args.dataset1, min_duration=0, max_duration=16000*60*cutoffMinutes, rand=False)
            loader       = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname  = str(args.resdir1)+"_test"
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                # name = fname[0].split("/")[-3].replace('.', '_')
                name = fname[0].split("/")[-2]
                chunk = int(fname[0].split("/")[-1].split('.')[0].split('-')[1])
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    if y1.shape[0] <= 8000:
                        pad_width = 8000 - y1.shape[0] + 1
                        # print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                    y1 = rearrange(window(y1, receptive_field, int(16000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                if len(y1[0]) < 32000:
                    y1 = rearrange(torch.cat((y1[0][:16000], y1[0])), 't -> 1 t')
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                if name in names:
                    names[name].append((chunk, embedings_np))
                else:
                    names[name] = [(chunk, embedings_np)]
                    threshold = "gold_standard"
                
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            print(threshold)
            dists = []
            for name in names:
                # print(name)
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = set(gt_spkrs)
                # print(gt_spkrs)
                AHC.n_clusters_vad_segments(names[name], len(gt_spkrs), os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
            cmd = "../../../shuaUtils/dscore/score.py --collar 0.25 --ignore_overlaps -r "
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
            print("callhome_neil_test_vad1 without overlap DER% = ", der)
        
        if score_plda:
            start_offset = 0.0
            dataset      = AudioDatasetEval(args.dataset1, min_duration=0, max_duration=16000*60*cutoffMinutes, rand=False)
            loader       = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname  = str(args.resdir1)+"_test"
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                # name = fname[0].split("/")[-3].replace('.', '_')
                name = fname[0].split("/")[-2]
                chunk = int(fname[0].split("/")[-1].split('.')[0].split('-')[1])
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    if y1.shape[0] <= 8000:
                        pad_width = 8000 - y1.shape[0] + 1
                        # print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                    y1 = rearrange(window(y1, receptive_field, int(16000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                if len(y1[0]) < 32000:
                    y1 = rearrange(torch.cat((y1[0][:16000], y1[0])), 't -> 1 t')
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                if name in names:
                    names[name].append((chunk, embedings_np))
                else:
                    names[name] = [(chunk, embedings_np)]
            os.makedirs(plda_score_dir, exist_ok=True)
            ivecs_fd = open(os.path.join(plda_score_dir, "ivecs.ark"), 'w')
            spk2utt_fd = open(os.path.join(plda_score_dir, "spk2utt"), 'w')
            frame_count_dict = {}
            segments_dict    = {}
            xvecs = {}
            for name in names:
                # print(name)
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = set(gt_spkrs)
                segments = []
                embeddings = []
                for segment in VAD:
                    segment = segment.split()
                    segments.append([int(segment[0].split("/")[-1].split('.')[0].split('-')[1]), float(segment[2]), float(segment[3])])
                segments = sorted(segments)
                segments_dict[name] = segments
                assert (len(names[name]) == len(segments)) , (name, len(names[name]), len(segments))
                frame_counts = []
                for name_ in sorted(names[name]):
                    embeddings.extend(name_[1])
                    frame_counts.append(len(name_[1]))
                frame_count_dict[name] = frame_counts
                i=0
                spk2utt_fd.write(name)
                xvecs[name] = copy.deepcopy(embeddings)
                while i < len(embeddings):
                    ivecs_fd.write(name+"_"+str(i)+"  " + str(embeddings[i].tolist()).replace('[', '[ ').replace(']', ' ]').replace(',', '') + '\n')
                    spk2utt_fd.write(" "+name+"_"+str(i))
                    i+=1
                spk2utt_fd.write('\n')
            ivecs_fd.close()
            spk2utt_fd.close()
            
            # # rec2numspk = open(os.path.join(plda_score_dir, "reco2num-spk"), 'w')
            # # for line in open(os.path.join(plda_score_dir, "spk2utt")).readlines():
                # # rec2numspk.write(line.split()[0] + " 2\n")
            # # rec2numspk.close()
            
            # os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_score_dir, plda_score_dir))
            # os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_score_dir, plda_score_dir, plda_score_dir))
            # os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_score_dir, plda_score_dir))
          
          # # os.system("agglomerative-cluster --threshold=-2.3 ark:%s/scores.ark ark:%s/spk2utt ark,t:%s/labels.rttm" % (plda_score_dir, plda_score_dir, plda_score_dir))
            # # os.system("agglomerative-cluster --reco2num-spk-rspecifier=ark:%s/reco2num-spk ark:%s/scores.ark ark:%s/spk2utt ark,t:%s/labels.rttm" % (plda_score_dir, plda_score_dir, plda_score_dir, plda_score_dir))
            threshold+="_plda"
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            plda_scores = {}
            scores = open("%s/plda_scores_t.ark" % plda_score_dir).readlines()
            i=0
            name = ""
            embedding_dict = {}
            embeddings = []
            while i < len(scores):
                if '[' in scores[i]:
                    name = scores[i].split()[0].split('_')[0]
                    i+=1
                    continue
                elif not ']' in scores[i]:
                    embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').split()]) * -1)
                    i+=1
                    continue
                elif ']' in scores[i]:
                    embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').replace(']', '').split()]) * -1)
                    embedding_dict[name] = copy.deepcopy(embeddings)
                    i+=1
                    name = ""
                    embeddings = []
                    continue
                else:
                    print("Dunno what this line is? ", scores[i])
            # labels_dict = {}
            # for line in open("%s/labels.rttm" % plda_score_dir , 'r').readlines():
                # if line.strip('\n') != "":
                    # name = line.split('_')[0]
                    # if name in labels_dict:
                        # labels_dict[name].append(line.split()[1].replace("\n", ""))
                    # else:
                        # labels_dict[name] = [line.split()[1].replace("\n", "")]
            for name in names:
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                gt = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                    gt.append((line.split()[3], line.split()[4], line.split()[7]))
                gt_spkrs = len(set(gt_spkrs))

                dpm_embeddings =  np.stack(xvecs[name])
                segs = []
                i = 0
                while i < len(xvecs[name]):
                    for segment, frame_len in zip(segments_dict[name], frame_count_dict[name]):
                        j = 0
                        start = segment[1] + start_offset
                        while j < frame_len:
                            segs.append((start, start + float(frame_step)))
                            start = start + float(frame_step)
                            i += 1
                            j += 1
                
                # import ipdb; ipdb.set_trace()
                outdir = "/yoav_stg/Shua/Datasets/callhome_unsup/DeepDPM/%s_%s_SPK_N2D/" % (name.upper(), str(gt_spkrs))
                os.makedirs(outdir, exist_ok=True)
                save_pt(gt, segs, dpm_embeddings, None, outdir)
                print(name, len(segs), dpm_embeddings.shape)
                if gt_spkrs > 8:
                    AHC.n_clusters_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], gt_spkrs, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
                    # AHC.clusterThreshold_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], .8, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
                # import ipdb; ipdb.set_trace()
                # AHC.clusterThreshold_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], 11.0, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD, None)
            
            cmd = "../../../shuaUtils/dscore/score.py --collar 0.25 -r " #  
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
            print("CallHome DER% = ", der)
            
            cmd = "../../../shuaUtils/dscore/score.py --ignore_overlaps --collar 0.25 -r " #  
            for fd in os.listdir(args.refdir1):
                if "._" in fd:
                    continue
                cmd += os.path.join(args.refdir1, fd) 
                cmd += ' '
            cmd += " -s "
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                cmd += os.path.join(rttmDirname, str(threshold), fd) 
                cmd += ' '
            cmd += " > %s/%s_derResults_wo_overlap.txt" % (str(rttmDirname), str(threshold))
            outScript = open("%s/%s_derResults_wo_overlap.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            
            cmd = "sh %s/%s_derResults_wo_overlap.sh" % (str(rttmDirname), str(threshold))
            os.system(cmd)
            der = open("%s/%s_derResults_wo_overlap.txt" % (str(rttmDirname), str(threshold))).readlines()[-1].split()[3]
            print("CallHome DER wo overlap% = ", der)
            
            ref = open('%s/ref.txt' % plda_score_dir, 'w')
            hyp = open('%s/hyp.txt' % plda_score_dir, 'w')
            cmd = "./tools/md-eval.pl -R %s/ref.txt -S %s/hyp.txt -c 0.25 -1" % (plda_score_dir, plda_score_dir) #  
            fds = []
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                
                name = fd.split('/')[-1].split('.')[0]
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = len(set(gt_spkrs))
                if True or gt_spkrs == 7:
                    hyp.write(str(os.path.join(rttmDirname, str(threshold), fd)) + '\n')
                    fds.append(fd)
            for fd in os.listdir(args.refdir1):
                if "._" in fd:
                    continue
                if fd in fds:
                    ref.write(str(os.path.join(args.refdir1, fd)) + '\n')
            cmd += " > %s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))
            print(cmd)
            outScript = open("%s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            ref.close()
            hyp.close()
            cmd = "sh %s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold))
            # os.system(cmd)
        
        if score_plda_post_vad:
            start_offset = 0.0
            dataset = AudioDatasetEval(args.dataset1, min_duration=0, max_duration=16000*60*cutoffMinutes, rand=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=None)
            rttmDirname = str(args.resdir1)+"_test"
            os.makedirs(rttmDirname, exist_ok=True)
            names = {}
            for step, data4eval in enumerate(loader, start=len(loader)):
                y1, fname = data4eval
                name = fname[0].split("/")[-1].split('.wav')[0]
                if overlap_frames:
                    y1 = y1.numpy().squeeze().squeeze()
                    y1 = np.pad(y1, (int(receptive_field), 0))
                    if y1.shape[0] <= 8000:
                        pad_width = 8000 - y1.shape[0] + 1
                        print(y1.shape, pad_width, fname)
                        y1 = np.pad(y1, (0, pad_width), mode='mean')
                    y1 = rearrange(window(y1, receptive_field, int(16000*frame_step)), 't -> 1 t')
                else:
                    y1 = rearrange(y1, 'b 1 t -> b t')
                if len(y1[0]) < 32000:
                    y1 = rearrange(torch.cat((y1[0][:16000], y1[0])), 't -> 1 t')
                y1 = y1.cuda(non_blocking=True)
                z, z1 = model.embed(y1)
                embedings_np = np.nan_to_num(z1.cpu().detach().numpy())
                
                # if name == "iaau":
                    # import matplotlib
                    # import matplotlib.pyplot as plt
                    # import seaborn as sns
                    # plt.matshow(np.transpose(embedings_np))
                    # plt.savefig('embeddings.png')
                    # exit(1)
                    # # fig = plt.figure()
                    # # plt.imshow(np.transpose(embedings_np))
                    # # plt.savefig('embeddings.png')
                    # # exit(1)
                trimmed_emmbeddings = []
                curr_time = 0 + start_offset
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                vad_segments = []
                yes_no = []
                previous_end = 0.0
                for line in VAD:
                    line = line.strip('\n').split()
                    begin = float(line[2])
                    end = float(line[3])
                    # vad_segments.append((begin, end))
                    if previous_end < end:
                       if previous_end > begin:
                           begin = previous_end
                       vad_segments.append((begin, end))
                       previous_end = end
                    else:
                        # print(previous_end, name, line)
                        vad_segments.append((previous_end, previous_end))
                for embedding in embedings_np:
                    trimmed_emmbeddings.append(embedding)
                    yes_no.append(1)
                    # flag = False
                    # # 
                    # for seg in vad_segments:
                        # # if curr_time+(frame_step) <= math.ceil(seg[1]*4)/4.0 and curr_time >= math.floor(seg[0]*4)/4.0:
                        # if curr_time+(frame_step) <= seg[1] and curr_time >= seg[0]:
                            # trimmed_emmbeddings.append(embedding)
                            # flag = True
                            # break
                    # if flag:
                        # yes_no.append(1)
                    # else:
                        # yes_no.append(0)
                    # curr_time += frame_step
                names[name] = (trimmed_emmbeddings, yes_no)
            plda_score_dir = "shuaDataDir/plda_score_GTVAD"
            os.makedirs(plda_score_dir, exist_ok=True)
            ivecs_fd = open(os.path.join(plda_score_dir, "ivecs.ark"), 'w')
            spk2utt_fd = open(os.path.join(plda_score_dir, "spk2utt"), 'w')
            for name in names:
                # print(name)
                # if name == "iaaf":
                    # import ipdb; ipdb.set_trace()
                    # print("")
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = set(gt_spkrs)
                i=0
                spk2utt_fd.write(name)
                while i < len(names[name][0]):
                    ivecs_fd.write(name+"_"+str(i)+"  " + str(names[name][0][i].tolist()).replace('[', '[ ').replace(']', ' ]').replace(',', '') + '\n')
                    spk2utt_fd.write(" "+name+"_"+str(i))
                    i+=1
                spk2utt_fd.write('\n')
            ivecs_fd.close()
            spk2utt_fd.close()
            os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_score_dir, plda_score_dir))
            os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_score_dir, plda_score_dir, plda_score_dir))
            os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_score_dir, plda_score_dir))

            # # os.system("ivector-subtract-global-mean plda_model/mean.vec ark:shuaDataDir/plda_score/ivecs.ark ark:shuaDataDir/plda_score/ivecs_mean.ark")
            # # os.system("time ivector-plda-scoring-dense plda_model/plda_model ark:shuaDataDir/plda_score/spk2utt ark:shuaDataDir/plda_score/ivecs_mean.ark ark:shuaDataDir/plda_score/scores.ark")
            # # os.system("time copy-feats ark:shuaDataDir/plda_score/scores.ark ark,t:shuaDataDir/plda_score/plda_scores_t.ark")
            threshold+="_plda"
            os.makedirs(os.path.join(rttmDirname, str(threshold)), exist_ok=True)
            plda_scores = {}
            scores = open("%s/plda_scores_t.ark" % plda_score_dir).readlines()
            i=0
            name = ""
            embedding_dict = {}
            embeddings = []
            while i < len(scores):
                if '[' in scores[i]:
                    name = scores[i].split()[0].split('_')[0]
                    i+=1
                    continue
                elif not ']' in scores[i]:
                    embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').split()]))
                    i+=1
                    continue
                elif ']' in scores[i]:
                    embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').replace(']', '').split()]))
                    embedding_dict[name] = copy.deepcopy(embeddings)
                    i+=1
                    name = ""
                    embeddings = []
                    continue
                else:
                    print("Dunno what this line is? ", scores[i])
            for name in names:
                VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()                
                gt_rttm = open(os.path.join(args.refdir1, name+'.rttm')).readlines()
                gt_spkrs = []
                for line in gt_rttm:
                    gt_spkrs.append(line.split()[7])
                gt_spkrs = set(gt_spkrs)
                AHC.n_clusters_plda_post_vad_segments(names[name][1], embedding_dict[name], len(gt_spkrs), os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
                # AHC.clusterThreshold_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], .66, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
            
            # cmd = "../../../shuaUtils/dscore/score.py --collar 0.25 -r " #  
            # for fd in os.listdir(args.refdir1):
                # if "._" in fd:
                    # continue
                # cmd += os.path.join(args.refdir1, fd) 
                # cmd += ' '
            # cmd += " -s "
            # for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                # cmd += os.path.join(rttmDirname, str(threshold), fd) 
                # cmd += ' '
            # cmd += " > %s/%s_derResults.txt" % (str(rttmDirname), str(threshold))
            # outScript = open("%s/%s_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            # outScript.write(cmd)
            # outScript.close()
            # cmd = "sh %s/%s_derResults.sh" % (str(rttmDirname), str(threshold))
            # os.system(cmd)
            # der = open("%s/%s_derResults.txt" % (str(rttmDirname), str(threshold))).readlines()[-1].split()[3]
            # print("CallHome DER% = ", der)
            
            # cmd = "../../../shuaUtils/dscore/score.py --ignore_overlaps --collar 0.25 -r " #  
            # for fd in os.listdir(args.refdir1):
                # if "._" in fd:
                    # continue
                # cmd += os.path.join(args.refdir1, fd) 
                # cmd += ' '
            # cmd += " -s "
            # for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                # cmd += os.path.join(rttmDirname, str(threshold), fd) 
                # cmd += ' '
            # cmd += " > %s/%s_derResults_wo_overlap.txt" % (str(rttmDirname), str(threshold))
            # outScript = open("%s/%s_derResults_wo_overlap.sh" % (str(rttmDirname), str(threshold)), 'w')
            # outScript.write(cmd)
            # outScript.close()
            
            # cmd = "sh %s/%s_derResults_wo_overlap.sh" % (str(rttmDirname), str(threshold))
            # os.system(cmd)
            # der = open("%s/%s_derResults_wo_overlap.txt" % (str(rttmDirname), str(threshold))).readlines()[-1].split()[3]
            # print("CallHome DER wo overlap% = ", der)
            

            ref = open('%s/ref.txt' % plda_score_dir, 'w')
            hyp = open('%s/hyp.txt' % plda_score_dir, 'w')
            cmd = "./tools/md-eval.pl -R %s/ref.txt -S %s/hyp.txt -c 0.25 -1 -u uem.rttm" % (plda_score_dir, plda_score_dir) #  
            for fd in os.listdir(args.refdir1):
                if "._" in fd:
                    continue
                ref.write(str(os.path.join(args.refdir1, fd)) + '\n')
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                hyp.write(str(os.path.join(rttmDirname, str(threshold), fd)) + '\n')
            cmd += " > %s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))
            print(cmd)
            outScript = open("%s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            ref.close()
            hyp.close()
            cmd = "sh %s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold))
            # os.system(cmd)  
        if score:
            cmd = "../../../shuaUtils/dscore/score.py --ignore_overlaps --collar 0.25 -r " #  
            for fd in os.listdir(args.refdir):
                if "._" in fd:
                    continue
                cmd += os.path.join(args.refdir, fd) 
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
            print("CallHome DER% = ", der)
                       
        if score_pl:
            ref = open('ref.txt', 'w')
            hyp = open('hyp.txt', 'w')
            cmd = "./tools/md-eval.pl -R ref.txt -S hyp.txt -c 0.25 -1" #  
            for fd in os.listdir(args.refdir):
                if "._" in fd:
                    continue
                ref.write(str(os.path.join(args.refdir, fd)) + '\n')
            for fd in os.listdir(os.path.join(rttmDirname, str(threshold))):
                hyp.write(str(os.path.join(rttmDirname, str(threshold), fd)) + '\n')
            cmd += " > %s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))
            print(cmd)
            outScript = open("%s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold)), 'w')
            outScript.write(cmd)
            outScript.close()
            ref.close()
            hyp.close()
            cmd = "sh %s/%s_md_eval_derResults.sh" % (str(rttmDirname), str(threshold))
            os.system(cmd)
            # der = open("%s/%s_md_eval_derResults.txt" % (str(rttmDirname), str(threshold))).readlines()[-1].split()[3]
            # print("CallHome DER% = ", der)
        return 
        
       
def main():
    args = parser.parse_args()
    print(args)
    args.ngpus_per_node = torch.cuda.device_count()
    gpu = args.ngpus_per_node
    if args.embed == "True":
        print("EMBEDDING")
        eval_neil_callhome_reg(args, None, args.load_epoch)
        # eval(args, None, args.load_epoch)
        return


if __name__ == '__main__':
    main()

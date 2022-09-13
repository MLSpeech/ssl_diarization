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
import random
from sklearn.cluster import AgglomerativeClustering



def generate_pair(ivecs, speaker_a=None, speaker_b=None, segment_sample_count=10):
    # if speaker_a == None:
        # speaker_a = random.choice(list(ivecs.keys()))    
    # if speaker_b == None:
        # speaker_b = random.choice(list(ivecs.pop(speaker_a).keys())) 
    if speaker_a == None and speaker_b == None:        
        speaker_a, speaker_b = random.sample(ivecs.keys(), 2)
    assert speaker_a != speaker_b
    ivec_pair = random.sample(ivecs[speaker_a], segment_sample_count)
    # import ipdb; ipdb.set_trace()
    ivec_pair.extend(random.sample(ivecs[speaker_b], segment_sample_count))
    return ivec_pair
    
def cos_similarity(a, b):
   return dot(a, b)/(norm(a)*norm(b))

all_ivecs = open("shuaDataDir/barlowTwins/plda_model_nottest_supervised_train5em_10spk_vad3_512model_threshold_tune_67_032222/ivecs.ark", 'r').readlines()
ivecs_dict = {}
for line in all_ivecs:
    spk = line.split()[0].split('_')[0]
    if spk in ivecs_dict:
        ivecs_dict[spk].append(line)
    else:
        ivecs_dict[spk] = [line]


plda_model_dir = "shuaDataDir/barlowTwins/plda_model_nottest_supervised_train5em_10spk_vad3_512model_threshold_tune_67_032222"
plda_score_dir  = "shuaDataDir/plda_score_0302222"

pair_count = 20000
if True:
    ivecs_fd = open(os.path.join(plda_score_dir, "ivecs.ark"), 'w')
    spk2utt_fd = open(os.path.join(plda_score_dir, "spk2utt"), 'w')
    for i in range(pair_count):
        pair = generate_pair(ivecs_dict)
        spk2utt_fd.write(str(i))
        for ivec in pair:
            # import ipdb; ipdb.set_trace()
            new_ivec = ivec.split()[0] + '_' + str(i) + ' ' + ' '.join(ivec.split()[1:]) + '\n'
            ivecs_fd.write(new_ivec)
            spk2utt_fd.write(" "+ivec.split()[0]+'_'+str(i))
        spk2utt_fd.write('\n')
        
    ivecs_fd.close()
    spk2utt_fd.close()

    os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_score_dir, plda_score_dir))
    os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_score_dir, plda_score_dir, plda_score_dir))
    os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_score_dir, plda_score_dir))

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
gt_np = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
for i in range(5, 20):
    threshold = i/20.0
    print("threshold: ", threshold)
    diff_sum = 0.0
    for j in range(pair_count):
        aggloclust=AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=threshold, linkage='average').fit(embedding_dict[str(j)])
        # print(aggloclust.labels_)
        hyp_np = np.array(aggloclust.labels_)
        diff_sum += min(np.sum(gt_np != hyp_np), np.sum(gt_np != np.flip(hyp_np)))
    print(diff_sum / pair_count)
exit(1)
import ipdb; ipdb.set_trace()   

i = 0
ivecs_fd = open(os.path.join(plda_score_dir, "ivecs.ark"), 'w')
spk2utt_fd = open(os.path.join(plda_score_dir, "spk2utt"), 'w')
spk2utt_fd.write('a')
spkSet = ['iaaa', 'iaac']
maxSegs = 10
frameCountPerSpk = {}
for spk in spkSet:
    frameCountPerSpk[spk] = 0
for line in all_ivecs:
    seg_spk = line.split()[0].split('_')[0]
    if seg_spk in spkSet:
        if frameCountPerSpk[seg_spk] >= maxSegs:
            continue
        else: 
            frameCountPerSpk[seg_spk] += 1
            ivecs_fd.write(line)
            spk2utt_fd.write(" "+line.split()[0])

i+=1
spk2utt_fd.write('\n')
ivecs_fd.close()
spk2utt_fd.close()
os.system("ivector-subtract-global-mean %s/mean.vec ark:%s/ivecs.ark ark:%s/ivecs_mean.ark" % (plda_model_dir, plda_score_dir, plda_score_dir))
os.system("time ivector-plda-scoring-dense %s/plda_model ark:%s/spk2utt ark:%s/ivecs_mean.ark ark:%s/scores.ark" % (plda_model_dir, plda_score_dir, plda_score_dir, plda_score_dir))
os.system("time copy-feats ark:%s/scores.ark ark,t:%s/plda_scores_t.ark" % (plda_score_dir, plda_score_dir))

AHC.clusterThreshold_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], .75, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)


# plda_score_dir = "shuaDataDir/barlowTwins/plda_model_nottest_offset_5_train5em_10spk_vad3_512model_threshold_tune"

# scores = open("%s/plda_scores_t.ark" % plda_score_dir).readlines()
# i=0
# names = []
# name = ""
# embedding_dict = {}
# embeddings = []
# while i < len(scores):
    # if '[' in scores[i]:
        # name = scores[i].split()[0].split('_')[0]
        # names.append(name)
        # i+=1
        # continue
    # elif not ']' in scores[i]:
        # embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').split()]))
        # i+=1
        # continue
    # elif ']' in scores[i]:
        # embeddings.append(np.array([float(ele) for ele in scores[i].replace('/n', '').replace(']', '').split()]))
        # embedding_dict[name] = copy.deepcopy(embeddings)
        # i+=1
        # name = ""
        # embeddings = []
        # continue
    # else:
        # print("Dunno what this line is? ", scores[i])
# names = list(set(names))
# for j in range(len(names)-1):
    # # VAD = open(os.path.join(args.VADdir1, name+'.segments')).readlines()
    # # AHC.n_clusters_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], len(gt_spkrs), os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
    # # AHC.clusterThreshold_plda_vad_segments(frame_count_dict[name], embedding_dict[name], segments_dict[name], .75, os.path.join(rttmDirname, str(threshold), name+".rttm"), name, frame_step, start_offset, False, None, VAD)
    # for i in embedding_dict[names[j]]:
        # print(len(i))
    
    # # embedings_np = embedding_dict[names[j]]
    # # embedings_np1 = embedding_dict[names[j+1]]
    # # cos_sim = cos_similarity(embedings_np[0], embedings_np[-1]) 
    # # cos_sim1 = cos_similarity(embedings_np[0], embedings_np1[0]) 
    # # print('%.2f'%cos_sim, '%.2f'%cos_sim1)
    
    
    
    
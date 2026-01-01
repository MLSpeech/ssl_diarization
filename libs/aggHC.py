from sklearn.cluster import AgglomerativeClustering
import os
# from sklearn.datasets.samples_generator import make_blobs
import numpy as np

# Optional visualization deps (not required for clustering/RTTM writing)
try:  # pragma: no cover
    import plotly.figure_factory as ff  # type: ignore
except Exception:  # pragma: no cover
    ff = None
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def labels2RTTM(labels, rttmFd, filename, stepSize=1, offset=0):
    RTTM = open(rttmFd, 'w')
    i = 0
    while i < len(labels):
        start = float(i*stepSize) + offset*stepSize
        duration = float(stepSize)
        while i < len(labels)-1 and labels[i] == labels[i+1]:
            duration += stepSize
            i += 1
        RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(filename, start, duration, labels[i]))
        i += 1
        
def clusterThreshold(embeddings, threshold, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None):
    aggloclust=AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=threshold).fit(embeddings)
    if print_labels:
        print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM(aggloclust.labels_, rttmOut, filename, stepSize, offset)
    return aggloclust.labels_
        
def n_clusters(embeddings, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None):
    aggloclust=AgglomerativeClustering(n_clusters=N, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    if print_labels:
        print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM(aggloclust.labels_, rttmOut, filename, stepSize, offset)
    return aggloclust.labels_

# def n_clusters(embeddings, N, rttmOut=None):
    # aggloclust=AgglomerativeClustering(n_clusters=N, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='single', distance_threshold=None).fit(embeddings)
    # print(aggloclust.labels_)
    # if rttmOut != None:
        # labels2RTTM(aggloclust.labels_, rttmOut, .2)
    # return aggloclust.labels_

def n_clustersCos(embeddings, N, rttmOut=None):
    aggloclust=AgglomerativeClustering(n_clusters=N, affinity='cosine', memory=None, connectivity=None, compute_full_tree='auto', linkage='single', distance_threshold=None).fit(embeddings)
    print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM(aggloclust.labels_, rttmOut, 1.28)
    return aggloclust.labels_


def labels2RTTM_vad(labels, rttmFd, filename, stepSize=1, offset=0, vad=None):
    # if vad != None:
        # VAD = open(vad).readlines()
    RTTM = open(rttmFd, 'w')
    i = 0
    while i < len(labels):
        start = float(i*stepSize) + offset
        duration = float(stepSize)
        while i < len(labels)-1 and labels[i] == labels[i+1]:
            duration += stepSize
            i += 1
        RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(filename, start, duration, labels[i]))
        i += 1
        
def clusterThreshold_vad(embeddings, threshold, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    aggloclust=AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=threshold).fit(embeddings)
    if print_labels:
        print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM_vad(aggloclust.labels_, rttmOut, filename, stepSize, offset, vad)
    return aggloclust.labels_


def labels2RTTM_vad_segments(labels, rttmFd, filename, stepSize=1, offset=0, segments=None, frame_counts=None):
    # if vad != None:
        # VAD = open(vad).readlines()
    RTTM = open(rttmFd, 'w')
    i = 0
    while i < len(labels):
        for segment, frame_len in zip(segments, frame_counts):
            j = 0
            start = segment[1] + offset
            while j < frame_len:
                duration = float(stepSize)
                while j < frame_len -1 and labels[i] == labels[i+1]:
                    duration += stepSize
                    i += 1
                    j += 1
                RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(filename, start, duration, labels[i]))
                i += 1
                j += 1
                start = start + duration
                
        
def clusterThreshold_vad_segments(names, threshold, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    segments = []
    embeddings = []
    for segment in vad:
        segment = segment.split()
        segments.append([int(segment[0].split("/")[-1].split('.')[0].split('-')[1]), float(segment[2]), float(segment[3])])
    segments = sorted(segments)
    assert (len(names) == len(segments)), (filename, len(names), len(segments))
    frame_counts = []
    for name in sorted(names):
        embeddings.extend(name[1])
        frame_counts.append(len(name[1]))
        # print(name[0], len(name[1]))
    # import ipdb; ipdb.set_trace() 
    aggloclust=AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=threshold).fit(embeddings)
    if print_labels:
        print(aggloclust.labels_)
    assert (len(aggloclust.labels_) == sum(frame_counts))
    if rttmOut != None:
        labels2RTTM_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, segments, frame_counts)
    return aggloclust.labels_
        
def n_clusters_vad_segments(names, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    segments = []
    embeddings = []
    for segment in vad:
        segment = segment.split()
        segments.append([int(segment[0].split("/")[-1].split('.')[0].split('-')[1]), float(segment[2]), float(segment[3])])
    segments = sorted(segments)
    assert (len(names) == len(segments)), (filename, len(names), len(segments))
    frame_counts = []
    for name in sorted(names):
        embeddings.extend(name[1])
        frame_counts.append(len(name[1]))
        # print(name[0], len(name[1]))
    
    aggloclust=AgglomerativeClustering(n_clusters=N, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    # import ipdb; ipdb.set_trace() 
    if print_labels:
        print(aggloclust.labels_)
    assert (len(aggloclust.labels_) == sum(frame_counts))
    if rttmOut != None:
        labels2RTTM_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, segments, frame_counts)
    return aggloclust.labels_


def labels2RTTM_post_vad_segments(labels, rttmFd, filename, stepSize=1, offset=0, yes_no=None, vad=None):
    vad_segments = []
    if vad != None:
        previous_end = 0.0
        for line in vad:
            # line = line.strip('\n').split()
            # vad_segments.append((float(line[2]), float(line[3])))
            line = line.strip('\n').split()
            begin = float(line[2])
            end = float(line[3])
            # vad_segments.append((begin, end))
            if previous_end < end:
               if previous_end > begin:
                   begin = previous_end
               vad_segments.append((begin, end))
               previous_end = end
            # else:
                # # print(previous_end, name, line)
                # vad_segments.append((previous_end, previous_end))
    # import ipdb; ipdb.set_trace()
    RTTM = open(rttmFd, 'w')
    i = 0
    j = 0
    start = offset
    curr_time = offset
    while i < len(yes_no) and j < len(labels)-1:
        duration = stepSize
        write_ = False
        while yes_no[i] == 1 and j < len(labels)-1 and labels[j] == labels[j+1]:
            # print(len(yes_no), len(labels), i, j, filename)
            duration += stepSize
            i += 1
            j += 1
            write_ = True
            curr_time += stepSize
        if write_:
            rttm_start = start
            rttm_end = duration
            
            # # Shrink frames to vad
            # for seg in vad_segments:
                # if start < seg[0] and start+stepSize > seg[0]:
                    # rttm_start = seg[0]
                # if rttm_start + duration > seg[1] and rttm_start + duration - stepSize < seg[1]:
                    # rttm_end = seg[1] - rttm_start
            
            # # Expand frames to vad
            # for seg in vad_segments:
                # if start > seg[0] and start-stepSize < seg[0]:
                    # rttm_start = seg[0]
                # if start + duration < seg[1] and start + duration + stepSize > seg[1]:
                    # rttm_end = seg[1] - start
                    
                    
            RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(filename, rttm_start, rttm_end, labels[j]))
        if yes_no[i] == 1:
            j += 1
        i += 1
        curr_time += stepSize
        start = start + duration
                
def n_clusters_post_vad_segments(names, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    embeddings = names[0]
    yes_no = names[1]
    aggloclust=AgglomerativeClustering(n_clusters=N, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    # import ipdb; ipdb.set_trace() 
    if print_labels:
        print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM_post_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, yes_no, vad)
    return aggloclust.labels_
        
def n_clusters_plda_post_vad_segments(yes_no, embeddings, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    # aggloclust=AgglomerativeClustering(n_clusters=N, affinity='cosine', memory=memory, connectivity=None, compute_full_tree=True, linkage='complete').fit(embeddings)
    aggloclust=AgglomerativeClustering(n_clusters=N, linkage='ward').fit(embeddings)

    # import ipdb; ipdb.set_trace() 
    if print_labels:
        print(aggloclust.labels_)
    if rttmOut != None:
        labels2RTTM_post_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, yes_no, vad)
    return aggloclust.labels_
        
def n_clusters_plda_vad_segments(frame_counts, embeddings, segments, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None):
    # aggloclust=AgglomerativeClustering(n_clusters=N, affinity='cosine', memory=memory, connectivity=None, compute_full_tree=True, linkage='complete').fit(embeddings)
    # aggloclust=AgglomerativeClustering(n_clusters=N, affinity='cosine', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    aggloclust=AgglomerativeClustering(n_clusters=N, metric='cosine', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    # aggloclust=AgglomerativeClustering(n_clusters=N, affinity='precomputed', memory=memory, connectivity=None, compute_full_tree=True, linkage='single').fit(embeddings)
    # import ipdb; ipdb.set_trace() 
    if print_labels:
        print(aggloclust.labels_)
    assert (len(aggloclust.labels_) == sum(frame_counts))
    if rttmOut != None:
        labels2RTTM_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, segments, frame_counts)
    return aggloclust.labels_    
    
def clusterThreshold_plda_vad_segments(frame_counts, embeddings, segments, threshold, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None, labels=None):
    # aggloclust=AgglomerativeClustering(n_clusters=None, affinity='cosine', memory=memory, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=threshold).fit(embeddings)
    # aggloclust=AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='ward').fit(embeddings)
    aggloclust=AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average', affinity='cosine').fit(embeddings)
    # import ipdb; ipdb.set_trace()
    if labels != None:
        if rttmOut != None:
            labels2RTTM_vad_segments(labels, rttmOut, filename, stepSize, offset, segments, frame_counts)
        return labels
    if print_labels:
        print(aggloclust.labels_)
    assert (len(aggloclust.labels_) == sum(frame_counts))
    if rttmOut != None:
        labels2RTTM_vad_segments(aggloclust.labels_, rttmOut, filename, stepSize, offset, segments, frame_counts)
    return aggloclust.labels_

from scipy.cluster.hierarchy import ward, fcluster, linkage
from scipy.spatial.distance import pdist

def labels2RTTM_vad_segments_4plda(labels, rttmFd, filename, stepSize=1, offset=0, segments=None, frame_counts=None, best_label=None, embeddings=[], plda_train_dir="plda_train"):
    # if vad != None:
        # VAD = open(vad).readlines()
    os.makedirs(plda_train_dir, exist_ok=True)
    RTTM = open(rttmFd, 'w')
    name = rttmFd.split('/')[-1].split('.')[0]
    ivecs_fd = open(os.path.join(plda_train_dir, name+".ivecs"), 'w')
    spk2utt_fd = open(os.path.join(plda_train_dir, name+".spk2utt"), 'w')
    spk2utt_fd.write(name)
    i = 0
    while i < len(labels):
        for segment, frame_len in zip(segments, frame_counts):
            j = 0
            start = segment[1] + offset
            while j < frame_len:
                duration = float(stepSize)
                while j < frame_len -1 and labels[i] == labels[i+1]:
                    if labels[i] == best_label:
                        ivecs_fd.write(name+"_"+str(i)+" " + str(embeddings[i].tolist()).replace('[', '[ ').replace(']', ' ]').replace(',', '') + '\n')
                        spk2utt_fd.write(" "+name+"_"+str(i))
                    duration += stepSize
                    i += 1
                    j += 1
                    # import ipdb; ipdb.set_trace()             
                if labels[i] == best_label:
                    RTTM.write('SPEAKER %s 0   %.3f   %.3f <NA> <NA> %s <NA> <NA>\n'%(filename, start, duration, labels[i]))                    
                i += 1
                j += 1
                start = start + duration
    spk2utt_fd.write('\n')
    ivecs_fd.close()
    spk2utt_fd.close()
    
def n_clusters_vad_segments_get_dendogram(names, N, rttmOut=None, filename=None, stepSize=0.2, offset=0, print_labels=False, memory=None, vad=None, plda_train_dir="plda_train"):
    segments = []
    embeddings = []
    for segment in vad:
        segment = segment.split()
        segments.append([int(segment[0].split("/")[-1].split('.')[0].split('-')[1]), float(segment[2]), float(segment[3])])
    segments = sorted(segments)
    assert (len(names) == len(segments)), (filename, len(names), len(segments))
    frame_counts = []
    for name in sorted(names):
        embeddings.extend(name[1])
        frame_counts.append(len(name[1]))
    # Z = ward(pdist(np.array(embeddings)))
    # Z = linkage(np.array(embeddings), 'average')

    # F = fcluster(Z, t=10, criterion='distance')
    # F2 = fcluster(Z, t=10, criterion='maxclust')
    aggloclust=AgglomerativeClustering(n_clusters=N, affinity='euclidean', memory=memory, connectivity=None, compute_full_tree=True, linkage='average').fit(embeddings)
    
    if print_labels:
        print(aggloclust.labels_)
    assert (len(aggloclust.labels_) == sum(frame_counts))
    l = list(aggloclust.labels_)
    counts = list(map(lambda labels: {labels: l.count(labels)}, set(l)))
    count_dict = {k: v for d in counts for k, v in d.items()}
    # import ipdb; ipdb.set_trace()
    # print(count_dict)
    best_label = max(zip(count_dict.values(), count_dict.keys()))[1]
        
    if rttmOut != None:
        labels2RTTM_vad_segments_4plda(aggloclust.labels_, rttmOut, filename, stepSize, offset, segments, frame_counts, best_label, embeddings, plda_train_dir)
     
    return aggloclust.labels_
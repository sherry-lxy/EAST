import os
import time
import pickle
import pdb
import glob
import itertools

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import random
import numpy as np
import copy

from PIL import Image
import matplotlib.pyplot as plt
import cv2

import collections

import pathlib

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

import json
from statistics import mode

import sys
sys.path.append(os.path.abspath("../.."))
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
# from cirtorch.utils.whiten import whitenlearn, whitenapply, pcawhitenlearn

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

class MyClass: pass
args = MyClass()
# args.network_path='retrievalSfM120k-resnet101-gem'; args.whitening='retrieval-SfM-120k'
args.network_path='gl18-tl-resnet152-gem-w'; args.whitening=None
args.multiscale='[1]'    #'[1, 1/2**(1/2), 1/2]'
args.image_size=1024
args.gpu_id="0"
args.data_root='./../../data'

# ????????????????????????
topk = 5
path_folder = "../../../dataset/wcp/annotation" 

# ??????????????????
json_pass = "../../../dataset/wcp/annotation_json/question_yolo.json"
# json_pass = "/mnt/vmlqnap02/home/li/main/dataset/wcp/annotation_json/yolo_annotation.json"
json_open = open(json_pass, 'r')
# ????????????????????????JSON?????????????????????
reference_question = json.load(json_open)

# ????????????????????????
correct_question_pass = "../../../dataset/wcp/annotation_json/query_correct_question_yolo.json"
# correct_question_pass = "/mnt/vmlqnap02/home/li/main/dataset/wcp/annotation_json/query_yolo_anno.json"
correct_question_open = open(correct_question_pass, 'r')
# ????????????????????????JSON?????????????????????
correct_question = json.load(correct_question_open)

# ????????????
# json_pass = "../../../dataset/wcp/annotation_json/question_yolo.json"
json_pass = "/mnt/vmlqnap02/home/li/main/dataset/wcp/annotation_json/reference_east.json"
json_open = open(json_pass, 'r')
# ????????????????????????JSON?????????????????????
reference_east = json.load(json_open)

# ??????????????????(EAST) 
# correct_question_pass = "../../../dataset/wcp/annotation_json/query_correct_question_yolo.json"
json_pass = "/mnt/vmlqnap02/home/li/main/dataset/wcp/annotation_json/query_east.json"
json_open = open(json_pass, 'r')
# ????????????????????????JSON?????????????????????
query_east = json.load(json_open)

def main():
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    print(">> Loading network:\n>>>> '{}'".format(args.network_path))
    if args.network_path in PRETRAINED:
        # pretrained networks (downloaded automatically)
        state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(args.data_root, 'networks'))
    else:
        # fine-tuned network from path
        state = torch.load(args.network_path)

    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    #net_params['mean'] = [0.486, 0.457, 0.405]
    net_params['std'] = state['meta']['std']
    #net_params['std'] = [0.234, 0.232, 0.224]
    net_params['pretrained'] = False

    # load network
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])
    
    # if whitening is precomputed
    if 'Lw' in state['meta']:
        net.meta['Lw'] = state['meta']['Lw']
    
    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))            
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1
    
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            if len(ms)>1:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:
            # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                if len(ms) > 1:
                    whiten_fn += '_ms'
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                raise ValueError('whitening')

    else:
        Lw = None
    # ------------------------------------------------------------------------------------------------------------------

    # evaluate on files
    start = time.time()

    # specify your query and db images
    reference = sorted(pathlib.Path('../../../dataset/wcp/reference_resize/').glob('*.jpg'))
    query = sorted(pathlib.Path('../../../dataset/wcp/query_resize/').glob('*.jpg'))

    bbxs = None

    # extract database and query vectors
    reference_vecs = extract_vectors(net, reference, args.image_size, transform, ms=ms, msp=msp)
    query_vecs = extract_vectors(net, query, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)

    # convert to numpy
    reference_vecs = reference_vecs.numpy()
    query_vecs = query_vecs.numpy()

    # ?????? 
    reference_vecs = reference_vecs.T
    query_vecs = query_vecs.T
    number_of_reference = reference_vecs.shape[0]
    number_of_query = query_vecs.shape[0]
    
    # ????????????
    
    # ---------------------------------- 1 direction ----------------------------------
    print("--------------------- 1 direction ---------------------")

    # map = np.array([])
    # top1 = np.array([])
    # accuracy = np.zeros(number_of_query)
    accuracy = np.zeros(100)
    count_question = np.zeros(number_of_query)
    direction = 4
    top5_fail = 0

    start = time.time()
    for i in range(100):
        dis_list = {}
        print("\nquery: ", query[i].name)

        # ??????1???????????????????????????????????????top-5????????????
        for j in range(0, number_of_reference, direction):
            # np.dot(reference_vecs[i+d].T, reference_vecs[j]
            dis = np.array([np.dot(query_vecs[i].T, reference_vecs[j+x]) for x in range(direction)])  
            distance = np.max(dis)
            idx = np.argmax(dis)

            ref = reference[j+idx].name[:-10]

            dis_list[ref] = distance       
        dis_sort = np.array(sorted(dis_list.items(), key=lambda x:x[1], reverse=True))
        # ???????????????
        ans_img = dis_sort[:topk, 0]
        print(" ans_img: ", ans_img)
        # ?????????
        img_sim = np.array(dis_sort[:topk, 1], dtype='float64')
        print(" sim: ", img_sim)
        new_img = np.copy(ans_img)
        # -------------------------------------------------------------------------------------------------------------------------------

        # ???????????????
        question, top5_info, question_east = Question(reference_east, reference_question, new_img, img_sim)
        # ???????????????????????????
        ans, accuracy[i] = QAndA(query[i].name[:-10], query_east, correct_question, question, top5_info, img_sim, question_east)

        print(" acc = ", accuracy[i])
        # print(" question count = ", count_question[i])
    
    print("**************************************************")
    print("acc. = ", accuracy)     
    print("Accuracy = ", np.mean(accuracy))
    # print("Max number of question = ", np.max(count_question))
    # print("One or two question = ", np.sum(count_question<=2))
    # print("Average number of question = ", np.mean(count_question))
    print("Top5 fail = ", top5_fail)

def Question(reference_east, reference_question, new_img, img_sim):
    ques_east, ques_yolo = np.array([]), np.array([]) # ?????????????????????????????????
    top5_info = {im:{} for im in new_img} # ?????????

    for k in range(topk):
        top5_info[new_img[k]]['similarity'] = img_sim[k]

        # EAST??????????????????
        tmp_east = copy.copy(reference_east[new_img[k]])
        ques_east = np.append(ques_east, tmp_east)
    

        # YOLO???????????????
        tmp_yolo = copy.copy(reference_question[new_img[k]])
        ques_yolo = np.append(ques_yolo, tmp_yolo)

        top5_info[new_img[k]]['question'] = tmp_east + tmp_yolo 

    question_east = list(np.unique(ques_east))
    question_yolo = list(np.unique(ques_yolo))
    question = question_east + question_yolo
    print("Question: ", question)

    return question, top5_info, question_east

def QAndA(query, query_east, correct_question, question, top5_info, img_sim, question_east):
    cq = query_east[query] + correct_question[query]

    # Top5?????????????????????????????????????????????
    if len(question) == 0:
        print("No Question")
        top5_fail += 1
        accuracy = Accuracy(path_folder, query, list(top5_info))
    
    else:            
        # ?????????????????????
        # print("Question: ", question)
        # print(question)

        # ???????????????????????????
        entropy =  Entropy(question, img_sim, top5_info)
        # ????????????????????????????????????????????????????????????
        ask_no = np.argsort(entropy)[0]
        ask = question[ask_no]

        print("---------------- ???Question??? ----------------")
        while True:
            # count_question[i] += 1
            print("???", ask, ": ", end='')
            # yes/no??????
            user_ans = True in [cq[x] in ask for x in range(len(cq))]
            print("\t{}".format(user_ans))
            # print(question)

            # ----------------------------- YES -----------------------------
            if user_ans:
                # ?????????????????????top5?????????
                before_len = len(top5_info)
                for im in list(top5_info):
                    # print("???: ", top5_info)
                    # ????????????????????????
                    if question[ask_no] in top5_info[im]['question']:
                        top5_info[im]['question'].remove(question[ask_no])
                    # Yes???????????????top??????????????????
                    else:
                        top5_info.pop(im)
                    
                    
                after_len = len(top5_info)
                # print("top5: ", top5_info)

                # ?????????1??????????????????
                if len(top5_info) <= 1:
                    break
                    
                # ?????????????????????????????????????????????
                if before_len == after_len:
                    break
            
                # ??????????????????????????????
                question = []
                for v in top5_info.values():
                    question.append(v["question"])
                
                question = list(set(list(itertools.chain.from_iterable(question))))
                # print("question: ", question)

                # ???????????????????????????????????????
                if len(question) == 0:
                    break
                
                entropy = Entropy(question, img_sim, top5_info)
                # ?????????????????????????????????
                # print("entropy before: ", entropy)
                east_check = [0.5 if x in question_east else 1 for x in question]
                # print("east check: ", east_check)
                entropy *= east_check
                # print("entropy after: ", entropy)
                
                # ????????????????????????????????????????????????????????????
                ask_no = np.argsort(entropy)[0]
                ask = question[ask_no]
            
            # ----------------------------- NO -----------------------------
            else:
                # ???????????????????????????????????????
                for im in list(top5_info):
                    if question[ask_no] in top5_info[im]['question']:
                        top5_info.pop(im)
                # print("Question: ", top5_info)
                # print("top5: ", top5_info)
                # ??????????????????????????????
                question = []
                for v in top5_info.values():
                    question.append(v["question"])
                question = list(set(list(itertools.chain.from_iterable(question))))
                # print("question: ", question)

                # ???????????????top5?????????????????????????????????
                if len(top5_info) == 0:
                    break

                # ???????????????????????????
                if len(question) == 0:
                    break

                # ???????????????????????????
                entropy = Entropy(question, img_sim, top5_info)
                # ?????????????????????????????????
                east_check = [0.5 if x in question_east else 1 for x in question]
                entropy *= east_check
                
                # ????????????????????????????????????????????????????????????
                ask_no = np.argsort(entropy)[0]
                ask = question[ask_no]

            # ???????????????????????????
            if len(question)==0:
                break
            
        print("--------------------------------------------")      
        ans = list(top5_info)
        print(" ans: ", ans)

        # One-Meter-Level Accuracy????????????
        # accuracy[i] = OneMeterLevelAccuracy(annotation, query[i].name[:-6], ans)
        accuracy = Accuracy(path_folder, query, ans)
    
    return ans, accuracy

# ???????????????????????????
def Entropy(question, img_sim, top5_info):
    num_of_ques = len(question)
    entropy = np.zeros(num_of_ques)
    # ????????????
    # prior = [0 if ans_img[x]=='0' else img_sim[x]/np.sum(img_sim) for x in range(topk)]

    # softmax
    prior = np.array([v['similarity'] for v in top5_info.values()])
    prior = np.exp(prior) / np.sum(np.exp(prior))
    num_img = len(prior)
    top = len(top5_info)

    # print("top: ", top)

    check = np.array([[q in v['question'] for v in top5_info.values()] for q in question])

    for q in range(num_of_ques):
        # ???????????????????????????  
        yes = [prior[x] if check[q][x] else 0 for x in range(top)]
        pa_t = [prior[x]*np.log2(prior[x]/np.sum(yes)) for x in range(top)]
        # pa_t = [prior[x] * (check[q][x]/np.sum(check[q])) if np.sum(check[q])!=0 else 0 for x in range(top)]
        # pa_t = [1 if pa_t[x]==0 else pa_t[x] for x in range(top)]
        # pa_t = pa_t*np.log2(pa_t)

        no = [prior[x] if not check[q][x] else 0 for x in range(top)]
        pa_f = [prior[x]*np.log2(prior[x]/np.sum(no)) if not np.sum(no)==0 else 0 for x in range(top)]
        # pa_f = [prior[x] * (check[q][x]/(top - np.sum(check[q]))) if np.sum(check[q])!=top else 0 for x in range(top)]
        # pa_f = [1 if pa_f[x]==0 else pa_f[x] for x in range(top)]
        # pa_f = pa_f * np.log2(pa_f)

        entropy[q] = -np.sum(pa_t)-np.sum(pa_f)
    # print("H = ", entropy)
    return entropy

# L2??????
def L2Distance(x1, x2):
    return np.sqrt(np.power((x1[0]-x2[0]),2) + np.power((x1[1]-x2[1]),2))

# top1??????????????????????????????1[m]?????????????????????????????????????????????
def OneMeterLevelAccuracy(annotation, query, ans):
    correct = annotation[query]
    judge = L2Distance(correct, ans)
    print(" dis=", judge)

    return judge <= 150

# top1?????????
def Accuracy(path_folder, files_query, ans_img):
    if len(ans_img) == 0:
        return False
    else:
        path = "{0}/{1}.txt".format(path_folder, files_query)
        with open(path) as f:
            correct_list = np.array([s.strip() for s in f.readlines()])
        
        return ans_img[0] in correct_list

    
if __name__ == '__main__':
    main()

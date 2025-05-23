"""Evaluation"""

from __future__ import print_function
import os
import sys
import time

import torch
import numpy as np

from data import get_test_loader
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]="3"
#+++++++++++++
import copy
import pdb
import tqdm
import argparse
def logging_func(logfile, message):
    print(message)
    with open(logfile, "a") as f:
        f.write(message + "\n")
    f.close()
#+++++++++++++
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)
def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    all_adjs = None
    all_amr_embs = None

    max_n_word = 0
    max_n_region = 0
    for i, (images, captions, lengths,  ids, adjs) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths,  ids, adjs) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            all_adjs = np.zeros((len(data_loader.dataset), adjs.size(1), 36))
            #all_depends = [1] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
        all_adjs[ids] = adjs.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
            #all_depends[nid] = depends[j]

        del images, captions, adjs
    return img_embs, cap_embs, cap_lens, all_adjs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = SGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))

    print('Computing results...')
    img_embs, cap_embs, cap_lens, adjs = encode_data(model, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        adjs = np.array([adjs[i] for i in range(0, len(adjs), 5)])

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, adjs, opt, shard_size=100)
        np.save("./sims.npy", sims)
        end = time.time()
        print("calculate similarity time:", end-start)

        # bi-directional retrieval
        r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            start = time.time()
            sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard,adjs, opt, shard_size=100)
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

def evalrank_ensemble(model_path1, model_path2, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full glo_data is
    used for evaluation.
    Please use split="testall", fold5=False for MSCOCO 5K and split="testall", fold5=True for MSCOCO1K
    Do not use split="test", fold5=False for MSCOCO1K which only includes 1000 images & 5000 captions and which produces higher but unfair results.
    """
    # load model and options
    checkpoint = torch.load(model_path1)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']
    save_epoch2 = checkpoint2['epoch']
    print(opt2)
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab('/home/shenxiang/sda/zhuliqi/NAAF/vocab/%s_vocab.json' % opt.data_name)
    opt.vocab_size = len(vocab)

    # construct model
    model = SGRAF(opt)
    model2 = SGRAF(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))
    print("=> loaded checkpoint_epoch {}".format(save_epoch2))

    print('Computing results...')
    img_embs, cap_embs, cap_lens, adjs  = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' % (img_embs.shape[0] / 5, cap_embs.shape[0]))

    img_embs2, cap_embs2, cap_lens2,adjs2 = encode_data(model2, data_loader)
    print('Images: %d, Captions: %d' % (img_embs2.shape[0] / 5, cap_embs2.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        img_embs2 = np.array([img_embs2[i] for i in range(0, len(img_embs2), 5)])
        adjs = np.array([adjs[i] for i in range(0, len(adjs), 5)])
        adjs2 = np.array([adjs2[i] for i in range(0, len(adjs2), 5)])


        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, adjs, opt, shard_size=100)
        sims2 = shard_attn_scores(model2, img_embs2, cap_embs2, cap_lens2, adjs2, opt, shard_size=100)
        end = time.time()
        print("calculate similarity time:", end - start)
        sims = (sims + sims2) / 2

        # bi-directional retrieval    其中rt和rti为ranks
        r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        ### 处理ranks
        rt = np.array(rt)
        rti = np.array(rti)
        print("**************** rt(i2t) 为 *****************\n")
        print(rt, rt.shape)
        print("**************** rti(t2i) 为 *****************\n")
        print(rti, rti.shape)

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            img_embs_shard2 = img_embs2[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard2 = cap_embs2[i * 5000:(i + 1) * 5000]
            cap_lens_shard2 = cap_lens2[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard,adjs, opt,
                                     shard_size=100)
            sims2 = shard_attn_scores(model2, img_embs_shard2, cap_embs_shard2, cap_lens_shard2, adjs2,opt,
                                      shard_size=100)
            end = time.time()
            print("calculate similarity time:", end - start)
            sims = (sims + sims2) / 2
            r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
def shard_attn_scores(model, img_embs, cap_embs, cap_lens, adjs,  opt, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                adj = torch.from_numpy(adjs[im_start:im_end]).float().cuda()
                #dep = depends[ca_start:ca_end]

                sim = model.forward_sim(im, ca, l, adj )

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == '__main__':

    """evalrank("/home/shenxiang/sda/Mengfanrong/SGRAF-G/runG_coco/runX/checkpoint/checkpoint_46.pth.tar",
             data_path="/home/shenxiang/sda/zhuliqi/NAAF/data", split="testall", fold5=True)"""
    evalrank_ensemble("/home/shenxiang/sda/Mengfanrong/SGRAF/runG_coco/runX/checkpoint/model_best.pth.tar",
                              "/home/shenxiang/sda/Mengfanrong/SGRAF-A/runA_coco/runX/checkpoint/model_best.pth.tar",
                     data_path='/home/shenxiang/sda/Mengfanrong/NAAF/data/', split="test", fold5=True)



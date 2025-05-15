"""SGRAF model"""

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
def l2norm_glo(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X, norm
def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)
class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, lengths, batch_first=True)

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb
class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global
class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global
class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Correpondence_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Correpondence_regulator, self).__init__()

        self.rcr_smooth_w = nn.Sequential(nn.Linear(sim_dim, sim_dim // 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim // 2, 1)).cuda()
        self.rcr_matrix_w = nn.Sequential(nn.Linear(sim_dim, sim_dim * 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim * 2, embed_dim)).cuda()
        self.tanh = nn.Tanh().cuda()
        self.relu = nn.ReLU().cuda()
        self.init_weights()
    def forward(self, x, matrix, smooth):

        matrix = (self.tanh(self.rcr_matrix_w(x)) + matrix).clamp(min=-1, max=1)
        smooth = self.relu(self.rcr_smooth_w(x) + smooth)

        return matrix, smooth

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def collect_emb(img_emb, adj_mtx, weights_raw, n_partial):
    batch_size = img_emb.size(0)
    n_region = img_emb.size(1)
    n_feat = img_emb.size(2)


    # 确保 adj_mtx 在 CUDA 上
    adj_mtx = adj_mtx.cuda()
    idx_background = torch.arange(0, n_region).unsqueeze(1).repeat(1, n_partial) \
        .unsqueeze(0).repeat(batch_size, 1, 1).cuda()

    effe_rels = torch.topk(adj_mtx, n_partial, -1, largest=True)[0]
    topk_idx = torch.topk(adj_mtx, n_partial, -1, largest=True)[1]
    effe_ids = torch.where(effe_rels == 1, topk_idx, idx_background)
    # effe_ids_weight shape: (n_image,36,12,1)
    effe_ids_weight = effe_ids.unsqueeze(3)
    # effe_ids_feat shape: (n_image,36,12,n_feat)
    effe_ids_feat = effe_ids_weight.repeat(1, 1, 1, n_feat)

    """ gather raw image embeddings
      """
    img_emb_rep = img_emb.unsqueeze(1).repeat(1, n_region, 1, 1)
    img_emb_gather = torch.gather(img_emb_rep, dim=2, index=effe_ids_feat)

    """ gather weights
      """
    weights_raw = weights_raw.unsqueeze(1).repeat(1, n_region, 1, 1)
    weights_gather = torch.gather(weights_raw, dim=2, index=effe_ids_weight)

    return img_emb_gather, weights_gather
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim, module_name='SAF', sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)
        self.emb_local_li = nn.Linear(embed_size, embed_size)
        self.emb_local_bn = nn.BatchNorm1d(36)
        self.emb_global_li = nn.Linear(embed_size, embed_size)
        self.emb_global_bn = nn.BatchNorm1d(embed_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.4)
        self.emb_common = nn.Linear(embed_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.num_partial = 5
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sim_transform_w = nn.Linear(embed_size, sim_dim)
        self.sigmoid = nn.Sigmoid()
        rar_step, rcr_step, alv_step = 0, 3, 3
        self.rcr_modules = nn.ModuleList([Correpondence_regulator(sim_dim, embed_size) for j in range(rcr_step)]
        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])

        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens,adjs):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)
        l_emb = self.dropout(self.tanh(self.emb_local_bn(self.emb_local_li(img_emb))))
        g_emb = self.dropout(self.tanh(self.emb_global_bn(self.emb_global_li(img_ave))))
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights_raw = self.emb_common(common)
        weights = self.softmax(weights_raw.squeeze(2)).unsqueeze(2)
        new_global = (weights * img_emb).sum(dim=1)
        img_glo, norm_glo = l2norm_glo(new_global, dim=-1)
        # remember to change cuda device
        img_gather_emb, weights_gather = collect_emb(img_emb, adjs, weights_raw, self.num_partial)
        weights_gather = self.softmax(weights_gather.squeeze(3)).unsqueeze(3)
        # compute partial embedding (n_images, 36, 1024)
        new_global = (weights_gather * img_gather_emb).sum(dim=2)
        norm_glo = norm_glo.unsqueeze(1).repeat(1, 36, 1)
        img_con = torch.div(new_global, norm_glo)
        new_img_emb = l2norm(nn.LeakyReLU(0.1)(img_emb.mul(img_con)), dim=-1)
        img_emb_1 = torch.cat([img_emb, new_img_emb], 1)
        img_emb_2, _ = self.img_rnn(img_emb_1)
        img_emb_3 = torch.cat([img_emb, img_emb], 1)
        img_emb = l2norm(img_emb_2 + img_emb_3, dim=-1)
        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)
            # local-global alignment construction
            matrix = torch.ones(1024).cuda()
            smooth = 10.0
            for m, rcr_module in enumerate(self.rcr_modules):
                query = torch.mul(cap_i_expand, matrix)
                Context_img = SCAN_attention(query, img_emb, smooth)
                sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
                sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
                matrix, smooth = rcr_module(sim_loc, matrix, smooth)
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)
            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)
            # compute the final similarity vector
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
                sim_i = self.sigmoid(self.sim_eval_w(sim_vec))

            else:
                sim_vec = self.SAF_module(sim_emb)
                sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            # compute the final similarity score
            sim_all.append(sim_i)
        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)
        return sim_all
    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SELF_attention(nn.Module):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """

    def __init__(self, embed_size):
        super(SELF_attention, self).__init__()
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

    def forward(self, query, context, smooth, eps=1e-8):
        sim_k = self.fck(context)
        sim_q = self.fcq(query)
        sim_v = context
        attn = torch.bmm(sim_k, sim_q.permute(0, 2, 1))

        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = F.softmax(attn.permute(0, 2, 1) * smooth, dim=2)

        # --> (batch, queryL, d)
        weightedContext = torch.bmm(attn, sim_v)
        weightedContext = l2norm(weightedContext, dim=-1)

        return weightedContext

def SCAN_attention(query, context,smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()
class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths,amr_targets,amr_lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        amr_embs=self.txt_enc(amr_targets,amr_lengths)
        cap_embs= torch.cat((cap_embs, amr_embs), dim=1)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens,adjs):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens,adjs)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids, adjs,amr_targets,amr_lengths,*args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.4)

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths,amr_targets,amr_lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens,adjs)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
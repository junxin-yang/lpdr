from collections import defaultdict

import torch
import numpy as np
from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01


def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape   # (T, C)，C为类别数

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size'] # 束宽度，每步保留 Top-K 个前缀
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))   # 概率阈值，低于该值的字符被剪枝，减少计算量

    length, class_count = emission_log_prob.shape   # 每个时间步的类别数

    beams = [(tuple(), (0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob)) lp_b = 0对应 log(1.0)（空前缀以 blank结尾的概率为 1），lp_nb = NINF（负无穷）对应 log(0.0)（空前缀不可能以非 blank结尾），前面这个 tuple() 表示空前缀
    # initial of beams: (empty_str, (log(1.0), log(0.0)))

    for t in range(length): # 对每个时间步分别进行处理
        new_beams_dict = defaultdict(lambda: (NINF, NINF))  # log(0.0) = NINF，# 新束字典，默认概率为 -inf

        for prefix, (lp_b, lp_nb) in beams: # 对每个时间步的每个前缀分别进行处理
            for c in range(class_count):    # 扩展前缀：处理每个字符 c，对当前束中每个前缀 prefix和其概率 (lp_b, lp_nb)，遍历所有类别 c（跳过概率低于阈值的字符）
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None  # prefix[-1]: 前缀的结尾字符

                # if new_prefix == prefix   不扩展前缀，只更新概率
                new_lp_b, new_lp_nb = new_beams_dict[prefix]    # 从 new_beams_dict 字典中获取当前前缀以 blank 结尾和以非 blank 结尾的累计对数概率

                if c == blank:  # 当前类别是空白符（blank）
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),   # new_lp_b：之前已累加的 blank 结尾概率（历史路径），lp_b + log_prob：前缀原本以 blank 结尾，这一步继续选择 blank，lp_nb + log_prob：前缀原本以非 blank 结尾，这一步转为 blank，用 logsumexp 把这三种路径的概率安全地累加（在对数空间下），得到所有以 blank 结尾的路径的总概率
                        new_lp_nb
                    )   # 统计所有能到达当前前缀并以 blank 结尾的路径概率
                    continue
                if c == end_t:  # 当前类别和前缀结尾字符相同，前缀不变，只更新非 blank 结尾的概率（CTC 规则，连续字符只能通过 blank 分隔，否则会被合并），计算那些扩展字符 c后，规整化前缀仍然保持为 "a"​ 的路径的概率
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                # if new_prefix == prefix + (c,)
                new_prefix = prefix + (c,)  # 扩展前缀为新序列（只是当前的前缀下得到的一条新的路径）
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:  # 当前类别和前缀结尾字符不同
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:   # c == end_t 计算那些扩展字符 c后，会形成一个全新的规整化前缀 "aa"​ 的路径的概率
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        # sorted by log(blank_prob + non_blank_prob)
        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True) # x[1] 是一个二元组 (blank结尾概率, 非blank结尾概率)，logsumexp(x[1]) 就是把这两种概率累加起来，得到该前缀的总概率（在对数空间下）
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list

from __future__ import print_function, division

import copy
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def gen_target(LOSS_MODE, tg, device):
    target, target2 = None, None
    if LOSS_MODE == "BCELoss":
        target = torch.FloatTensor(tg)
    return target.to(device), target2


def cal_loss(LOSS_MODE, scores, target, target2, criterion, criterion2):
    loss = None
    if LOSS_MODE == 'BCELoss':
        loss = criterion(scores[0], target.unsqueeze(1))
    return loss


def trans_scores(LOSS_MODE, scores, cos_dis):
    if LOSS_MODE == 'BCELoss':
        scores_n = scores[0]
    return scores_n


def collect_metrics(metrics_records, loss_records, step_name, metrics, label_predict, total_loss):
    avg_loss = np.mean(total_loss)
    loss_records[step_name].append(avg_loss)
    accuracy, recall, F1 = cal_f1(metrics)
    metrics_records[step_name]["acc"].append(accuracy)
    metrics_records[step_name]["rec"].append(recall)
    metrics_records[step_name]["f1"].append(F1)
    metrics_records[step_name]["auc"].append(
        roc_auc_score(y_true=label_predict["label"], y_score=label_predict["predict"]))
    print("{} loss:{:.4f} acc:{:.4f} rec:{:.4f} f1:{:.4f}".format(step_name, avg_loss, accuracy, recall, F1))
    return metrics_records, loss_records, avg_loss


def hit_rate(rank_list, topk):
    hit_list = []
    for rank in rank_list:
        rank = rank + 1
        if rank > topk:
            hit = 0
        else:
            hit = 1.0 / topk * (topk + 1 - rank)
        hit_list.append(hit)
    return hit_list


def cal_metrics_batch(metrics, scores, tg, loss_mode):
    val, idxx = scores.to("cpu").detach().topk(1)
    if loss_mode in ['BCELoss']:
        pre = (val.numpy() >= 0.5) + 0
    for i in range(len(tg)):
        if pre[i] == 1 and tg[i] == 1:
            metrics['TP'] += 1
        elif pre[i] == 1 and tg[i] == 0:
            metrics['FP'] += 1
        elif pre[i] == 0 and tg[i] == 1:
            metrics['FN'] += 1
        elif pre[i] == 0 and tg[i] == 0:
            metrics['TN'] += 1
    return metrics, val.numpy().tolist()


def cal_f1(metrics):
    if metrics["TP"] + metrics["FP"] == 0:
        accuracy = 0
    else:
        accuracy = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    if metrics['TP'] + metrics['FN'] == 0:
        recall = 0
    else:
        recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    if accuracy == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * accuracy * recall / (accuracy + recall)
    return accuracy, recall, F1


def collect_loss(loss):
    tmp = np.mean(loss.to("cpu").detach().numpy(), dtype=np.float64)
    return tmp


def gen_batch_similarity(samples, device=None, poi=False):
    top_lens, down_lens, target = [], [], []
    for sample in samples:
        top_lens.append(len(sample[0][0]))
        down_lens.append(len(sample[1][0]))
        target.append(sample[2])

    top_maxlen = max(top_lens)
    down_maxlen = max(down_lens)
    loc_top, loc_down = [], []
    tim_top, tim_down = [], []
    if poi:
        poi_top, poi_down = [], []
    for sample in samples:
        if poi:
            loc_tmp, tim_tmp, poi_tmp = copy.deepcopy(sample[0])
        else:
            loc_tmp, tim_tmp = copy.deepcopy(sample[0])
        loc_tmp.extend([0] * (top_maxlen - len(loc_tmp)))
        loc_top.append(loc_tmp)
        tim_tmp.extend([0] * (top_maxlen - len(tim_tmp)))
        tim_top.append(tim_tmp)
        if poi:
            for _ in range((top_maxlen - len(poi_tmp))):
                poi_tmp.append([0] * 21)
            poi_top.append(poi_tmp)

        if poi:
            loc_tmp, tim_tmp, poi_tmp = copy.deepcopy(sample[1])
        else:
            loc_tmp, tim_tmp = copy.deepcopy(sample[1])
        loc_tmp.extend([0] * (down_maxlen - len(loc_tmp)))
        loc_down.append(loc_tmp)
        tim_tmp.extend([0] * (down_maxlen - len(tim_tmp)))
        tim_down.append(tim_tmp)
        if poi:
            for _ in range((down_maxlen - len(poi_tmp))):
                poi_tmp.append([0] * 21)
            poi_down.append(poi_tmp)
    loc_top = torch.LongTensor(loc_top).to(device)
    loc_down = torch.LongTensor(loc_down).to(device)
    tim_top = torch.LongTensor(tim_top).to(device)
    tim_down = torch.LongTensor(tim_down).to(device)
    top_lens = torch.LongTensor(top_lens).to(device)
    down_lens = torch.LongTensor(down_lens).to(device)
    if poi:
        poi_top = torch.FloatTensor(poi_top).to(device)
        poi_down = torch.FloatTensor(poi_down).to(device)
        return loc_top, tim_top, poi_top, top_lens, loc_down, tim_down, poi_down, down_lens, target
    else:
        return loc_top, tim_top, top_lens, loc_down, tim_down, down_lens, target

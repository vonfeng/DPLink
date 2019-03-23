from __future__ import print_function, division

import os
import torch.optim as optim

from tqdm import tqdm
from json import encoder

from utils import *
from models import *
from preprocessing import *

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
# GPU config
LR_LOWER_BOUND = 0.9 * 1e-6


def run_siamese(SN, args, data_input, LR, train_mode='cross',
                reproduction=False, SAVE_PATH=None, device=None, USE_POI=False, test_pretrain=False):
    BATCH_SIZE = args.batch_size
    criterion, criterion2, cos_dis = None, None, None

    if args.loss_mode == 'BCELoss':
        criterion = nn.BCELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, SN.parameters()),
                           lr=LR, weight_decay=args.l2, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_step, factor=args.lr_decay,
                                                     threshold=1e-3)

    # training and inferring
    loss_records = {"training": [], "validation": [], "testing": []}
    metrics_records = {"training": {"acc": [], "rec": [], "f1": [], "auc": []},
                       "validation": {"rank": [], "hit": [], "list": []},
                       "testing": {"rank": [], "hit": [], "list": []}}
    if SAVE_PATH is None:
        SAVE_PATH = args.save_path + "/"
    tmp_path = 'checkpoint/'
    os.mkdir(SAVE_PATH + tmp_path)
    for e in tqdm(range(args.epoch), desc="EPOCH"):
        training_len = len(data_input["train"])
        testing_len = len(data_input["test"])
        validing_len = len(data_input["valid"])

        if reproduction is False:
            # training
            ###############
            if train_mode == 'self':
                training_len = min(3000 * 32, training_len)  # only select partial data for faster training
                if e == 0:
                    print("\ntraining instances:{}".format(training_len))
            ###############
            total_loss = []
            SN.train(True)
            metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            label_predict = {"label": [], "predict": []}
            np.random.shuffle(data_input["train"])
            for i in range(0, training_len, BATCH_SIZE):
                if i + BATCH_SIZE > training_len:
                    continue
                samples = copy.deepcopy(data_input["train"][i:i + BATCH_SIZE])
                batch_data = gen_batch_similarity(samples, device=device, poi=USE_POI)

                for _ in range(2):
                    optimizer.zero_grad()
                    if USE_POI:
                        loc_top, tim_top, poi_top, top_lens, loc_down, tim_down, poi_down, down_lens, tg = batch_data
                        scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens, poi_top, poi_down)
                    else:
                        loc_top, tim_top, top_lens, loc_down, tim_down, down_lens, tg = batch_data
                        scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens)
                    loc_top, tim_top, top_lens, loc_down, tim_down, down_lens = loc_down, tim_down, down_lens, loc_top, tim_top, top_lens
                    if USE_POI:
                        poi_top, poi_down = poi_down, poi_top

                    target, target2 = gen_target(args.loss_mode, tg, device=device)
                    loss = cal_loss(args.loss_mode, scores, target, target2, criterion, criterion2)

                    loss.backward()
                    optimizer.step()

                    scores = trans_scores(args.loss_mode, scores, cos_dis)
                    metrics, pre = cal_metrics_batch(metrics=metrics, scores=scores, tg=tg, loss_mode=args.loss_mode)
                    total_loss.append(collect_loss(loss))
                    label_predict["predict"].extend(pre)
                    label_predict["label"].extend(tg)

            metrics_records, loss_records, avg_loss = collect_metrics(metrics_records, loss_records,
                                                                      "training", metrics, label_predict, total_loss)
            torch.save(SN.state_dict(), SAVE_PATH + tmp_path + str(e) + '.m')

            # validation
            total_loss = []
            rank_list = []
            SN.train(False)
            metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            for i in range(0, validing_len):
                pre_list, tg_list = [], []
                for j in range(int(args.neg / BATCH_SIZE)):
                    samples = copy.deepcopy(data_input["valid"][i][j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
                    batch_data = gen_batch_similarity(samples, device=device, poi=USE_POI)
                    if USE_POI:
                        loc_top, tim_top, poi_top, top_lens, loc_down, tim_down, poi_down, down_lens, tg = batch_data
                        scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens, poi_top, poi_down)
                    else:
                        loc_top, tim_top, top_lens, loc_down, tim_down, down_lens, tg = batch_data
                        scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens)

                    target, target2 = gen_target(args.loss_mode, tg, device=device)
                    loss = cal_loss(args.loss_mode, scores, target, target2, criterion, criterion2)
                    scores = trans_scores(args.loss_mode, scores, cos_dis)
                    metrics, pre = cal_metrics_batch(metrics=metrics, scores=scores, tg=tg, loss_mode=args.loss_mode)
                    total_loss.append(collect_loss(loss))
                    pre_list.extend(pre)
                    tg_list.extend(tg)
                rank_batch = [x[1] for x in sorted(zip(pre_list, tg_list), key=lambda xx: xx[0], reverse=True)].index(1)
                rank_list.append(rank_batch)
            hit_list = hit_rate(rank_list, args.topk)
            metrics_records["validation"]["rank"].append(np.mean(rank_list))
            metrics_records["validation"]["hit"].append(np.mean(hit_list))
            metrics_records["validation"]["list"].append(rank_list)
            print("Validation loss:{:.4f} avg-rank:{:.4f}/{:d} avg-hit:{:.4f}/{:d}".format(
                np.mean(total_loss), np.mean(rank_list), args.neg, np.mean(hit_list), args.topk))
            avg_loss = np.mean(total_loss)
            loss_records["validation"].append(avg_loss)

            scheduler.step(avg_loss)
            lr_last = LR
            LR = optimizer.param_groups[0]['lr']
            if lr_last > LR:
                load_epoch = np.argmax(metrics_records["validation"]["hit"])
                SN.load_state_dict(torch.load(SAVE_PATH + tmp_path + str(load_epoch) + '.m'))
                print('load epoch={} model state'.format(load_epoch))
                # continue
            if LR <= LR_LOWER_BOUND:
                break
            if reproduction:
                break

    mid = np.argmax(metrics_records["validation"]["hit"])
    SN.load_state_dict(torch.load(SAVE_PATH + tmp_path + str(mid) + '.m'))

    # testing
    SN.train(False)
    if train_mode in ['cross']:
        total_loss = []
        rank_list = []
        metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        testing_len = len(data_input["test"])
        for i in range(0, testing_len):
            pre_list, tg_list = [], []
            for j in range(int(args.neg / BATCH_SIZE)):
                samples = copy.deepcopy(data_input["test"][i][j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
                batch_data = gen_batch_similarity(samples, device=device, poi=USE_POI)

                if USE_POI:
                    loc_top, tim_top, poi_top, top_lens, loc_down, tim_down, poi_down, down_lens, tg = batch_data
                    scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens, poi_top, poi_down)
                else:
                    loc_top, tim_top, top_lens, loc_down, tim_down, down_lens, tg = batch_data
                    scores = SN(loc_top, tim_top, loc_down, tim_down, top_lens, down_lens)

                target, target2 = gen_target(args.loss_mode, tg, device=device)
                loss = cal_loss(args.loss_mode, scores, target, target2, criterion, criterion2)
                scores = trans_scores(args.loss_mode, scores, cos_dis)
                metrics, pre = cal_metrics_batch(metrics=metrics, scores=scores, tg=tg, loss_mode=args.loss_mode)
                total_loss.append(collect_loss(loss))
                pre_list.extend(pre)
                tg_list.extend(tg)
            rank_batch = [x[1] for x in sorted(zip(pre_list, tg_list), key=lambda xx: xx[0], reverse=True)].index(1)
            rank_list.append(rank_batch)
        loss_records["testing"].append(np.mean(total_loss))
        hit_list = hit_rate(rank_list, args.topk)
        metrics_records["testing"]["rank"].append(np.mean(rank_list))
        metrics_records["testing"]["hit"].append(np.mean(hit_list))
        metrics_records["testing"]["list"].append(rank_list)
        print("Testing loss:{:.4f} avg-rank:{:.4f}/{:d} avg-hit:{:.4f}/{:d}".format(
            np.mean(total_loss), np.mean(rank_list), args.neg, np.mean(hit_list), args.topk))

    mid = 0
    if reproduction is False:

        if test_pretrain:
            for i in range(len(metrics_records["validation"]["hit"])):
                SN.load_state_dict(torch.load(SAVE_PATH + tmp_path + str(i) + '.m'))
                torch.save(SN.state_dict(), SAVE_PATH + 'SN-pre-' + str(i) + '.m')
        mid = np.argmax(metrics_records["validation"]["hit"])
        SN.load_state_dict(torch.load(SAVE_PATH + tmp_path + str(mid) + '.m'))

        save_name = '-'.join([args.data_name, args.loss_mode, args.rnn_mod, args.attn_mod, str(args.layers)])
        if train_mode == 'self':
            save_name = 'SN-pre'
        json.dump({'loss': loss_records, 'metrics': metrics_records},
                  fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
        # for web view
        metrics_records_view = {}
        for key1 in metrics_records:
            metrics_records_view[key1] = {}
            for key2 in metrics_records[key1]:
                if key2 == 'list':
                    continue
                metrics_records_view[key1][key2] = metrics_records[key1][key2]

        json.dump({'loss': loss_records, 'metrics': metrics_records_view},
                  fp=open(SAVE_PATH + save_name + '-view.txt', 'w'), indent=4)
        torch.save(SN.state_dict(), SAVE_PATH + save_name + '.m')

        for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(SAVE_PATH + tmp_path)

    rank_re, hit_re = None, None
    if train_mode == 'cross':
        rank_re, hit_re = metrics_records["testing"]["rank"][0], metrics_records["testing"]["hit"][0]
    elif train_mode == 'self':
        rank_re, hit_re = metrics_records["validation"]["rank"][mid], metrics_records["validation"]["hit"][mid]
    return SN, rank_re, hit_re


def run_experiments(args, run_id=0, device=None, USE_POI=False, model_type='S', unit=None):
    IS_NEG = (args.intersect == 1)

    if args.data_name in ["weibo"]:
        dense_name = 'isp'
        vid_list, _, _ = load_vids(args.data_path)
        vid_size = len(vid_list)
        sample_users = samples_generator(args.data_path, args.data_name, threshold=args.threshold)
        data_dense = load_data_match_telecom(args.data_path, 'isp', sample_users=sample_users,
                                             poi_type=args.poi_type)
        data_sparse = load_data_match_sparse(args.data_path, args.data_name, sample_users=sample_users,
                                             poi_type=args.poi_type)
    elif args.data_name == 'foursquare':
        dense_name = 'twitter'
        data_dense, data_sparse, global_location, global_location_lookup = load_data_match_tf(args.data_path)
        vid_size = len(global_location)
        USE_POI = False

    data_dense_split, user_locations_dense = data_split2(data_dense, match_label=True,
                                                         vid_size=vid_size, noise_th=args.noise, poi=USE_POI)
    data_sparse_split, user_locations_sparse = data_split2(data_sparse, match_label=True,
                                                           vid_size=vid_size, noise_th=args.noise, poi=USE_POI)

    print("load {} data!".format(dense_name))
    data_input_dense_siamese = data_train_match_fix2(data_dense_split, data_dense_split,
                                                     negative_sampling=args.neg, negative_candidates=False,
                                                     user_locations_dense=user_locations_dense,
                                                     user_locations_sparse=user_locations_sparse)
    print("load {} data!".format(args.data_name))
    data_input = data_train_match_fix2(data_sparse_split, data_dense_split,
                                       negative_sampling=args.neg, negative_candidates=IS_NEG,
                                       user_locations_dense=user_locations_dense,
                                       user_locations_sparse=user_locations_sparse)

    # build model
    SN_sparse = SiameseNet(loc_size=vid_size, tim_size=24, loc_emb_size=args.loc_emb_size,
                           tim_emb_size=args.tim_emb_size, hidden_size=args.hidden_size,
                           batch_size=args.batch_size, device=device, loss_mode=args.loss_mode,
                           mod=args.rnn_mod, attn_mod=args.attn_mod, layers=args.layers, fusion=model_type,
                           poi_size=args.poi_size if USE_POI else None,
                           poi_emb_size=args.poi_emb_size if USE_POI else None)
    SN_dense = SiameseNet(loc_size=vid_size, tim_size=24, loc_emb_size=args.loc_emb_size,
                          tim_emb_size=args.tim_emb_size, hidden_size=args.hidden_size,
                          batch_size=args.batch_size, device=device, loss_mode=args.loss_mode,
                          mod=args.rnn_mod, attn_mod=args.attn_mod, layers=args.layers, fusion=model_type,
                          poi_size=args.poi_size if USE_POI else None,
                          poi_emb_size=args.poi_emb_size if USE_POI else None)
    SN_sparse = SN_sparse.to(device)
    SN_dense = SN_dense.to(device)

    # pretrain step
    rank_pre, hit_pre = None, None
    if args.pretrain:
        if run_id == 0:
            SN_dense, rank_pre, hit_pre = run_siamese(SN_dense, args, data_input_dense_siamese, args.lr_pretrain,
                                                      train_mode='self', device=device, USE_POI=USE_POI)
        else:
            SN_dense.load_state_dict(torch.load(args.save_path + "/SN-pre.m"))

        if unit == "N":
            pass
        elif unit == "E":
            # SN_sparse.embed = SN_dense.embed
            SN_sparse.encoder_top = SN_dense.encoder_top
            SN_sparse.encoder_down = SN_dense.encoder_down
            SN_sparse.attn_top = SN_dense.attn_top
            SN_sparse.attn_down = SN_dense.attn_down
            SN_sparse.fc_final2 = SN_dense.fc_final2
            SN_sparse.fc_final1 = SN_dense.fc_final1
        elif unit == "R":
            SN_sparse.embed = SN_dense.embed
            # SN_sparse.encoder_top = SN_dense.encoder_top
            # SN_sparse.encoder_down = SN_dense.encoder_down
            SN_sparse.attn_top = SN_dense.attn_top
            SN_sparse.attn_down = SN_dense.attn_down
            SN_sparse.fc_final2 = SN_dense.fc_final2
            SN_sparse.fc_final1 = SN_dense.fc_final1
        elif unit == "C":
            SN_sparse.embed = SN_dense.embed
            SN_sparse.encoder_top = SN_dense.encoder_top
            SN_sparse.encoder_down = SN_dense.encoder_down
            # SN_sparse.attn_top = SN_dense.attn_top
            # SN_sparse.attn_down = SN_dense.attn_down
            SN_sparse.fc_final2 = SN_dense.fc_final2
            SN_sparse.fc_final1 = SN_dense.fc_final1
        elif unit == "F":
            SN_sparse.embed = SN_dense.embed
            SN_sparse.encoder_top = SN_dense.encoder_top
            SN_sparse.encoder_down = SN_dense.encoder_down
            SN_sparse.attn_top = SN_dense.attn_top
            SN_sparse.attn_down = SN_dense.attn_down
            # SN_sparse.fc_final2 = SN_dense.fc_final2
            # SN_sparse.fc_final1 = SN_dense.fc_final1
        else:
            # default: ERCF
            SN_sparse.load_state_dict(SN_dense.state_dict())
    # normal experiments
    SN_sparse, rank_32, hit_32 = run_siamese(SN_sparse, args, data_input, args.lr_match,
                                             train_mode='cross', device=device, USE_POI=USE_POI)
    rank_neg, hit_neg = rank_32, hit_32

    return SN_sparse, rank_neg, hit_neg, rank_pre, hit_pre

from __future__ import print_function, division

import time
import json
import numpy as np
from sklearn.neighbors import KDTree

TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.3
VALID_SPLIT = 0.1


# codes for loading data from private telecom trajectories.
def samples_generator(data_path, data_name, threshold=2000, seed=1):
    tmp = []
    np.random.seed(seed=seed)
    with open(data_path + data_name) as fid:
        for line in fid:
            user, trace = line.split("\t")
            trace_len = len(trace.split('|'))
            # trace_len = 0
            tmp.append([trace_len, user])
    np.random.shuffle(tmp)
    samples = sorted(tmp, key=lambda x: x[0], reverse=True)
    samples_return = {}
    for u in [x[1] for x in samples[:threshold]]:
        samples_return[u] = len(samples_return)
    return samples_return


def load_vids(data_path, data_name="baseLoc"):
    vid_list = {}
    vid_lookup = {}
    vid_array = []
    poi_info = json.load(open(data_path + "poi_info.json"))
    with open(data_path + data_name) as fid:
        for line in fid:
            bid, lat, lon = line.strip("\r\n").split("_")
            lat, lon = float(lat), float(lon)
            if bid not in vid_list:
                cid = len(vid_list) + 1
                vid_list[bid] = [cid, (lat, lon), poi_info[bid][3:]]
                vid_lookup[cid] = [bid, (lat, lon)]
                vid_array.append((lat, lon))
    vid_array = np.array(vid_array)
    kdtree = KDTree(vid_array)
    return vid_list, vid_lookup, kdtree


def load_data_match_telecom(data_path, data_name, sample_users=None, poi_type=0):
    ##################
    filter_short_session = 3
    sessions_count_min = 3
    ##################
    vid_list, vid_lookup, kdtree = load_vids(data_path)
    data = {}
    with open(data_path + data_name) as fid:
        for line in fid:
            user, traces = line.strip("\r\n").split("\t")
            if sample_users is not None:
                if user not in sample_users:
                    continue
            sessions = {}
            for tr in traces.split('|'):
                points = tr.split(",")
                if len(points) > 1:
                    if len(points) == 3:
                        tim, bid, lat_lon = points
                    elif len(points) == 2:
                        tim, lon_lat = points
                        lon, lat = [float(x) for x in lon_lat.split("_")]
                        dist, ind = kdtree.query([(lat, lon)], k=1)
                        bid = vid_lookup[ind[0][0] + 1][0]
                    if bid in vid_list:
                        vid = vid_list[bid][0]
                        tim = int(tim)
                        day = int(tim / 24)
                        if poi_type == 3:
                            poi = np.zeros(21).tolist()
                        else:
                            poi = vid_list[bid][2][poi_type]
                        if day not in sessions:
                            sessions[day] = [[vid, tim % 24, poi]]
                        else:
                            sessions[day].append([vid, tim % 24, poi])
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
            if len(sessions_filter) >= sessions_count_min:
                data[user] = {"sessions": sessions_filter}
        print("telecom users:{}".format(len(data.keys())))
    return data


def load_data_match_sparse(data_path, data_name, sample_users, poi_type=0):
    vid_list, vid_lookup, kdtree = load_vids(data_path)
    #######################
    # default settings
    hour_gap = 24  # 24
    session_max = 20  # 20
    #######################
    filter_short_session = 3
    sessions_count_min = 1
    data = {}
    with open(data_path + data_name) as fid:
        for line in fid:
            user, traces = line.strip("\r\n").split("\t")
            if user not in sample_users:
                continue
            sessions = {}
            for i, tr in enumerate(traces.split('|')):
                points = tr.split(",")
                if len(points) > 1:
                    if len(points) == 3:
                        tim, bid, lat_lon = points
                    elif len(points) == 2:
                        tim, lon_lat = points
                        lon, lat = [float(x) for x in lon_lat.split("_")]
                        dist, ind = kdtree.query([(lat, lon)], k=1)
                        bid = vid_lookup[ind[0][0] + 1][0]
                else:
                    continue
                if bid in vid_list:
                    vid = vid_list[bid][0]
                    tid = int(tim)
                else:
                    continue
                if poi_type == 3:
                    poi = np.zeros(21).tolist()
                else:
                    poi = vid_list[bid][2][poi_type]
                record = [vid, tid % 24, poi]
                sid = len(sessions)
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) > hour_gap or len(sessions[sid - 1]) > session_max:
                        sessions[sid] = [record]
                    else:
                        sessions[sid - 1].append(record)
                last_tid = tid
                sessions_filter = {}
                for s in sessions:
                    if len(sessions[s]) >= filter_short_session:
                        sessions_filter[len(sessions_filter)] = sessions[s]
            if len(sessions_filter) >= sessions_count_min:
                data[user] = {"sessions": sessions_filter}
    return data


def data_extract2(id_list, sessions, poi=False, uid=None):
    tmp = []
    for did in id_list:
        trace = sessions[did]

        loc_np = [s[0] for s in trace]
        tim_np = [s[1] for s in trace]

        if poi is False:
            tmp.append((loc_np, tim_np, uid))
        else:
            poi_np = [s[2] for s in trace]
            tmp.append((loc_np, tim_np, poi_np, uid))
    return tmp


def extract_locations(id_list, sessions):
    locations = []
    for did in id_list:
        trace = sessions[did]
        locations.extend([s[0] for s in trace])
    return set(locations)


def data_split2(data_neural, candidates=None, match_label=False, poi=False, noise_th=0, vid_size=0):
    if noise_th > 0:
        assert poi is False
    data_split = []
    uid_encode = {}
    user_locations = {}
    for uid in data_neural:
        if match_label:
            uid_encode[uid] = uid
        else:
            uid_encode[uid] = len(uid_encode)
    if candidates is None:
        candidates = data_neural.keys()
    for uid in candidates:
        sessions = data_neural[uid]['sessions']
        uuid = uid_encode[uid]
        days = sessions.keys()
        if noise_th > 0:
            sessions_noise = random_noise2(days, sessions, vid_size, noise_th)
        else:
            sessions_noise = sessions
        traces = data_extract2(days, sessions_noise, poi=poi, uid=uuid)
        data_split.extend(traces)
        user_locations[uuid] = extract_locations(days, sessions)

    return data_split, user_locations


def random_noise2(days, sessions, vid_size, noise_th=0):
    location_candidates = range(vid_size)
    location_self = []

    for did in days:
        trace = sessions[did]
        loc_np = [s[0] for s in trace]
        tim_np = [s[1] for s in trace]
        location_self.extend(loc_np)

    sessions_noise = {}
    noise = list(set(location_candidates) - set(location_self))
    for did in days:
        trace = sessions[did]
        loc_np = [s[0] for s in trace]
        tim_np = [s[1] for s in trace]

        loc_id = range(len(loc_np))
        np.random.shuffle(loc_id)
        np.random.shuffle(noise)
        for i in loc_id[:noise_th]:
            loc_np[i] = noise[i]
        trace_noise = zip(loc_np, tim_np)
        sessions_noise[did] = trace_noise
    return sessions_noise


def candidates_intersect(user_locations_sparse, user_locations_dense, sparse_users, dense_users):
    candidates = {}
    candidates_count = [0, 0]
    for user in sparse_users:
        sparse_locations = user_locations_sparse[user]
        candidates[user] = []
        for user2 in dense_users:
            dense_locations = user_locations_dense[user2]
            intersect_locations = dense_locations & sparse_locations
            if user2 != user and len(intersect_locations) > 0:
                candidates[user].append(user2)
        if len(candidates[user]) == 0:
            candidates[user] = list(set(dense_users) - set([user]))
        else:
            candidates_count[0] += 1
            candidates_count[1] += len(candidates[user])
    print("sparse users:{} dense users:{} find candidates:{} average candidates:{}".format(
        len(sparse_users), len(dense_users), candidates_count[0], candidates_count[1] / candidates_count[0]))
    return candidates


def extract_user_from_trace(idx, data, range_list):
    users_traces_id = {}
    for i in range_list:
        trace_id = idx[i]
        user_id = data[trace_id][-1]
        if user_id not in users_traces_id:
            users_traces_id[user_id] = [trace_id]
        else:
            users_traces_id[user_id].append(trace_id)
    return users_traces_id


def data_train_match_fix2(data_sparse, data_dense, seed=1, negative_sampling=32,
                          negative_candidates=False, user_locations_sparse=None, user_locations_dense=None):
    np.random.seed(seed)
    data_input = {"train": [], "valid": [], "test": []}
    lens_sparse, lens_dense = len(data_sparse), len(data_dense)
    idx_sparse, idx_dense = range(lens_sparse), range(lens_dense)
    np.random.shuffle(idx_sparse)
    np.random.shuffle(idx_dense)

    # for training
    for data_mode in ["train", "valid", "test"]:
        rl_sparse, rl_dense = None, None
        if data_mode == "train":
            rl_sparse = range(int(lens_sparse * TRAIN_SPLIT))
            rl_dense = range(int(lens_dense * TRAIN_SPLIT))
        elif data_mode == "valid":
            rl_sparse = range(int(lens_sparse * TRAIN_SPLIT), int(lens_sparse * (TRAIN_SPLIT + VALID_SPLIT)))
            rl_dense = range(int(lens_dense * TRAIN_SPLIT), int(lens_dense * (TRAIN_SPLIT + VALID_SPLIT)))
        elif data_mode == "test":
            rl_sparse = range(int(lens_sparse * (TRAIN_SPLIT + VALID_SPLIT)), lens_sparse)
            rl_dense = range(int(lens_dense * (TRAIN_SPLIT + VALID_SPLIT)), lens_dense)

        users_sparse = extract_user_from_trace(idx_sparse, data_sparse, rl_sparse)
        users_dense = extract_user_from_trace(idx_dense, data_dense, rl_dense)
        common_users = list(set(users_dense.keys()) & set(users_sparse.keys()))
        if negative_candidates:
            candidate_users2 = candidates_intersect(user_locations_sparse, user_locations_dense,
                                                    users_sparse.keys(), users_dense.keys())
        print(data_mode + " common users:{}".format(len(common_users)))
        for user_id in common_users:
            trace_pool_sparse = users_sparse[user_id]
            trace_pool_dense = users_dense[user_id]
            candidate_users = list(set(users_dense.keys()) - set([user_id]))
            for i in range(len(trace_pool_sparse)):
                for j in range(len(trace_pool_dense)):
                    if data_mode in ["train"]:
                        data_input[data_mode].append(
                            (data_sparse[trace_pool_sparse[i]][:-1],
                             data_dense[trace_pool_dense[j]][:-1],
                             1, user_id, user_id))
                        if negative_candidates:
                            fid = candidate_users2[user_id][np.random.randint(0, len(candidate_users2[user_id]))]
                        else:
                            fid = candidate_users[np.random.randint(0, len(candidate_users))]
                        trace_pool_dense_fake = users_dense[fid]
                        did = trace_pool_dense_fake[np.random.randint(0, len(trace_pool_dense_fake))]
                        fi = np.random.randint(0, len(trace_pool_sparse))
                        data_input[data_mode].append(
                            (data_sparse[trace_pool_sparse[fi]][:-1],
                             data_dense[did][:-1], 0, user_id, fid))
                    elif data_mode in ["test", "valid"]:
                        test_set = []
                        test_set.append(
                            (data_sparse[trace_pool_sparse[i]][:-1],
                             data_dense[trace_pool_dense[j]][:-1],
                             1, user_id, user_id))
                        for _ in range(negative_sampling - 1):
                            if negative_candidates:
                                fid = candidate_users2[user_id][np.random.randint(0, len(candidate_users2[user_id]))]
                            else:
                                fid = candidate_users[np.random.randint(0, len(candidate_users))]
                            trace_pool_dense_fake = users_dense[fid]
                            did = trace_pool_dense_fake[np.random.randint(0, len(trace_pool_dense_fake))]
                            test_set.append(
                                (data_sparse[trace_pool_sparse[i]][:-1],
                                 data_dense[did][:-1], 0, user_id, fid))
                        data_input[data_mode].append(test_set)
    print("train:{} valid:{} test:{}".format(len(data_input["train"]),
                                             len(data_input["valid"]), len(data_input["test"])))
    return data_input


# codes for loading data from Transferring heterogeneous links across location-based social networks.
# Jiawei Zhang, Xiangnan Kong, and Philip S. Yu. WSDM 2014.

def load_txt_tf(data_path, data_name):
    traces_f = {}
    location_f = {}
    with open(data_path + data_name) as fid:
        for line in fid:
            user_id, time_id, loca_id = line.strip("\r\n").split("\t")
            # decimals=2 means 1KMx1KM grids
            lon, lat = [np.around(float(x), decimals=2) for x in loca_id.split("_")]
            grid = "_".join([str(lon), str(lat)])
            if user_id not in traces_f:
                traces_f[user_id] = [[time_id, grid, loca_id]]
            else:
                traces_f[user_id].append([time_id, grid, loca_id])
            if grid not in location_f:
                location_f[grid] = 1
            else:
                location_f[grid] += 1
    return traces_f, location_f


def location_filter_tf(traces_f, global_location, threshold=None):
    traces_f_n = {}
    for i, user_id in enumerate(traces_f):
        traces = traces_f[user_id]
        traces_filter = []
        for tr in traces:
            if tr[1] not in global_location:
                continue
            else:
                tr_new = [tr[0], global_location[tr[1]]]
            traces_filter.append(tr_new)
        traces_sorted = sorted(traces_filter, key=lambda x: x[0], reverse=False)
        traces_f_n[user_id] = traces_sorted
        if threshold is not None and i > threshold:
            break
    return traces_f_n


def generate_sessions_tf(traces):
    #######################
    # default settings
    hour_gap = 24
    session_max = 20
    filter_short_session = 3
    sessions_count_min = 1
    #######################

    data = {}
    for user_id in traces:
        sessions = {}
        for i, trace in enumerate(traces[user_id]):
            tim, vid = trace
            # tim format: '200903012314'
            tid = int(tim)
            struct_time = time.strptime(tim, "%Y%m%d%H%M")
            record = [vid, struct_time.tm_hour]
            sid = len(sessions)
            if i == 0 or len(sessions) == 0:
                sessions[sid] = [record]
            else:
                if (tid - last_tid) > hour_gap * 60 or len(sessions[sid - 1]) > session_max:
                    sessions[sid] = [record]
                else:
                    sessions[sid - 1].append(record)
            last_tid = tid
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
        if len(sessions_filter) >= sessions_count_min:
            data[user_id] = {"sessions": sessions_filter}
    return data


def load_data_match_tf(data_path, threshold=None):
    traces_f, location_f = load_txt_tf(data_path, "Foursquare")
    traces_t, location_t = load_txt_tf(data_path, "Tweet")
    print("Primary Foursquare users:{} locations:{}".format(len(traces_f), len(location_f)))
    print("Primary Tweet users:{} locations:{}".format(len(traces_t), len(location_t)))

    location_t1 = [x for x in location_t if location_t[x] > 1]
    location_f1 = [x for x in location_f if location_f[x] > 1]
    global_location = {}
    global_location_lookup = {}
    # prepare data for baselines
    baseloc = []
    basecount = []
    basemap = []
    for x in list(set(location_t1) | set(location_f1)):
        # the start id of location should be 1
        gid = len(global_location) + 1
        global_location[x] = gid
        global_location_lookup[gid] = [gid, [float(y) for y in x.split("_")]]
        baseloc.append("_".join([str(gid), x]))
        if x in location_f:
            count_f = location_f[x]
        else:
            count_f = 0
        if x in location_t:
            count_t = location_t[x]
        else:
            count_t = 0
        basecount.append("\t".join([str(gid), str(count_t + count_f)]))
        basemap.append("\t".join([str(gid), str(gid)]))
    # with open("baseLoc", "w") as wid:
    #     wid.write("\n".join(baseloc))
    # with open("BS_Count.dat", "w") as wid:
    #     wid.write("\n".join(basecount))
    # with open("BSMap_New.txt", "w") as wid:
    #     wid.write("\n".join(basemap))

    traces_f = location_filter_tf(traces_f, global_location, threshold=threshold)
    traces_t = location_filter_tf(traces_t, global_location, threshold=threshold)
    print("Filtered Foursquare: users:{} Tweet: users{}".format(len(traces_f), len(traces_t)))

    traces_f = generate_sessions_tf(traces_f)
    traces_t = generate_sessions_tf(traces_t)
    return traces_t, traces_f, global_location, global_location_lookup

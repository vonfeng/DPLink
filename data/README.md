This data(Weibo-ISP) is used in our paper `DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data.`

# Rights
We have collected and processed this data in collaboration with our partners, and it is intended for academic use only. Redistribution of this data is not permitted without our explicit permission.

# Data
The data contains four parts as follows
- isp           # dense trajectory, you should first extract the isp.zip file before processing
- weibo         # sparses trajectory
- baseLoc       # location info
- poi_info.json # location info with POI

You can process these data by using codes in [preprocessing](https://github.com/vonfeng/DPLink/blob/master/codes/preprocessing.py), including functions, `load_vids` for baseLoc and poi_info.json, `load_data_match_telecom` for isp, and `load_data_match_sparse` for weibo

# Citation
```latex
@inproceedings{feng2019dplink,
  title={DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data},
  author={Feng, Jie and Zhang, Mingyang and Wang, Huandong and Yang, Zeyu and Zhang, Chao and Li, Yong and Jin, Depeng},
  booktitle={The World Wide Web Conference},
  pages={459--469},
  year={2019},
  organization={ACM}
}
```
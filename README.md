# DPLink

PyTorch implementation  for **DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data.** *Jie Feng, Mingyang Zhang, Huandong Wang, Zeyu Yang, Chao Zhang, Yong Li, Depeng Jin. WWW 2019.* If you find our code is useful for your research, you can cite our paper by:
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

# Datasets (updated 2024.06.16)
- **ISP-Weibo Data** (main data used in the paper)
	- ~~This is the private data collected and processed by ourselves and partners. We cannot directly published it due to the privacy issue. *If you are interested in it and want to use it in your paper for academic purpose, you can contact with us via the email in this [page](http://fi.ee.tsinghua.edu.cn/~liyong/) with your identity information.*~~ We have uploaded the data in [data](./data/), please follow the README.md to process the data. This data is intended for academic use only. Redistribution of this data is not permitted without our explicit permission.
- **Foursquare-Twitter Data**
	- This data is  from *Transferring heterogeneous links across location-based social networks.  Jiawei Zhang, Xiangnan Kong, and Philip S. Yu. WSDM 2014.* We have no right to directly publish it. If you are interested in this dataset, you can contact with the original author to access the dataset.

# Requirements
- **Python 2.7**
- **PyTorch 0.4**
- tqdm 4.22
- mlflow 0.5
- numpy 1.14.0
- setproctitle 1.1.10
- scikit-learn 0.19.1

# Project Structure
- run.py # scripts for run experiments
- match.py # training codes
- preprocessing.py # trajectory data preprocessing
- utils.py # utils for training
- models.py # models
- GlobalAttention.py # attention scripts from opennmt-py

# Usage
To train a new model (default settings are recorded in the run.py)

> python run.py --data=foursquare --model=ERPC --pretrain=1 --pretrain_unit=ERCF

*E: embedding, R: rnn, P: pooling, C: co-attention, F: fully connected network.* *ERPC* is the default model in paper, model name can also be *ERC*(without pooling). *ERCF* is the default pretrain mode in paper, which means all the components in the model are pretrained. You can choose *E, R, C, F* for only pretrain selected component and *N* is for non-pretrain.

# Acknowledgements
Baselines from [traditional baselines](https://github.com/whd14/De-anonymization-of-Mobility-Trajectories), [TULER](https://github.com/gcooq/TUL) and [t2vec](https://github.com/boathit/t2vec). Some codes from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), [InferSent](https://github.com/facebookresearch/InferSent) and [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm).

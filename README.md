# DPLink

PyTorch implementation  for **DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data.** *Jie Feng, Mingyang Zhang, Huandong Wang, Zeyu Yang, Chao Zhang, Yong Li, Depeng Jin. WWW 2019.* If you find our code is useful for your research, you can cite our paper by:
```latex
@article{jie2019dplink,
title={DPLink: User Identity Linkage via Deep Neural Network From Heterogeneous Mobility Data},
author={Feng, Jie and Zhang, Mingyang and Wang, Huandong and Yang, Zeyu and Zhang, Chao and Li, Yong and Jin, Depeng},
booktitle={Proceedings of the 2019 World Wide Web Conference},
year={2019}
}
```

# Datasets
There are two datasets in our paper: ISP and Foursquare. ISP data cannot be published due to the privacy issue. Foursquare data is from *Transferring heterogeneous links across location-based social networks.  Jiawei Zhang, Xiangnan Kong, and Philip S. Yu. WSDM 2014.* We have no right to directly publish it. If you are interested in this dataset, you can contact with the original author to access the dataset.

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

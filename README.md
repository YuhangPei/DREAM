# DREAM: Dual Data-centric Separation with Circular Mixup for Noise-resistant Time Series Learning

This is the training code for our work "DREAM: Dual Data-centric Separation with Circular Mixup for Noise-resistant Time Series Learning". 
## Abstract
Deep Neural Networks (DNNs) have garnered criticism since they are easily overfitted on noisy labels. To enhance the robustness of DNNs, label noise learning (LNL) endeavors to identify and mitigate the impact of noisy labels during the training process. Nevertheless, most of these methods focus on computer vision, and time series data suffer from the same issue in real world scenario. Moreover, we argue that existing methods heavily rely on static confidence to identify clean samples and neglect the potential for interaction between noisy and clean data that could better harness the noisy data. Toward this end, in this paper, we propose an effective data mixup framework for time series LNL. On the one hand, we assume that samples with similar features share similar labels and infer each sample's pseudo label via the constructed $k$-NN graph to capture the corresponding pseudo margins. Based on this, a flexible threshold is learned to identify confident clean samples from both neighbor and model perspectives. On the other hand, we interpolate between clean and noisy samples via the proposed circular embedding mixup to improve their representation learning. Then, we introduce a delta-based consistency to better utilize noisy data. Experimental results on a wide range of publicly accessible datasets reveal the effectiveness of our MixTS. 
<div align="center">
<img src="Pic/MixTS.png" width="70%">
</div>

## Data
We evaluate our model on publicly available time-series classification datasets from the UCR and UEA repositories:

[The UCR time series archive](https://ieeexplore.ieee.org/abstract/document/8894743)

[The UEA multivariate time series classification archive, 2018](https://arxiv.org/abs/1811.00075)

All datasets are downloaded automatically during training if they do not exist.

## Usage
To train CTW on 13 benchmark datasets mentioned in this paper, run
```bash
nohup python ./src/main.py --model MixTS --epochs 300 --lr 1e-3 --label_noise 0 --embedding_size 32 --ni 0.3 --num_workers 1 --mean_loss_len 10 --gamma 0.3 --cuda_device 0 --outfile MixTS.csv >/dev/null 2>&1 &
```
The results are put in ```./statistic_results/```.


## Acknowledgement
We adapted the CTW open-source code to implement our algorithms
* [CTW](https://github.com/qianlima-lab/CTW.git)

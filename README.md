# Toward Stable, Interpretable, and Lightweight Hyperspectral Super-resolution
Wen-jin Guo, Weiying Xie, Kai Jiang, Yunsong Li, Jie Lei, Leyuan Fang

## Abstract
For real applications, existing HSI-SR methods are not only limited to unstable performance under unknown scenarios but also suffer from high computation consumption. 

In this paper, we develop a new coordination optimization framework for stable, interpretable, and lightweight HSI-SR. Specifically, we create a positive cycle between fusion and degradation estimation under a new probabilistic framework. The estimated degradation is applied to fusion as guidance for a degradation-aware HSI-SR. Under the framework, we establish an explicit degradation estimation method to tackle the indeterminacy and unstable performance caused by the black-box simulation in previous methods. Considering the interpretability in fusion, we integrate spectral mixing prior into the fusion process, which can be easily realized by a tiny autoencoder, leading to a dramatic release of the computation burden. Based on the spectral mixing prior, we then develop a partial fine-tune strategy to reduce the computation cost further. 

Comprehensive experiments demonstrate the superiority of our method against the state-of-the-arts under synthetic and real datasets. For instance, we achieve a $2.3$ dB promotion on PSNR with $120\times$ model size reduction and $4300\times$ FLOPs reduction under the CAVE dataset. 

## Requirements
* cuda 10.1
* Python 3.7.6, Pytorch 1.5.0

## Evaluation on CAVE
1. The synthesized six blur kernels used in our paper can be obtained from [here](./Degradation_params/blur_kernel). 

2. Before test our method, we should synthesized the datasets under different degradations. 

   Firstly, we download the CAVE dataset from [here](http://www.cs.columbia.edu/CAVE/databases/) into ./CAVE. 

   Then, run this command:
    ```
    python ./DatasetSyn/ProcessCave.py
    ```
   Next, run this command to generate HR-MSIs and LR-HSIs: 
    ```
    python ./DatasetSyn/SynImagesCave.py
    ```
3. To test our method, run this command:
    ```
    python demo_cave.py
    ```
    The test results will be saved in [here](./results). Metrics will be recorded by [index_record_test.xls](./index_record_test.xls) and [index_record_train.xls](./index_record_train.xls). 
4. To evaluate the final performance, run this command:
    ```
    python statistic.py
    ```
5. The settings of network structure and training/fine-tuning parameters are contained in [setting.json](./setting.json). Note that we adjust the decoder to 5-layer CNN, which improves the accuracy with a negligible increase on compututational burden. 

## Contact
If you have any question, please feel free to concat Hong Wang (Email: guowenjin@stu.xidian.edu.cn)



   

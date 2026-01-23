
<div align=center>

<h1> S²KT: Modeling Uncertainty in Knowledge Tracing via Semantic-aware Structured Gaussian Distributions </h1>





</div>



## 🚀 Getting Start


### 0️⃣ Environment

Before you start, make sure you have the following installed:
* **Python**: 3.9
* **CUDA**: 11.7 (if using GPU)
* **PyTorch**: 1.13.1
* **PyTorch Geometric**: 2.4.0
* **NumPy**: 1.26.2
* **OmegaConf**: 2.3.0
* **NetworkX**: 3.1

### 1️⃣ Datasets

#### Step 1: Download Raw Datasets
Please download the specific raw data files from the official repository or the provided links below.
| Dataset | Filename | Size | Download Link |
| :--- | :--- | :--- | :--- |
| **Assistment09** | `skill_builder_data_corrected_collapsed.csv` | 63MB | [Official](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data?authuser=0) |

#### Step 2: Organize Data Directory
Unzip the downloaded files and organize them into the `data/assist09/raw_data` directory as follows. **This structure is strictly required** for the data loader to work.

### 2️⃣ Guide for Running S²KT
To train the S²KT model on the default dataset, simply run:
```bash
# Train on the default dataset (e.g., Assistments09)
python train.py --model_name S2KT --dataset_name assist2009
```

## 🛠️ Modules

The S²KT architecture is composed of several specialized modules designed to capture different aspects of the learning process. Below is a detailed breakdown of the **core components**, illustrating how they process data and interact to predict student performance.
### 0️⃣ Semantic Propagation Model
#### 💡 Illustrative Example (Case Study)
[Semantic Propagation Case Study](./model-case%20study/Semantic%20Propagation%20Model.md)

### 1️⃣ Dynamic GMM Attention Model

#### 💡 Illustrative Example (Case Study)
[Dynamic GMM Attention Case Study](./model-case%20study/Dynamic%20GMM%20Attention.md)


## 🎨Colorblind-Friendly View (Color4Good)

We recognize the importance of inclusive data visualization. Therefore, we have incorporated a Colorblind-Friendly View to further enhance the accessibility and readability of our results.

### 0️⃣ Ablation
![ablation](./assets/ablation_Color4good.png)

### 1️⃣ Visualization of Semantic Structure

![kc_semantic|center](./assets/kc_semantic_Color4Good.png)


## 📚Datasets

Place the [assist09](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data?authuser=0), [assist12](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect), [algebra05](https://pslcdatashop.web.cmu.edu/KDDCup),  [junyi](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy), [slepemapy](https://www.fi.muni.cz/adaptivelearning/?a=data) and [ednet](https://github.com/riiid/ednet) source files in the dataset directory, and process the data using the following commands respectively:



The statistics of the 6 datasets after processing are as follows:

| Datasets | #students | #questions | #concepts | #interactions |
| :--- | :--- | :--- | :--- | :--- |
| Assistment09 | 3,852 | 17,737 | 123 | 282,619 |
| Assistment12 | 46,674 | 179,999 | 265 | 6,123,270 |
| AL2005 | 574 | 173,650 | 112 | 609,979 |
| Junyi | 191,874 | 721 | 39 | 25,852,548 |
| Slepemapy | 84,911 | 2,911 | 1,458 | 9,797,342 |
| Ednet | 735,190 | 12,283 | 189 | 95,190,437 |



## ⚖️Baselines

We compare our proposed method with the following state-of-the-art KT models:



| Model | Category | Paper | Code | Venue |
| :--- | :--- | :--- | :--- | :--- |
| **DKT** | Sequential | [Deep Knowledge Tracing](https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf) | [Official](https://github.com/chrispiech/DeepKnowledgeTracing) | NIPS  |
| **DKVMN** | Memory | [Dynamic Key-Value Memory Networks for Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3038912.3052580) | [Official](https://github.com/jennyzzt/DKVMN) | WWW |
| **DKT+** | Sequential | [Addressing Two Problems in Deep Knowledge Tracing](https://dl.acm.org/doi/10.1145/3231644.3231647) | [Official](https://github.com/ckyeungac/deep-knowledge-tracing-plus) | L@S  |
| **KQN** | Sequential | [Knowledge Query Network for Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3303772.3303786) | [Official](https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing) | LAK |
| **Deep-IRT**| Memory | [Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable](https://arxiv.org/abs/1904.11738) | [Link](https://github.com/pykt-team/pykt-toolkit) | EDM  |
| **SAKT** | Attention | [A Self-Attentive Model for Knowledge Tracing](https://arxiv.org/abs/1907.06837) | [Link](https://github.com/pykt-team/pykt-toolkit) | EDM  |
| **GKT** | Graph | [Graph-based Knowledge Tracing](https://dl.acm.org/doi/10.1145/3350546.3352513) | [Official](https://github.com/jhljx/GKT) | WI 🏆|
| **SKVMN** | Memory | [Knowledge Tracing with Sequential Key-Value Memory Networks](https://dl.acm.org/doi/abs/10.1145/3331184.3331195) | [Link](https://github.com/HFUT-LEC/EduStudio/tree/f7b71b6325e2d7c31715ab789160c9de87e149f8) | SIGIR  |
| **PEBG** | Graph | [Improving Knowledge Tracing via Pre-training Question Embeddings](https://www.ijcai.org/Proceedings/2020/0219.pdf) | [Link](https://github.com/lyf-1/PEBG) | IJCAI  |
| **SKT** | Graph | [Structure-Based Knowledge Tracing: An Influence Propagation View](https://ieeexplore.ieee.org/document/9338285) | [Official](https://github.com/bigdata-ustc/EduKTM) | ICDM  |
| **AKT** | Attention | [Context-Aware Attentive Knowledge Tracing](https://dl.acm.org/doi/10.1145/3394486.3403282) | [Official](https://github.com/arghosh/AKT) | KDD  |
| **ATKT** | Sequential | [Enhancing Knowledge Tracing via Adversarial Training](https://dl.acm.org/doi/abs/10.1145/3474085.3475554) | [Official](https://github.com/xiaopengguo/ATKT) | MM  |
| **QIKT** | Sequential | [Improving Interpretability of Deep Sequential Knowledge Tracing Models](https://ojs.aaai.org/index.php/AAAI/article/view/26661) | [Official](https://github.com/pykt-team/pykt-toolkit) | AAAI  |
| **SimpleKT**| Attention | [SimpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing](https://openreview.net/pdf?id=9HiGqC9C-KA) | [Official](https://github.com/pykt-team/pykt-toolkit) | ICLR  |
| **DTransformer**| Attention | [Tracing Knowledge Instead of Patterns](https://dl.acm.org/doi/10.1145/3543507.3583255) | [Official](https://github.com/yxonic/DTransformer) | WWW  |
| **AT-DKT** | Sequential | [Enhancing Deep Knowledge Tracing with Auxiliary Tasks](https://dl.acm.org/doi/10.1145/3543507.3583866) | [Official](https://github.com/pykt-team/pykt-toolkit) | WWW  |
| **FoLiBiKT**| Attention | [Forgetting-Aware Linear Bias for Attentive Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3583780.3615191) | [Official](https://github.com/skewondr/FoLiBi-toolkit) | CIKM |
| **SparseKT**| Attention | [Towards Robust Knowledge Tracing Models via k-Sparse Attention](https://dl.acm.org/doi/10.1145/3539618.3592073) | [Official](https://github.com/pykt-team/pykt-toolkit) | SIGIR |
| **StableKT** | Attention | [Enhancing Length Generalization for Attention Based Knowledge Tracing Models](https://www.ijcai.org/proceedings/2024/0654.pdf) | [Official](https://github.com/pykt-team/pykt-toolkit) | IJCAI |
| **SinKT** | Graph | [SinKT: A Structure-Aware Inductive Knowledge Tracing Model with LLM](https://dl.acm.org/doi/10.1145/3627673.3679760) | [Official](https://github.com/tubehao/SINKT) | CIKM |
| **FlucKT** | Attention | [Cognitive Fluctuations Enhanced Attention Network for Knowledge Tracing](https://ojs.aaai.org/index.php/AAAI/article/view/33562) | [Official](https://github.com/pykt-team/pykt-toolkit) | AAAI |
| **RobustKT**| Attention | [Enhancing Knowledge Tracing Through Decoupling Cognitive Pattern from Error-Prone Data](https://dl.acm.org/doi/10.1145/3696410.3714486) | [Official](https://github.com/pykt-team/pykt-toolkit) | WWW |

Note！！！: We strongly recommend referring to [pykt](https://pykt.org/)'s benchmark for baseline implementations. Their reproduced models are known to be more accurate and concise.





## 🌟Main Results

| Model | Assist09 (AUC) | Assist09 (ACC) | Assist12 (AUC) | Assist12 (ACC) | AL2005 (AUC) | AL2005 (ACC) | Junyi (AUC) | Junyi (ACC) | Slepemapy (AUC) | Slepemapy (ACC) | Ednet (AUC) | Ednet (ACC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential** | | | | | | | | | | | | |
| DKT | 0.7571 ±0.0003 | 0.7296 ±0.0002 | 0.7207 ±0.0004 | 0.7299 ±0.0001 | 0.8153 ±0.0003 | 0.8088 ±0.0002 | 0.7510 ±0.0004 | 0.8388 ±0.0001 | 0.7400 ±0.0002 | 0.8105 ±0.0003 | 0.6607 ±0.0001 | 0.6614 ±0.0004 |
| DKT Plus | 0.7614 ±0.0002 | 0.7327 ±0.0003 | 0.7241 ±0.0001 | 0.7315 ±0.0002 | 0.8188 ±0.0005 | 0.8110 ±0.0002 | 0.7506 ±0.0003 | 0.8383 ±0.0002 | 0.7414 ±0.0004 | 0.8122 ±0.0001 | 0.6643 ±0.0003 | 0.6664 ±0.0002 |
| AT-DKT | 0.7544 ±0.0004 | 0.7259 ±0.0002 | 0.7173 ±0.0003 | 0.7278 ±0.0004 | 0.8136 ±0.0002 | 0.8077 ±0.0001 | 0.7500 ±0.0003 | 0.8382 ±0.0002 | 0.7401 ±0.0004 | 0.8115 ±0.0003 | 0.6613 ±0.0002 | 0.6642 ±0.0005 |
| ATKT | 0.7344 ±0.0003 | 0.7175 ±0.0004 | 0.7029 ±0.0001 | 0.7197 ±0.0003 | 0.7925 ±0.0005 | 0.7985 ±0.0002 | 0.7360 ±0.0002 | 0.8329 ±0.0001 | 0.6906 ±0.0005 | 0.7983 ±0.0003 | 0.6297 ±0.0001 | 0.6519 ±0.0004 |
| KQN | 0.7546 ±0.0002 | 0.7288 ±0.0004 | 0.7206 ±0.0003 | 0.7300 ±0.0002 | 0.8099 ±0.0003 | 0.8063 ±0.0004 | 0.7513 ±0.0001 | 0.8394 ±0.0003 | 0.7419 ±0.0002 | 0.8103 ±0.0004 | 0.6669 ±0.0005 | 0.6652 ±0.0002 |
| QIKT | 0.7858 ±0.0004 | 0.7438 ±0.0001 | 0.7486 ±0.0003 | 0.7418 ±0.0002 | 0.8313 ±0.0005 | 0.8204 ±0.0002 | 0.7910 ±0.0004 | 0.8433 ±0.0001 | 0.7422 ±0.0002 | 0.8118 ±0.0003 | 0.7159 ±0.0002 | 0.6913 ±0.0004 |
| **Memory-Augmented** | | | | | | | | | | | | |
| DKVMN | 0.7533 ±0.0003 | 0.7274 ±0.0002 | 0.7183 ±0.0002 | 0.7313 ±0.0004 | 0.7990 ±0.0001 | 0.8013 ±0.0003 | 0.7508 ±0.0005 | 0.8395 ±0.0002 | 0.7427 ±0.0004 | 0.8098 ±0.0001 | 0.6608 ±0.0002 | 0.6634 ±0.0003 |
| SKVMN † | 0.7409 ±0.0002 | 0.7203 ±0.0004 | 0.7095 ±0.0001 | 0.7210 ±0.0003 | 0.7905 ±0.0002 | 0.7952 ±0.0002 | 0.7390 ±0.0004 | 0.8288 ±0.0001 | 0.7315 ±0.0005 | 0.8008 ±0.0003 | 0.6488 ±0.0002 | 0.6562 ±0.0001 |
| Deep-IRT | 0.7561 ±0.0001 | 0.7285 ±0.0003 | 0.7178 ±0.0004 | 0.7309 ±0.0002 | 0.7873 ±0.0005 | 0.8447 ±0.0004 | 0.7499 ±0.0002 | 0.8394 ±0.0003 | 0.7403 ±0.0001 | 0.8090 ±0.0004 | 0.6595 ±0.0002 | 0.6624 ±0.0003 |
| **Graph-Based** | | | | | | | | | | | | |
| GKT † | 0.7371 ±0.0003 | 0.7218 ±0.0001 | 0.7182 ±0.0002 | 0.7296 ±0.0004 | 0.8012 ±0.0005 | 0.8021 ±0.0003 | 0.7385 ±0.0001 | 0.8290 ±0.0002 | 0.7255 ±0.0004 | 0.7985 ±0.0003 | 0.6455 ±0.0005 | 0.6501 ±0.0002 |
| SKT † | 0.7563 ±0.0004 | 0.7321 ±0.0002 | 0.7224 ±0.0001 | 0.7312 ±0.0003 | 0.8141 ±0.0002 | 0.8085 ±0.0004 | 0.7525 ±0.0005 | 0.8402 ±0.0003 | 0.7415 ±0.0001 | 0.8115 ±0.0002 | 0.6592 ±0.0004 | 0.6610 ±0.0001 |
| PEBG | 0.7583 ±0.0001 | 0.7312 ±0.0003 | 0.7258 ±0.0002 | 0.7325 ±0.0004 | 0.8192 ±0.0005 | 0.8118 ±0.0002 | 0.7538 ±0.0003 | 0.8405 ±0.0001 | 0.7432 ±0.0004 | 0.8130 ±0.0005 | 0.6655 ±0.0003 | 0.6672 ±0.0002 |
| SinKT † | 0.7773 ±0.0003 | 0.7343 ±0.0004 | 0.7455 ±0.0002 | 0.7384 ±0.0002 | 0.8291 ±0.0001 | 0.8169 ±0.0003 | 0.7812 ±0.0004 | 0.8413 ±0.0001 | 0.7449 ±0.0002 | 0.8127 ±0.0005 | 0.7212 ±0.0004 | 0.6905 ±0.0003 |
| **Attention-based** | | | | | | | | | | | | |
| SAKT | 0.7356 ±0.0002 | 0.7129 ±0.0003 | 0.6980 ±0.0004 | 0.7197 ±0.0002 | 0.7831 ±0.0005 | 0.7948 ±0.0002 | 0.7265 ±0.0001 | 0.8324 ±0.0004 | 0.7165 ±0.0003 | 0.8088 ±0.0005 | 0.6561 ±0.0002 | 0.6610 ±0.0003 |
| AKT ‡ | 0.7850 ±0.0005 | 0.7433 ±0.0002 | 0.7457 ±0.0001 | 0.7409 ±0.0004 | 0.8241 ±0.0003 | 0.8059 ±0.0002 | 0.7908 ±0.0005 | 0.8460 ±0.0003 | 0.7502 ±0.0004 | 0.8108 ±0.0002 | 0.7154 ±0.0001 | 0.6901 ±0.0003 |
| SimpleKT | 0.7804 ±0.0002 | 0.7419 ±0.0004 | 0.7414 ±0.0005 | 0.7364 ±0.0001 | 0.8245 ±0.0002 | 0.8129 ±0.0003 | 0.7837 ±0.0004 | 0.8423 ±0.0005 | 0.7493 ±0.0001 | 0.8131 ±0.0003 | 0.7126 ±0.0002 | 0.6885 ±0.0004 |
| DTransformer | 0.7778 ±0.0004 | 0.7391 ±0.0002 | 0.7346 ±0.0003 | 0.7348 ±0.0005 | 0.8087 ±0.0001 | 0.8053 ±0.0004 | 0.7855 ±0.0002 | 0.8432 ±0.0003 | 0.7440 ±0.0005 | 0.8116 ±0.0001 | 0.7197 ±0.0004 | 0.6918 ±0.0002 |
| FlucKT ‡ | 0.7895 ±0.0003 | 0.7456 ±0.0005 | 0.7453 ±0.0002 | 0.7382 ±0.0004 | 0.8237 ±0.0003 | 0.8102 ±0.0001 | 0.7914 ±0.0005 | 0.8455 ±0.0003 | 0.7505 ±0.0002 | 0.8127 ±0.0004 | 0.7237 ±0.0001 | 0.6916 ±0.0005 |
| FoLiBiKT ‡ | 0.7899 ±0.0004 | 0.7443 ±0.0002 | 0.7454 ±0.0003 | 0.7408 ±0.0001 | 0.8230 ±0.0004 | 0.8102 ±0.0003 | 0.7923 ±0.0002 | 0.8461 ±0.0004 | 0.7510 ±0.0002 | 0.8126 ±0.0002 | 0.7233 ±0.0003 | 0.6958 ±0.0001 |
| SparseKT | 0.7772 ±0.0002 | 0.7367 ±0.0005 | 0.7377 ±0.0004 | 0.7354 ±0.0002 | 0.8231 ±0.0003 | 0.8116 ±0.0001 | 0.7757 ±0.0004 | 0.8395 ±0.0002 | 0.7388 ±0.0001 | 0.8098 ±0.0003 | 0.7150 ±0.0005 | 0.6891 ±0.0002 |
| StableKT | 0.7866 ±0.0001 | 0.7456 ±0.0004 | 0.7421 ±0.0003 | 0.7397 ±0.0002 | 0.8292 ±0.0005 | 0.8121 ±0.0003 | 0.7857 ±0.0002 | 0.8441 ±0.0004 | 0.7466 ±0.0001 | 0.8119 ±0.0002 | 0.7148 ±0.0003 | 0.6911 ±0.0005 |
| RobustKT | 0.7895 ±0.0003 | 0.7443 ±0.0002 | 0.7472 ±0.0005 | 0.7428 ±0.0001 | 0.8293 ±0.0004 | 0.8137 ±0.0003 | 0.7934 ±0.0002 | 0.8459 ±0.0001 | 0.7454 ±0.0003 | 0.8117 ±0.0004 | 0.7127 ±0.0001 | 0.6866 ±0.0003 |
| **Proposed** | | | | | | | | | | | | |
| **S²KT** | **0.7949 ±0.0004** | **0.7548 ±0.0001** | **0.7518 ±0.0003** | **0.7472 ±0.0002** | **0.8378 ±0.0003** | **0.8242 ±0.0001** | **0.8238 ±0.0003** | **0.8497 ±0.0002** | **0.7663 ±0.0002** | **0.8161 ±0.0002** | **0.7332 ±0.0002** | **0.7026 ±0.0001** |
| %Improv. | 0.63% | 1.23% | 0.43% | 0.73% | 0.78% | 0.46% | 3.83% | 0.43% | 2.04% | 0.42% | 1.31% | 0.98% |








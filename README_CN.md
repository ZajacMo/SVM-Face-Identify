# SVM 人脸识别系统

*[English](README.md)*

> **状态：已归档。** 本仓库作为课程项目保留供参考。代码可以运行、有测试、有文档，
> 但所采用的路线（直接对原始像素分类）属于基线方案，而非有竞争力的人脸识别方法。
> 复用前请先阅读[局限性](#局限性)一节。

本项目是一个人脸识别**基线**，在原始像素上比较两种分类器：使用 NumPy 从零实现的
反向传播神经网络，以及 scikit-learn 提供的 RBF 核支持向量机。

## 目录

- [项目概述](#项目概述)
- [快速开始——无需数据集](#快速开始无需数据集)
- [数据集获取与许可](#数据集获取与许可)
- [使用自有数据集](#使用自有数据集)
- [技术架构](#技术架构)
- [安装](#安装)
- [命令行参数](#命令行参数)
- [输出产物](#输出产物)
- [实验结果](#实验结果)
- [可复现性](#可复现性)
- [测试](#测试)
- [本版本改动](#本版本改动)
- [局限性](#局限性)
- [许可证](#许可证)

## 项目概述

原始项目在 CASIA-FaceV5 上训练了一个 500 类的分类器，并将神经网络与 SVM 作对比。
它有两个明显的可用性障碍：数据集不可再分发，因此任何人都无法运行；此外所报告的准确率
来自单次训练/测试划分，且超参数是手工挑选的。

本版本保留了原有算法与原有实验结果，并修复了影响复用的问题：

- **无需下载即可运行完整流程。** `--synthetic` 会即时生成一个小型标注数据集，
  整个流程数秒内完成。
- **所有报告数值均为交叉验证的均值与标准差**，不再是单次划分上的点估计。
- **训练集与测试集共用同一套类别映射。** 原代码中两个划分各自独立建立映射，
  可能对「哪个下标代表哪个人」产生分歧，导致所报告的准确率失去意义。
- **全流程均有测试覆盖**，既不需要数据集，也不需要 GPU。

## 快速开始——无需数据集

```bash
git clone https://github.com/ZajacMo/SVM-Face-Identify.git
cd SVM-Face-Identify

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --synthetic --epochs 200
```

该命令会生成合成数据集、完成划分、训练两个模型、进行评估，并把全部产物写入
`results/`。整个过程无需数据集、无需联网、无需 GPU。

合成类别之间通过颜色区分，因此其准确率会很高。**这些数值与人脸识别能力无关**，
该运行仅用于证明流程通畅。

## 数据集获取与许可

**CASIA-FaceV5 不可再分发，本仓库也不包含该数据集。** 使用它需要与 CASIA 人脸数据库
所属机构签署许可协议。

获取步骤：

1. 访问 CASIA 人脸数据库门户：
   <http://www.cbsr.ia.ac.cn/users/jianquan/english/CASIA-FaceV5.html>
2. 下载并填写许可协议表格。
3. 按页面说明将签署好的协议发送至指定邮箱。
4. 等待审批。协议处理完成后，数据集通常以压缩包形式通过 FTP 交付。

解压后请将数据集放置为如下结构（项目根目录下，每个子目录对应一个人，目录内为该人的图像）：

```
64_CASIA-FaceV5/
    000/
        000_0.bmp
        000_1.bmp
        ...
    001/
        001_0.bmp
        ...
```

随后对数据集运行流程：

```bash
python main.py --dataset 64_CASIA-FaceV5 --tune
```

**请勿提交该数据集。** 它体积庞大，且不属于你再分发的权利；`.gitignore` 中也已排除
`64_CASIA-FaceV5/`。

## 使用自有数据集

任何具备相同「一类一目录」结构的数据集均可使用。代码中没有任何针对 CASIA 的硬编码：
类别名取自目录名，图像支持 `.bmp`、`.png` 与 `.jpg`。

```bash
python main.py --dataset /path/to/my_dataset --test-size-per-class 2
```

类别名可以是数字也可以是非数字字符串；若能解析为整数则按数值排序，否则按字典序排序。

## 技术架构

```
main.py                       命令行入口与流程编排
src/
    data/
        data_split.py         训练/测试划分与合成数据生成器
        dataset.py            共享的类别映射与清单加载
    training/
        train.py              反向传播网络训练
        svm_classifier.py     SVM 训练、调参与交叉验证
    testing/
        test.py               神经网络在清单上的评估
        compare_models.py     两个模型的对比
        metrics.py            准确率、宏平均 P/R/F1、混淆矩阵
    plotting/
        plot_training_history.py  损失曲线、准确率、置信度、混淆矩阵
    utils/
        utils.py              图像预处理与 NumPy 神经网络
tests/                        pytest 测试套件，不依赖数据集
```

两个模型刻意保持独立，以便公平比较：

- **BP 神经网络**（`src/utils/utils.py`）——单隐藏层，sigmoid 激活，softmax 输出层，
  交叉熵损失，小批量梯度下降训练，仅依赖 NumPy 实现。
- **SVM**（`src/training/svm_classifier.py`）——`sklearn.svm.SVC`，RBF 核，
  可选交叉验证网格搜索调参。

两者共用 `src/data/dataset.py` 中的加载器，从而保证训练集与测试集在类别下标上完全一致。

## 安装

需要 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

运行时依赖为 `numpy`、`Pillow`、`matplotlib`、`seaborn` 与 `scikit-learn`；
`pytest` 仅在运行测试时需要。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `64_CASIA-FaceV5` | 每个类别一个子目录的数据集根路径 |
| `--synthetic` | 关闭 | 生成合成数据集而非读取 `--dataset` |
| `--synthetic-classes` | `12` | 合成模式下的类别数量 |
| `--label-dir` | `labels` | 存放训练/测试清单的目录 |
| `--output-dir` | `results` | 模型、指标与图表的输出目录 |
| `--image-size` | `32` | 输入模型的方形图像边长 |
| `--epochs` | `1000` | 神经网络训练轮数 |
| `--learning-rate` | `0.01` | 梯度更新步长 |
| `--batch-size` | `32` | 每步梯度更新的样本数 |
| `--hidden-size` | `128` | 隐藏层神经元数量 |
| `--test-size-per-class` | `1` | 每类保留的测试图像数 |
| `--seed` | `42` | 所有随机操作的种子 |
| `--cv` | `5` | 交叉验证与超参数搜索的折数 |
| `--tune` | 关闭 | 为 SVM 执行交叉验证网格搜索 |
| `--steps` | 全部 | `split,train,test,svm,compare,plot` 的逗号分隔子集 |

`--steps` 允许只运行单个阶段，例如重新评估已有模型：

```bash
python main.py --dataset 64_CASIA-FaceV5 --steps test,plot
```

## 输出产物

| 路径 | 内容 |
| --- | --- |
| `labels/classes.json` | 有序的类别名列表 |
| `labels/train_labels.json` | 训练清单，元素为 `{path, label}` |
| `labels/test_labels.json` | 测试清单，元素为 `{path, label}` |
| `results/face_model.pkl` | 训练好的神经网络（**不纳入版本控制**） |
| `results/svm_face_model.pkl` | 训练好的 SVM（**不纳入版本控制**） |
| `results/class_labels.pkl` | 神经网络的类别映射（**不纳入版本控制**） |
| `results/training_history.pkl` | 损失历史（**不纳入版本控制**） |
| `results/summary.json` | 运行摘要，含实测准确率 |
| `results/svm_metrics.json` | SVM 指标、交叉验证各折得分与最优参数 |
| `results/plots/test_results.json` | 神经网络的逐样本预测结果 |
| `results/plots/test_log.txt` | 可读的评估日志 |
| `results/plots/*.png` | 损失曲线、准确率、置信度分布、混淆矩阵 |
| `results/comparison/comparison_results.json` | 两个模型的准确率对比 |

训练好的模型被有意排除在版本控制之外。原先仅 SVM 模型就有 61 MB，使仓库臃肿，
且每次重新运行都会让工作区变脏。流程可在数秒内重新生成全部模型。

## 实验结果

### CASIA-FaceV5 原始实验（500 类）

以下数值来自本项目**改动之前**的原始运行，记录于 `results/` 中已提交的产物里。
它们基于单次训练/测试划分，此处仅作历史参考。**这些数值未被重新测量**——
数据集不可再分发，也不在本仓库中。

| 模型 | 测试准确率 | 出处 |
| --- | --- | --- |
| BP 神经网络 | **0.702**（正确 351 / 500） | `results/plots/test_results.json` |
| SVM | **0.620** | `results/comparison/comparison_results.json` |

划分细节（取自已提交的清单文件）：

- 共 500 个类别（`labels/classes.json`），2000 张训练图像、500 张测试图像。
- 原始划分为**每类保留 1 张测试图像**，因此 500 张测试图像分别来自 500 个不同的人。

两个数值均来自单次划分，没有交叉验证也没有超参数搜索，因而误差范围未知。
**应以当前代码产出的交叉验证数值为准。**

### 合成数据集运行结果

以下是在本机执行下列命令实测所得，基于当前代码生成的合成数据集。
**这些数值反映的是按颜色可分的合成图形，而非人脸**，列出仅为展示一次正常运行应有的表现：

```bash
python main.py --synthetic --synthetic-classes 12 --epochs 200 \
    --image-size 32 --cv 5 --tune
```

| 指标 | 数值 |
| --- | --- |
| SVM 交叉验证准确率 | 1.0000 +/- 0.0000 |
| SVM 测试准确率 | 1.0000 |
| SVM 最优参数 | `C=1.0`、`gamma='scale'`、`kernel='rbf'` |
| BP 神经网络测试准确率 | 0.8333（正确 10 / 12） |

## 可复现性

所有随机操作均由种子控制。`--seed`（默认 `42`）决定训练/测试划分、神经网络权重初始化、
批次打乱以及交叉验证的折分配。使用相同种子与相同参数运行两次，会得到完全相同的划分与
损失历史。

交叉验证使用带 `shuffle=True` 与固定种子的 `StratifiedKFold`，因此折分配同样可复现。

## 测试

```bash
pip install pytest
python -m pytest tests/ -v
```

测试套件覆盖数据集划分、类别映射、两个模型以及端到端的命令行流程。
**测试完全不接触 CASIA-FaceV5**：夹具会即时生成一个小型合成数据集，
因此测试在笔记本和 CI 上都只需数秒即可完成。

持续集成在 Ubuntu 与 Windows、Python 3.10 与 3.12 上运行该套件，
并额外对合成流程做冒烟测试（`.github/workflows/ci.yml`）。

## 本版本改动

正确性修复：

- **训练集与测试集现共用同一套类别映射。** 此前每个清单都各自建立
  `label -> index` 映射，两个划分可能不一致，使所报告的准确率实际上失去意义。
- **训练/测试划分现在带有随机种子。** 原先使用无种子的 `random.sample`，
  每次运行划分都不同，结果无法复现。
- **划分不再通过成员判断过滤训练列表。** 原先
  `train_images = [img for img in images if img not in test_images]`
  是 O(n²) 扫描，遇到重复文件名还可能漏掉条目。
- **类别名不再必须为数字。** `sorted(..., key=int)` 遇到任何非数字标签都会抛
  `ValueError`，使代码无法用于自定义数据集。
- **sigmoid 不再溢出。** `1 / (1 + np.exp(-x))` 在输入为较大负数时溢出并产生
  `nan` 梯度，现已改为数值稳定的两分支写法。
- **路径不再硬编码。** 评估代码混用了 `plots/` 与 `results/plots/`，
  并硬编码了空的项目根路径，导致图像路径依赖当前工作目录。
- **图像不再通过 `freopen` 式流重定向读取。** 清单与模型均改用常规文件 API。

方法学改进：

- SVM 超参数改由交叉验证网格搜索确定（`--tune`），不再硬编码。
- 结果以交叉验证各折的均值与标准差报告，并附带宏平均 precision、recall、F1
  以及混淆矩阵。

工程改进：

- 提供带文档化 `--steps` 参数的 `argparse` 命令行。
- 公开 API 全部补充英文 docstring 与类型标注。
- 增加 pytest 测试套件与 GitHub Actions 工作流。
- 模型二进制文件不再纳入版本控制。

## 局限性

复用本代码前请先阅读本节。

- **原始像素不是人脸特征向量。** 两个模型都直接使用扁平化的 RGB 图像。
  真正的人脸识别系统使用嵌入模型（ResNet 或人脸专用网络）把图像映射为紧凑描述子，
  准确率远高于此，分类代价也低得多。本仓库的数值反映的是基线水平，
  而非人脸识别能达到的水平。
- **原始测试集每人仅一张图像。** 500 个类别各留 1 张，实测准确率方差很大，
  并不能稳定反映泛化能力。
- **原始准确率未经交叉验证。** 它们来自单次划分。请以当前代码产出的
  交叉验证数值为准。
- **数据集不在仓库中，也不能加入仓库。** 所有 CASIA-FaceV5 上的结果
  都必须由已签署许可协议的使用者在本地产出。
- **合成数据集仅是流程自检。** 其准确率反映按颜色可分的图形，与人脸无关。
- **神经网络仅单隐藏层。** 没有正则化、没有早停、没有学习率调度，
  在小数据集上容易过拟合。
- **pickle 模型文件不可信。** pickle 反序列化会执行任意代码，
  不要加载来源不明的模型文件。

## 许可证

MIT，见 [LICENSE](LICENSE)。

CASIA-FaceV5 数据集**不适用**本许可证。它仅能由许可协议签署者在与所属机构
另行签订的协议范围内再分发。

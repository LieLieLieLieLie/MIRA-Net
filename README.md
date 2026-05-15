# MIRA-Net: Multi-task Microscopic Image Restoration and Attention Network

本项目实现了一个面向实验室医学图像的多任务深度学习框架 **MIRA-Net**。模型以共享残差编码器为核心，引入 CBAM 注意力模块，同时支持 HEp-2 间接免疫荧光图像的 ANA 模式分类与去噪重建，以及 PatchCamelyon 病理图像的肿瘤二分类实验。

代码主要用于论文实验复现、消融对比、指标统计、论文图表生成，以及 draw.io 论文示意图素材导出。

## 主要功能

- HEp-2 IIF 六分类：Homogeneous、Coarse speckled、Fine speckled、Centromere、Nucleolar、Cytoplasmic。
- PatchCamelyon RGB patch 二分类：Normal / Tumor。
- 多模型对比：`simple_cnn`、`residual_cnn`、`skipnet_no_cbam`、`proposed`。
- Proposed 模型：残差编码器 + CBAM 注意力 + 分类头 + HEp-2 重建解码器。
- 自动输出训练曲线、混淆矩阵、指标表、跨数据集对比、注意力图、重建图等论文图表。
- 提供 draw.io 摘要图、PSFNet 图等素材生成脚本。

## 项目结构

```text
DIP_code/
├── configs/
│   └── default.yaml
├── dataset/
│   ├── HEp-Dataset/
│   └── histopathologic-cancer-detection/
├── drawio/
│   ├── export_framework_materials.py
│   ├── generate_numbered_drawio_materials.py
│   ├── PSFNet/
│   └── *.drawio
├── outputs/
├── src/
│   └── dip_medimg/
│       ├── cli.py
│       ├── config.py
│       ├── datasets.py
│       ├── engine.py
│       ├── losses.py
│       ├── models.py
│       ├── plotting.py
│       └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

说明：

- `src/dip_medimg/`：核心训练、模型、数据、指标和绘图代码。
- `configs/default.yaml`：默认实验配置。
- `dataset/`：本地数据目录。数据集通常较大，不建议提交到 GitHub。
- `outputs/`：训练权重、指标文件和生成图表目录。通常也不建议提交完整内容。
- `drawio/`：论文示意图与素材生成脚本。

## 环境安装

建议使用 Python 3.10 或更新版本，并在独立虚拟环境中运行。

```powershell
cd D:\PythonProject\DIP\DIP_code
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果你使用已有 Anaconda 环境，也可以直接安装依赖：

```powershell
pip install -r requirements.txt
```

注意：本项目依赖 `numpy<2`。如果遇到 `AttributeError: _ARRAY_API not found`、`numpy.dtype size changed` 或 PyTorch / h5py / matplotlib 二进制兼容问题，请执行：

```powershell
pip install "numpy<2" --force-reinstall
```

## 数据准备

请将数据放在 `DIP_code/dataset/` 下，保持如下结构：

```text
dataset/
├── HEp-Dataset/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
└── histopathologic-cancer-detection/
    ├── camelyonpatch_level_2_split_train_x.h5
    ├── camelyonpatch_level_2_split_train_y.h5
    ├── camelyonpatch_level_2_split_valid_x.h5
    ├── camelyonpatch_level_2_split_valid_y.h5
    ├── camelyonpatch_level_2_split_test_x.h5
    └── camelyonpatch_level_2_split_test_y.h5
```

HEp-2 文件夹 `1` 到 `6` 分别对应六类 ANA 模式。PatchCamelyon 使用 H5 文件，`x` 文件保存图像，`y` 文件保存标签。

运行前可先检查数据路径：

```powershell
python main.py --task summary
```

## 快速开始

所有命令建议在 `DIP_code/` 根目录执行。

### 运行 HEp-2 proposed 模型

```powershell
python main.py --task hep2 --model proposed
```

### 运行 PatchCamelyon proposed 模型

```powershell
python main.py --task camelyon --model proposed
```

### 同时运行两个任务

```powershell
python main.py --task both --model proposed
```

### 运行完整消融对比

```powershell
python main.py --task hep2 --suite
python main.py --task camelyon --suite
```

或同时运行两套数据：

```powershell
python main.py --task both --suite
```

### 快速调试小样本实验

```powershell
python main.py --task both --suite --hep2-max 1000 --camelyon-max 1000 --epochs 2 --device cpu
```

## 使用已有权重评估并重生成图表

如果 `outputs/` 中已经存在训练好的权重，例如：

```text
outputs/hep2_proposed_best.pt
outputs/camelyon_proposed_best.pt
```

可以使用 eval-only 模式重新评估：

```powershell
python main.py --task both --suite --eval-only --checkpoint-dir outputs --output-dir outputs
```

## 常用命令参数

| 参数 | 说明 |
|---|---|
| `--task` | `summary`、`hep2`、`camelyon`、`both` |
| `--model` | `simple_cnn`、`residual_cnn`、`skipnet_no_cbam`、`proposed` |
| `--suite` | 运行四个模型的完整对比实验 |
| `--eval-only` | 使用已有 checkpoint 评估，不重新训练 |
| `--output-dir` | 指定输出目录 |
| `--checkpoint-dir` | 指定 checkpoint 读取目录 |
| `--device` | `auto`、`cpu`、`cuda`、`cuda:0` |
| `--hep2-max` | 限制 HEp-2 使用样本数 |
| `--camelyon-max` | 限制 PatchCamelyon 使用样本数 |
| `--epochs` | 覆盖配置文件中的训练轮数 |

查看完整帮助：

```powershell
python main.py --help
```

## 输出文件

训练和评估结果默认保存在 `outputs/`：

| 文件 | 说明 |
|---|---|
| `{dataset}_{model}_best.pt` | 最佳模型权重 |
| `{dataset}_{model}_checkpoint.pt` | 最近 checkpoint |
| `{dataset}_{model}_metrics.json` | 测试集详细指标 |
| `{dataset}_suite_summary.csv` | 完整消融实验指标汇总 |
| `{dataset}_fig1_training_curves.pdf` | 训练/验证 loss 与 accuracy |
| `{dataset}_fig2_perclass_f1_bar.pdf` | 每类 F1 对比 |
| `{dataset}_fig5_confusion_heatmaps.pdf` | 混淆矩阵 |
| `{dataset}_fig6_metric_bars.pdf` | 参数量与关键指标对比 |
| `{dataset}_fig7_stacked_correct.pdf` | 每类正确预测数量 |
| `{dataset}_fig8_violin_precision.pdf` | 每类 precision 分布 |
| `hep2_fig10_recon_panel.pdf` | HEp-2 去噪重建面板 |
| `hep2_fig11_attention_maps.pdf` | HEp-2 注意力可视化 |
| `camelyon_fig12_patch_panel.pdf` | PatchCamelyon TP/FP/FN 示例 |
| `combined_fig9_cross_dataset.pdf` | 跨数据集指标对比 |

## draw.io 素材生成

`drawio/` 目录中包含论文示意图相关脚本。

### 摘要图素材

生成 HEp-2 重建、注意力、分类预测，以及 PatchCamelyon patch 素材：

```powershell
cd drawio
python export_framework_materials.py --max-per-class 2 --device cpu
python generate_numbered_drawio_materials.py
```

输出包括：

- `drawio/framework_materials/`
- `drawio/01.png` 到 `drawio/18.png`
- `drawio/numbered_materials_mapping.md`
- `drawio/abstract_summary_preview.png`

### PSFNet 右侧素材

```powershell
cd drawio\PSFNet
python generate_psfnet_right_materials.py
```

输出包括：

- `01_psf.png`
- `02_gaussian.png`
- `03_normalization.png`
- `04_final_fixed_size_normalized_psf.png`
- `psfnet_right_four_preview.png`

## 配置文件

默认配置位于：

```text
configs/default.yaml
```

其中可修改：

- 数据路径
- batch size
- 学习率
- weight decay
- epoch 数
- HEp-2 噪声强度范围
- 重建损失权重 `reconstruction_weight`
- 模型基础通道数 `base_channels`

## 常见问题

### 1. 找不到 HEp-2 数据

确认工作目录是 `DIP_code/`，并且存在：

```text
dataset/HEp-Dataset/
```

### 2. h5py 或 torch 报 NumPy 兼容错误

执行：

```powershell
pip install "numpy<2" --force-reinstall
```

### 3. CUDA out of memory

减小 `configs/default.yaml` 中的 batch size，或使用小样本调试：

```powershell
python main.py --task both --suite --hep2-max 1000 --camelyon-max 1000 --epochs 2
```

### 4. 只想生成图表，不想重新训练

如果已有 checkpoint，使用：

```powershell
python main.py --task both --suite --eval-only --checkpoint-dir outputs
```

## 备注

本项目主要服务于 MIRA-Net 论文实验与图表复现。若公开仓库，请确认数据集、预训练权重和论文素材是否符合各自许可协议后再上传。

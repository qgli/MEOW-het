<div align="center">

[English](README.md) | **简体中文** | [日本語](README.ja.md)

# MEOW: Many Eyes, One World

### 混合相机图像的前馈式三维重建

Qiaoge Li<sup>1</sup>, Yifan Zhan<sup>2</sup>, Haijun Yang<sup>1</sup>, Haiyang Liu<sup>2</sup>, Yiyi Cai<sup>2</sup>, Chenchi Luo<sup>1</sup>

<sup>1</sup>中国移动通信有限公司研究院 &nbsp;&nbsp; <sup>2</sup>东京大学

[![arXiv](https://img.shields.io/badge/arXiv-2609.35658-b31b1b)](https://arxiv.org/abs/2609.35658)
[![项目主页](https://img.shields.io/badge/Project-Website-526a35)](https://qgli.github.io/MEOW-het/)
[![在线演示](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Try%20MEOW-e9b949)](https://huggingface.co/spaces/rubidiumcaesium/MEOW)
[![模型权重](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Manual%20approval-ffd21e)](https://huggingface.co/qgli/MEOW-4D-Space-Reconstruction)
[![激光基准](https://img.shields.io/badge/Laser%20benchmark-v1.0%20%C2%B7%20CC%20BY%204.0-2f6fb3)](https://github.com/qgli/MEOW-het/releases/tag/v1.0)
[![代码许可证](https://img.shields.io/badge/Code-Apache%202.0-3b873b)](LICENSE)

<img src="assets/inputs.jpg" alt="同一房间的四张输入：一张圆形鱼眼、一张完整全景和两张透视图" width="100%">
<img src="assets/orbit.gif" alt="四张图联合重建的点云与预测相机" width="640">

<em>来自三类相机的四张房间照片，不提供标定或相机类型标签，一次前馈联合重建；视锥表示预测相机。</em>

</div>

## 最新消息

- **2026-10-22**：[项目主页](https://qgli.github.io/MEOW-het/)与[在线 Demo](https://huggingface.co/spaces/rubidiumcaesium/MEOW)已上线；Demo 提供不消耗 GPU 推理时间的混合相机缓存样例。
- **2026-10-22**：最终 MEOW 权重（sB25，FP32）已托管至 [Hugging Face](https://huggingface.co/qgli/MEOW-4D-Space-Reconstruction)，下载申请由作者人工批准。
- **2026-09-29**：代码、配置、基准定义与激光扫描基准已发布。当时模型权重尚待发布。

## 概述

MEOW 将透视图、鱼眼图和完整 360° 全景组成同一个多视图元组，仅凭图像，通过一次前馈同时预测度量点图与相机位姿。各视图均无需提供标定、畸变参数、相机类型标签或位姿。完整全景由图像像素自动识别；部分评测集按整套数据设置全景标志，详见 [评测说明](docs/EVALUATION.md)。

MEOW 保留在透视图上预训练的 MapAnything 骨干，通过程序化数据引擎学习异构相机。引擎在连续相机模型范围内渲染场景，提供精确射线与深度，并认证每个相机采样训练元组的共视关系。

<p align="center"><img src="assets/teaser.jpg" alt="Lenscope 数据引擎及真实混合相机元组重建" width="100%"></p>

<sub>MEOW 无需外部相机标定或类型标签，一次前馈重建共享三维场景。左：Lenscope 从程序化场景生成混合相机视图，检查共视关系，提供精确射线、度量深度与位姿监督；黄色标记为选中的全景中心。右：Stanford 2D3DS 与激光扫描全景构造的混合元组重建。MEOW 仅接收图像，Wid3R 额外接收相机类型标签。可视化中各点云分别与真值做相似变换对齐；VGGT 面板放大 2 倍。</sub>

## 主要特点

- **混合相机，一次前馈。** 同一个元组可混合透视、鱼眼与完整全景；一次联合前馈输出所有视图的度量点图和位姿。
- **真实场景零样本迁移。** 仅用合成元组训练，在 Stanford 2D3DS 异构元组上达到 80.4 mAA@30；逐视图提供相机类型的 Wid3R 为 54.3。
- **激光扫描基准。** 包含多房间办公室中 12 个已配准的 Leica BLK360 G2 站点、四条评测轨。混合轨达到 79.4 AUC@30，Wid3R 为 29.3。
- **Lenscope 数据引擎。** 程序化场景、精确逐像素射线与深度，以及采样相机后重新检查共视关系的在线采样器。

## 方法

<p align="center"><img src="assets/method.jpg" alt="混合相机联合前馈架构" width="100%"></p>

<sub>不同相机视图在保留全部视场的条件下缩放到统一张量形状。每张源图的长宽比（检测出的完整全景固定为 2:1）通过 MLP 加到 patch token。16 层交替全局/帧内注意力、逐视图稠密头（全景使用环绕填充）、位姿头与尺度头，在同一世界坐标下输出射线、深度、OpenCV 轴约定的 camera-to-world 位姿及度量尺度。图中输入为零样本真实元组。</sub>

<p align="center"><img src="assets/lenscope.jpg" alt="Lenscope 离线缓存与在线相机采样" width="100%"></p>

<sub>离线阶段每场景执行一次：由种子生成场景，在自由空间放置全景位姿，渲染精确射线与深度，并以双向可见性计算共视关系（连线：c<sub>kl</sub> ≥ 0.40）。在线阶段每元组执行：连通游走选取 2–8 个位姿，各自重采样为随机相机；重新计算共视关系（边宽表示强度，低于 0.25 为虚线），仅保留连通元组。</sub>

## 结果

Stanford 2D3DS 异构评测：88 个元组，每组 3–24 个视图，一次前馈。位姿指标先在每个元组内计算，再对 88 个元组取平均；ATE 在 Sim(3) 对齐后计算。

| 方法 | 相机侧输入 | RRA@30 | RTA@30 | mAA@30 | ATE ↓ |
|---|---|:-:|:-:|:-:|:-:|
| VGGT | 无 | 32.6 | 47.3 | 10.5 | 1.65 |
| π³ | 无 | 45.9 | 59.7 | 19.8 | 1.24 |
| MapAnything | 无 | 52.8 | 53.9 | 16.4 | 1.48 |
| CAM3R† | 无 | – | – | – | – |
| **MEOW（本文）** | 无 | 95.2 | **96.2** | **80.4** | **0.62** |
| Wid3R | 逐视图类型 | **96.9** | 86.5 | 54.3 | 0.83 |

† 未发布多视图权重。

激光扫描基准：每条轨道 24 个四视图元组。前六个指标列为混合轨（全景、鱼眼、两张针孔图），最后三列为单类型轨的 AUC@30。AUC@30 对元组取平均；Acc、Comp 在每个元组一次对齐后计算，单位米；NC 为法向一致性。

| 方法 | 相机侧输入 | RRA@30 | RTA@30 | AUC@30 | Acc ↓ | Comp ↓ | NC | 全景 | 针孔 | 鱼眼 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DUSt3R | 无 | 62.5 | 67.4 | 32.9 | 0.164 | 0.863 | 0.758 | 0.5 | 96.0 | 73.8 |
| MASt3R | 无 | 66.7 | 66.0 | 31.8 | 0.182 | 0.508 | 0.737 | 2.7 | 96.2 | 56.2 |
| VGGT | 无 | 45.8 | 63.9 | 19.0 | 0.279 | 0.591 | 0.665 | 1.9 | 97.1 | 55.0 |
| π³ | 无 | 66.0 | 63.9 | 26.6 | 0.240 | 0.693 | 0.736 | 1.6 | **97.9** | 75.1 |
| MapAnything | 无 | 66.0 | 57.3 | 24.5 | 0.255 | 0.830 | 0.653 | 1.5 | 86.8 | 75.1 |
| **MEOW（本文）** | 无 | **100** | **100** | **79.4** | **0.142** | **0.367** | **0.809** | 71.0 | 85.7 | 73.2 |
| Wid3R | 逐视图类型 | 97.2 | 58.0 | 29.3 | 0.199 | 0.467 | 0.764 | 81.9 | 90.5 | **87.3** |
| PanoVGGT | 全景 | – | – | – | – | – | – | **90.0** | – | – |

<p align="center"><img src="assets/laser_benchmark.jpg" alt="激光混合元组：输入、激光真值、MEOW 和 Wid3R" width="100%"></p>

<sub>零样本激光扫描混合元组。每行依次为四张输入、激光真值、MEOW（仅像素）与 Wid3R（给定相机类型）的融合点图，使用评测器对齐；俯视地板至墙面切片按高度着色，三角形为站点或预测相机。</sub>

各项数值的缩放方式、全景标志等设置见 [评测说明](docs/EVALUATION.md)。

## 激光扫描基准

<p align="center"><img src="assets/laser_stations.jpg" alt="12 个注册扫描站点与一个混合元组" width="72%"></p>

<sub>(a) 12 个站点合并点云，编号 0–11，覆盖走廊与会议室。高亮站点 5、4、7、6 构成一个混合元组，连线共视值为 0.84、0.68、0.87。(b) 对应四张输入：完整全景、鱼眼及两张针孔视图，均从站点扫描重采样。</sub>

基准以独立压缩包按 CC BY 4.0 发布：[meow_laser_benchmark.tar.gz](https://github.com/qgli/MEOW-het/releases/download/v1.0/meow_laser_benchmark.tar.gz)（636 MB，SHA-256：`5279045d2c92d4290d83564422f91054cbdad6ca1930257575187feb17d36aa3`）。轨道、筛选条件和构建方式见 [基准说明](docs/BENCHMARKS.md)。

## 发布内容

| 内容 | 说明 |
|---|---|
| 数据引擎 | 论文中的 Lenscope：两代 Blender 场景生成器、精确射线与深度、离线共视、训练时在线相机采样器，以及激光扫描转同格式工具（[说明](docs/DATA_ENGINE.md)） |
| 训练 | 三阶段 Hydra 配置，2×RTX 5090、4/8×H200、8×A100 40 GB 硬件配置，启动脚本与第三阶段接力（[说明](docs/TRAINING.md)） |
| 评测 | 基准构建与坐标约定测试、各数据集评测器与设置、基线驱动（[评测](docs/EVALUATION.md)、[基准](docs/BENCHMARKS.md)） |
| 模型权重 | 最终 sB25 托管于 [Hugging Face](https://huggingface.co/qgli/MEOW-4D-Space-Reconstruction)，需人工批准；见下文 |
| 可复现性 | 本代码执行的检查（[说明](docs/REPRODUCIBILITY.md)） |

两个父检查点及插值脚本暂未发布。训练数据不随库分发：两代场景均可通过引擎按种子重新生成；所需 Blender 版本与资产库见 [数据引擎说明](docs/DATA_ENGINE.md)。

## 模型权重

最终推理检查点：[qgli/MEOW-4D-Space-Reconstruction](https://huggingface.co/qgli/MEOW-4D-Space-Reconstruction)。请按模型页面说明申请；提交申请不代表自动获得下载权限。

请邮件联系 [li.qiaoge@image.iit.tsukuba.ac.jp](mailto:li.qiaoge@image.iit.tsukuba.ac.jp)，提供 Hugging Face 用户名、单位和用途，并用同一账号在模型页面提交申请。作者人工批准后方可下载。

| 检查点 | 格式 | 大小 | 评测变体 |
|---|---|---|---|
| `MEOW-sB25.pth` | PyTorch，完整 FP32 模型参数 | 5.05 GB | `:wrap` |

这是最终插值后的推理模型，不是中间训练检查点；包含完整模型参数，不含优化器状态，也不是量化模型。

获批后使用 Hugging Face CLI 下载：

```bash
pip install -U huggingface_hub
hf auth login
hf download qgli/MEOW-4D-Space-Reconstruction MEOW-sB25.pth --local-dir checkpoints
export CKPT="$PWD/checkpoints/MEOW-sB25.pth"
```

现有评测脚本使用 `MEOW=$CKPT:wrap`。混合相机输入应保留完整视场，并从图像自动判断全景：

```bash
python scripts/realset/predict_ours.py \
    --tuples /path/to/tuples.json --frames-root /path/to/frames \
    --ckpt "MEOW=$CKPT:wrap" --input-resize squeeze --pano auto --out preds
```

各数据集命令见 [评测说明](docs/EVALUATION.md)。

## 快速开始

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/third_party/uniception
python scripts/convert_hf_to_benchmark_checkpoint.py --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

生成并渲染一个第一代场景，计算共视，在线采样一个训练元组，并进行 10 个优化步以检查完整流水线：

```bash
export WORK=$PWD/work BLENDER=/path/to/blender-4.5/blender
"$BLENDER" -b --python lenscope/gen1/generate_scene.py -- --seed 3 \
    --output_dir "$WORK/gen1/scenes" --room_type random --n_scenes 1
python lenscope/gen1/make_pick.py --blend-dir "$WORK/gen1/scenes" \
    --scenes proc_scene_000003 --out "$WORK/gen1/pick.json"
"$BLENDER" -b --python lenscope/gen1/render_fair_focal.py -- \
    --pick "$WORK/gen1/pick.json" --out "$WORK/gen1/packs" --num_poses 8 --samples 64
python scripts/compute_covisibility.py --root "$WORK/gen1/packs" --scenes proc_scene_000003 --overwrite
python scripts/repro/make_smoke_splits.py --scene proc_scene_000003 --out-dir "$WORK/gen1/splits"
python scripts/repro/smoke_injector_tuple.py --repo . --root "$WORK/gen1/packs" \
    --splits "$WORK/gen1/splits" --out "$WORK/gen1/tuple.json"
ENCODER_LR=0 scripts/repro/run_train_10steps.sh gen1 "$WORK/gen1/packs" "$WORK/gen1/splits" \
    checkpoints/facebook_map-anything-apache.pth "$WORK"      # 24 GB GPU 使用 ENCODER_LR=0
```

## 目录结构

```text
lenscope/           数据引擎：gen1/ 第一代；genesis/、core/、bpy/、fixtures/ 第二代；blk/ 激光扫描；tests/
mapanything/        加入 MEOW 修改的 MapAnything：模型、损失、训练器、数据集、在线相机采样器
third_party/        包含变分辨率修改的 UniCeption 0.1.7
configs/            Hydra 配置；meow_stage/、meow_hardware/ 为训练与硬件配方
scripts/            训练、数据准备、评测、基准构建、竞品驱动与流水线检查
resources/          两代数据划分、第一代帧统计与训练加载器所用可行性表
benchmarks/         评测集定义（帧列表与元组，不含图像）
docs/               安装、数据引擎、训练、评测、基准与可复现性文档（英文）
assets/             本页使用的图片与动画
```

## 引用

```bibtex
@article{li2026meow,
  title   = {Many Eyes, One World: Feed-Forward 3D Reconstruction from Mixed Cameras},
  author  = {Li, Qiaoge and Zhan, Yifan and Yang, Haijun and Liu, Haiyang and Cai, Yiyi and Luo, Chenchi},
  journal = {arXiv preprint arXiv:2609.35658},
  year    = {2026}
}
```

## 许可证

代码按 [Apache License 2.0](LICENSE) 发布，基于 MapAnything（Apache 2.0）与 UniCeption（BSD 3-Clause）。相关许可及运行时数据、资产说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

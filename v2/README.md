# v2 裂缝语义分割

这是基于 PyTorch 的裂缝语义分割训练、验证和推理流程，默认使用
`data/data-202604` 数据集。

## 数据结构

```text
data/data-202604/
  images/
    0001.png
  masks/
    0001-0.png
    0001-1.png
```

`images` 中存放原图，`masks` 中存放标注 mask。同一张图片可以对应多个 mask，
例如 `0001-0.png`、`0001-1.png`。训练和验证时，这些同 stem 的 mask 会自动合并
成一个二分类前景目标：背景为 `0`，裂缝区域为 `1`。

## 安装依赖

```bash
pip install -r v2/requirements.txt
```

如果要使用额外模型族，可以再安装可选依赖：

```bash
pip install segmentation-models-pytorch transformers
```

## 训练

```bash
python v2/train.py \
  --image-dir data/data-202604/images \
  --mask-dir data/data-202604/masks \
  --model unet \
  --batch-size 2 \
  --epochs 30 \
  --lr 1e-4
```

训练结果会写入项目根目录下的 `runs/{timestamp}`：

- `config.json`：本次训练参数
- `metrics.csv`：每轮训练/验证指标
- `report.json`：实验报告汇总
- `training_curves.png`：训练曲线
- `weights/best.pt`：验证 IoU 最好的模型权重
- `weights/last.pt`：最后一轮模型权重
- `samples/*_overlay.png`：示例图推理叠加结果
- `samples/*_prob.png`：示例图前景概率图

查看支持的模型名称：

```bash
python v2/train.py --list-models
```

内置模型示例：

```bash
python v2/train.py --model fcn_resnet50
python v2/train.py --model deeplabv3_resnet50
python v2/train.py --model lraspp_mobilenet_v3_large
```

可选扩展模型示例：

```bash
python v2/train.py --model smp:Unet:resnet50
python v2/train.py --model hf:nvidia/segformer-b0-finetuned-ade-512-512
```

说明：

- `smp:Unet:resnet50` 需要安装 `segmentation-models-pytorch`。
- `hf:<model_id>` 需要安装 `transformers`，例如可接入 Hugging Face 上的 SegFormer。
- SAM / Segment Anything 本身不是这个训练流程里的监督式语义分割训练架构，更适合
  作为 proposal、prompt 模型，或作为推理阶段的前后处理模块。如果要接入，可以在
  `v2/models.py` 中新增 wrapper。
- `--pretrained` 默认开启。首次使用 torchvision 预训练模型时，如果本地没有权重缓存，
  可能需要联网下载；离线环境可以加 `--no-pretrained`。

## 验证

```bash
python v2/validate.py \
  --weights runs/20260418-120000/weights/best.pt \
  --image-dir data/data-202604/images \
  --mask-dir data/data-202604/masks
```

验证指标会保存到 `output/{timestamp}/validation_metrics.json`，包含：

- `loss`
- `iou`
- `dice`
- `pixel_accuracy`
- `map`
- `num_samples`

## 推理

```bash
python v2/predict.py \
  --weights runs/20260418-120000/weights/best.pt \
  --input data/data-202604/images \
  --save-mask \
  --save-prob
```

推理结果会保存到 `output/{timestamp}`：

- `overlays/`：浅色实例区域叠加图，主要查看结果
- `masks/`：二值 mask，使用 `--save-mask` 时生成
- `probabilities/`：前景概率图，使用 `--save-prob` 时生成
- `manifest.json`：本次推理输入、输出路径和参数记录

## 常用参数

训练脚本常用参数：

```bash
python v2/train.py --help
```

关键参数包括：

- `--image-dir`：训练图片路径，默认 `data/data-202604/images`
- `--mask-dir`：训练 mask 路径，默认 `data/data-202604/masks`
- `--model`：模型名称，默认 `unet`
- `--batch-size`：batch size
- `--epochs`：训练轮数
- `--lr`：学习率
- `--image-size`：训练输入尺寸，默认 `512`
- `--threshold`：推理时前景阈值，默认 `0.5`
- `--device`：运行设备，默认 `auto`，会优先使用 CUDA，其次 MPS，最后 CPU

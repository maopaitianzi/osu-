# 谱面生成器模型模块

本模块包含了用于谱面生成的深度学习模型和相关工具。

## 模块组成

- `transformer.py` - 基于Transformer的谱面生成模型
- `training.py` - 模型训练器和数据集处理
- `evaluation.py` - 模型评估和谱面生成工具
- `config.py` - 模型和训练配置参数

## 模型架构

模型采用了Transformer的编码器-解码器架构，结构如下：

1. **音频编码器**：将音频特征（如节拍、强度、频谱特征等）编码为隐藏表示
2. **谱面解码器**：自回归地生成谱面物件序列（坐标、时间和类型）

![模型架构](https://example.com/model_architecture.png)

### 主要特点

- 基于**Transformer**架构，充分利用注意力机制捕获音频特征与谱面物件之间的关系
- 支持**自回归生成**，根据已生成的物件预测下一个物件
- 采用**编码器-解码器**结构，允许音频特征和谱面特征之间的双向信息流动
- 使用**位置编码**来保持序列中的位置信息
- 支持**批量处理**，提高训练和推理效率

## 使用方法

### 模型训练

```python
from models.transformer import TransformerModel
from models.training import BeatmapDataset, Trainer
from models.config import NORMAL_CONFIG, DEFAULT_TRAINING_CONFIG
import torch
from torch.utils.data import DataLoader

# 创建模型
model = TransformerModel(
    input_dim=NORMAL_CONFIG.input_dim,
    d_model=NORMAL_CONFIG.d_model,
    output_dim=NORMAL_CONFIG.output_dim,
    nhead=NORMAL_CONFIG.nhead,
    num_encoder_layers=NORMAL_CONFIG.num_encoder_layers,
    num_decoder_layers=NORMAL_CONFIG.num_decoder_layers,
    dim_feedforward=NORMAL_CONFIG.dim_feedforward,
    dropout=NORMAL_CONFIG.dropout
)

# 创建数据集
train_dataset = BeatmapDataset(
    data_path="path/to/train/dataset.json",
    max_audio_seq_len=NORMAL_CONFIG.max_audio_seq_len,
    max_beatmap_seq_len=NORMAL_CONFIG.max_beatmap_seq_len,
    audio_feature_keys=NORMAL_CONFIG.audio_feature_keys
)

val_dataset = BeatmapDataset(
    data_path="path/to/val/dataset.json",
    max_audio_seq_len=NORMAL_CONFIG.max_audio_seq_len,
    max_beatmap_seq_len=NORMAL_CONFIG.max_beatmap_seq_len,
    audio_feature_keys=NORMAL_CONFIG.audio_feature_keys
)

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=DEFAULT_TRAINING_CONFIG.batch_size,
    shuffle=DEFAULT_TRAINING_CONFIG.shuffle_dataset,
    num_workers=DEFAULT_TRAINING_CONFIG.num_workers
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=DEFAULT_TRAINING_CONFIG.batch_size,
    shuffle=False,
    num_workers=DEFAULT_TRAINING_CONFIG.num_workers
)

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    lr=DEFAULT_TRAINING_CONFIG.lr,
    weight_decay=DEFAULT_TRAINING_CONFIG.weight_decay,
    device=DEFAULT_TRAINING_CONFIG.device,
    checkpoint_dir=DEFAULT_TRAINING_CONFIG.checkpoint_dir,
    log_dir=DEFAULT_TRAINING_CONFIG.log_dir,
    save_every=DEFAULT_TRAINING_CONFIG.save_every
)

# 训练模型
history = trainer.train(
    num_epochs=DEFAULT_TRAINING_CONFIG.num_epochs,
    progress_callback=lambda progress: print(f"进度: {progress}%"),
    epoch_callback=lambda epoch, train_loss, val_loss: print(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}"),
    log_callback=lambda message: print(message)
)

# 保存最终模型
trainer.save_checkpoint("final_model.pth")

# 绘制损失曲线
trainer.plot_losses(save_path="loss_curve.png")
```

### 谱面生成

```python
from models.transformer import TransformerModel
from models.evaluation import Evaluator
from models.config import NORMAL_CONFIG
import torch

# 加载模型
model = TransformerModel(
    input_dim=NORMAL_CONFIG.input_dim,
    d_model=NORMAL_CONFIG.d_model,
    output_dim=NORMAL_CONFIG.output_dim,
    nhead=NORMAL_CONFIG.nhead,
    num_encoder_layers=NORMAL_CONFIG.num_encoder_layers,
    num_decoder_layers=NORMAL_CONFIG.num_decoder_layers,
    dim_feedforward=NORMAL_CONFIG.dim_feedforward,
    dropout=NORMAL_CONFIG.dropout
)

# 加载模型权重
checkpoint = torch.load("path/to/model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# 创建评估器
evaluator = Evaluator(model=model, device="cuda")

# 加载音频特征
audio_features = torch.load("path/to/audio_features.pt")

# 生成谱面
beatmap_sequence = evaluator.generate_beatmap(
    audio_features=audio_features,
    max_length=500,
    temperature=1.0
)

# 可视化谱面
evaluator.visualize_beatmap(
    beatmap_sequence=beatmap_sequence,
    save_path="generated_beatmap.png"
)

# 转换为OSU格式
osu_content = evaluator.convert_to_osu_format(
    beatmap_sequence=beatmap_sequence,
    audio_path="path/to/audio.mp3",
    title="Song Title",
    artist="Artist Name",
    creator="AI Generator",
    version="Normal"
)

# 保存OSU文件
evaluator.save_osu_file(osu_content, "generated_beatmap.osu")
```

## 配置说明

模型配置参数存储在`config.py`中，提供了不同难度级别的预设配置：

- `EASY_CONFIG` - 适用于简单难度谱面生成的小型模型
- `NORMAL_CONFIG` - 适用于一般难度谱面生成的中型模型（默认）
- `HARD_CONFIG` - 适用于困难难度谱面生成的大型模型
- `EXPERT_CONFIG` - 适用于专家难度谱面生成的超大型模型

训练配置也提供了不同的预设：

- `DEFAULT_TRAINING_CONFIG` - 默认训练配置
- `FAST_TRAINING_CONFIG` - 快速训练配置，用于原型开发
- `FULL_TRAINING_CONFIG` - 完整训练配置，用于最终模型训练

## 数据集格式

训练数据集应该是JSON格式，包含谱面分析结果和对应的音频特征。详细格式请参考[数据集格式文档](../数据集格式.txt)。

## 模型保存格式

模型检查点包含以下内容：

```python
{
    "model_state_dict": model.state_dict(),  # 模型参数
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
    "scheduler_state_dict": scheduler.state_dict(),  # 学习率调度器状态
    "epoch": current_epoch,  # 当前训练轮次
    "best_val_loss": best_val_loss,  # 最佳验证损失
    "train_losses": train_losses,  # 训练损失历史
    "val_losses": val_losses  # 验证损失历史
}
```

## 依赖项

- PyTorch >= 1.9.0
- NumPy
- Matplotlib
- tqdm 
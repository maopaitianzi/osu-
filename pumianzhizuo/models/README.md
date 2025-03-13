# osu!风格谱面生成器 - 模型实现计划

## 概述

本文档描述了osu!风格谱面生成器中深度学习模型的实现计划，包括模型架构设计、训练流程和评估方法。模型将基于Transformer架构，能够学习音频特征与谱面设计之间的关系，生成符合节奏和游戏规则的谱面。

## 模型架构设计

### 核心架构：Transformer序列生成模型

我们将实现一个基于Transformer的序列到序列模型，用于将音频特征序列映射到谱面元素序列。核心架构包括：

1. **特征编码器**：将音频特征（频谱、节拍、能量等）编码为模型可处理的表示
2. **Transformer编码器**：处理时序特征，捕捉长期依赖
3. **Transformer解码器**：生成谱面元素序列
4. **谱面解码器**：将模型输出解码为具体的谱面元素（物件类型、位置、时间等）

```
                       ┌─────────────────┐
                       │                 │
音频特征 ───────────────┤  特征编码器     │
                       │                 │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │                 │
                       │ Transformer     │
                       │ 编码器          │
                       │                 │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │                 │
                       │ Transformer     │
                       │ 解码器          │
                       │                 │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │                 │
                       │  谱面解码器      │
                       │                 │
                       └────────┬────────┘
                                │
                                ▼
                             谱面元素
```

### 详细设计规范

#### 1. 特征编码器

- 输入：音频特征矩阵 (batch_size, seq_length, feature_dim)
- 输出：编码特征 (batch_size, seq_length, hidden_dim)
- 结构：多层前馈网络 + 位置编码
- 实现：
  ```python
  class FeatureEncoder(nn.Module):
      def __init__(self, feature_dim, hidden_dim, dropout=0.1):
          super().__init__()
          self.feature_projection = nn.Linear(feature_dim, hidden_dim)
          self.position_encoder = PositionalEncoding(hidden_dim, dropout)
          self.layer_norm = nn.LayerNorm(hidden_dim)
          
      def forward(self, x):
          # x: [batch_size, seq_length, feature_dim]
          x = self.feature_projection(x)
          x = self.position_encoder(x)
          return self.layer_norm(x)
  ```

#### 2. Transformer编码器

- 输入：编码特征 (batch_size, seq_length, hidden_dim)
- 输出：上下文特征 (batch_size, seq_length, hidden_dim)
- 结构：多头自注意力 + 前馈网络，4-6层
- 实现：使用PyTorch的`nn.TransformerEncoder`

#### 3. Transformer解码器

- 输入：上下文特征 + 已生成序列
- 输出：下一元素预测 (batch_size, seq_length, hidden_dim)
- 结构：掩码多头自注意力 + 交叉注意力 + 前馈网络，4-6层
- 实现：使用PyTorch的`nn.TransformerDecoder`

#### 4. 谱面解码器

- 输入：Transformer解码器输出
- 输出：谱面元素参数（类型概率、位置、时间等）
- 结构：多头预测网络，分别预测不同特征
- 实现：
  ```python
  class BeatmapDecoder(nn.Module):
      def __init__(self, hidden_dim, n_object_types, n_positions):
          super().__init__()
          self.object_type_head = nn.Linear(hidden_dim, n_object_types)
          self.position_head = nn.Linear(hidden_dim, n_positions)
          self.time_offset_head = nn.Linear(hidden_dim, 1)
          
      def forward(self, x):
          # x: [batch_size, seq_length, hidden_dim]
          object_type_logits = self.object_type_head(x)
          position_logits = self.position_head(x)
          time_offset = self.time_offset_head(x)
          
          return {
              "object_type": object_type_logits,
              "position": position_logits,
              "time_offset": time_offset
          }
  ```

### 整体模型结构

```python
class BeatmapTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dim=256,
        n_object_types=4,
        n_positions=100,
        n_encoder_layers=4,
        n_decoder_layers=4,
        nhead=8,
        dropout=0.1
    ):
        super().__init__()
        
        # 特征编码器
        self.feature_encoder = FeatureEncoder(feature_dim, hidden_dim, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_encoder_layers
        )
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=n_decoder_layers
        )
        
        # 谱面解码器
        self.beatmap_decoder = BeatmapDecoder(hidden_dim, n_object_types, n_positions)
        
        # 目标嵌入
        self.target_embedding = nn.Embedding(n_object_types, hidden_dim)
        self.position_embedding = nn.Embedding(n_positions, hidden_dim)
        self.time_embedding = nn.Linear(1, hidden_dim)
        
    def forward(self, audio_features, target_seq=None, target_mask=None):
        # audio_features: [batch_size, seq_length, feature_dim]
        
        # 编码音频特征
        memory = self.feature_encoder(audio_features)
        memory = self.transformer_encoder(memory)
        
        if self.training and target_seq is not None:
            # 训练模式：使用教师强制
            # 处理目标序列
            embedded_targets = self.process_targets(target_seq)
            
            # 解码
            output = self.transformer_decoder(
                embedded_targets, memory, tgt_mask=target_mask
            )
            
            # 预测输出
            return self.beatmap_decoder(output)
        else:
            # 推理模式：自回归生成
            return self.generate(memory, max_length=100)
            
    def process_targets(self, target_seq):
        # 将目标序列嵌入
        # target_seq包含[object_type, position, time]
        object_embeds = self.target_embedding(target_seq[:, :, 0].long())
        position_embeds = self.position_embedding(target_seq[:, :, 1].long())
        time_embeds = self.time_embedding(target_seq[:, :, 2:3])
        
        # 组合嵌入
        return object_embeds + position_embeds + time_embeds
        
    def generate(self, memory, max_length=100):
        # 自回归生成谱面序列
        # 实现推理时的自回归解码
        # ...（详细实现略）
```

## 损失函数设计

模型将使用多任务损失函数，包括：

1. **物件类型损失**：交叉熵损失，预测物件类型（圆圈、滑条、转盘等）
2. **位置损失**：交叉熵或MSE损失，预测物件位置坐标
3. **时间损失**：MSE损失，预测物件时间点与音乐节拍的关系
4. **节奏一致性损失**：自定义损失，确保生成的谱面与音乐节奏一致

总损失函数将是这些组件的加权和：

```python
def compute_loss(predictions, targets, weights):
    # 物件类型损失（分类问题）
    type_loss = F.cross_entropy(
        predictions["object_type"].view(-1, predictions["object_type"].size(-1)),
        targets[:, :, 0].long().view(-1)
    )
    
    # 位置损失（分类或回归问题）
    position_loss = F.cross_entropy(
        predictions["position"].view(-1, predictions["position"].size(-1)),
        targets[:, :, 1].long().view(-1)
    )
    
    # 时间损失（回归问题）
    time_loss = F.mse_loss(
        predictions["time_offset"].view(-1),
        targets[:, :, 2].view(-1)
    )
    
    # 节奏一致性损失（自定义）
    rhythm_loss = compute_rhythm_consistency_loss(predictions, targets)
    
    # 总损失
    total_loss = (
        weights["type"] * type_loss + 
        weights["position"] * position_loss + 
        weights["time"] * time_loss + 
        weights["rhythm"] * rhythm_loss
    )
    
    return total_loss, {
        "type_loss": type_loss.item(),
        "position_loss": position_loss.item(),
        "time_loss": time_loss.item(),
        "rhythm_loss": rhythm_loss.item(),
        "total_loss": total_loss.item()
    }
```

## 训练流程

### 训练参数

- **批次大小**：16-32（取决于可用GPU内存）
- **学习率**：0.0005，使用余弦退火调度
- **优化器**：Adam (β1=0.9, β2=0.999, ε=1e-8)
- **训练周期**：50-100轮
- **梯度裁剪**：最大范数 1.0
- **正则化**：权重衰减 1e-5，Dropout 0.1

### 训练配置

```python
training_config = {
    # 数据配置
    "train_data_path": "data/processed/train",
    "val_data_path": "data/processed/val",
    "batch_size": 16,
    "num_workers": 4,
    
    # 模型配置
    "feature_dim": 128,
    "hidden_dim": 256,
    "n_encoder_layers": 4,
    "n_decoder_layers": 4,
    "nhead": 8,
    "dropout": 0.1,
    
    # 优化器配置
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    
    # 训练配置
    "epochs": 50,
    "grad_clip": 1.0,
    "early_stopping_patience": 10,
    
    # 损失权重
    "loss_weights": {
        "type": 1.0,
        "position": 1.0,
        "time": 1.0,
        "rhythm": 0.5
    },
    
    # 杂项
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_mixed_precision": True,
    "log_interval": 10,
    "checkpoint_dir": "models/checkpoints"
}
```

### 训练循环

训练循环将包括：

1. 数据加载与预处理
2. 前向传播
3. 损失计算
4. 反向传播与优化
5. 模型评估
6. 检查点保存

## 评估方法

### 定量评估指标

1. **生成准确率**：生成的谱面物件与基准谱面的匹配程度
2. **节奏一致性**：谱面物件与音乐节拍的对齐程度
3. **难度准确性**：生成谱面的难度与目标难度的接近程度
4. **游戏规则兼容性**：生成谱面符合游戏规则的程度

### 定性评估

1. **可玩性评估**：由测试玩家评分
2. **谱面流畅性**：物件排布的自然度和连贯性
3. **音乐表现力**：谱面对音乐特点的表达程度

## 目录结构与文件规划

```
models/
├── README.md                   # 本文档
├── transformer.py              # Transformer模型实现
├── positional_encoding.py      # 位置编码实现
├── loss.py                     # 损失函数实现
├── training.py                 # 训练循环实现
├── evaluation.py               # 评估方法实现
├── inference.py                # 推理与生成实现
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── data.py                 # 数据处理工具
│   ├── metrics.py              # 评估指标工具
│   └── visualization.py        # 可视化工具
├── checkpoints/                # 模型检查点保存目录
│   └── README.md               # 检查点说明
└── configs/                    # 配置文件目录
    ├── default.json            # 默认配置
    ├── small.json              # 小型模型配置
    └── large.json              # 大型模型配置
```

## 实施计划

1. **基础架构实现**（3天）
   - 实现基本的Transformer模型架构
   - 实现位置编码和嵌入层

2. **数据接口开发**（2天）
   - 实现数据加载和预处理接口
   - 创建训练样本生成器

3. **训练循环实现**（3天）
   - 实现完整训练循环
   - 添加评估和检查点功能

4. **损失函数设计**（2天）
   - 实现基本损失函数
   - 开发节奏一致性损失

5. **推理与生成**（3天）
   - 实现自回归解码
   - 开发谱面后处理功能

6. **评估与优化**（2天）
   - 实现评估指标
   - 进行初步模型调优

## 后续优化方向

1. **探索更复杂的编码器架构**，如CNN+Transformer混合模型
2. **实现注意力可视化**，增强模型可解释性
3. **添加流派感知特征**，针对不同音乐风格优化生成
4. **实现渐进式难度生成**，改进多难度谱面的设计
5. **探索强化学习优化**，使用玩家反馈优化生成模型

## 风险与挑战

1. **数据不足**：高质量谱面数据可能不足以训练复杂模型
2. **训练不稳定**：Transformer模型训练可能不稳定
3. **泛化能力**：模型在未见过的音乐类型上可能表现不佳
4. **计算资源**：完整训练可能需要大量GPU资源

## 结论

本文档提供了osu!风格谱面生成器中深度学习模型的详细实现计划。基于Transformer架构的模型设计旨在捕捉音频特征与谱面设计之间的复杂关系，生成高质量、符合游戏规则的谱面。实施计划分步骤推进，确保系统化开发和全面测试。 
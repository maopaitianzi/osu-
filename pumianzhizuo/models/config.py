#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模块 - 存储模型和训练的所有配置参数
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any


@dataclass
class ModelConfig:
    """模型配置参数"""
    
    # 模型架构参数
    input_dim: int = 6  # 音频特征维度
    d_model: int = 512  # Transformer隐藏维度
    output_dim: int = 4  # 谱面特征维度 (x, y, time, type)
    nhead: int = 8  # 多头注意力的头数
    num_encoder_layers: int = 6  # 编码器层数
    num_decoder_layers: int = 6  # 解码器层数
    dim_feedforward: int = 2048  # 前馈神经网络维度
    dropout: float = 0.1  # Dropout率
    
    # 序列长度限制
    max_audio_seq_len: int = 1000  # 音频特征序列最大长度
    max_beatmap_seq_len: int = 500  # 谱面特征序列最大长度
    
    # 音频特征配置
    audio_feature_keys: List[str] = field(default_factory=lambda: [
        "beat_times", "beat_strengths", "spectral_centroids", 
        "mfccs", "onsets", "transitions"
    ])
    
    # 生成参数
    generation_temperature: float = 1.0  # 生成温度
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_audio_seq_len": self.max_audio_seq_len,
            "max_beatmap_seq_len": self.max_beatmap_seq_len,
            "audio_feature_keys": self.audio_feature_keys,
            "generation_temperature": self.generation_temperature
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """训练配置参数"""
    
    # 基本训练参数
    batch_size: int = 32  # 批次大小
    num_epochs: int = 100  # 总训练轮次
    lr: float = 1e-4  # 学习率
    weight_decay: float = 1e-5  # 权重衰减
    clip_grad_norm: float = 1.0  # 梯度裁剪阈值
    
    # 学习率调度器参数
    scheduler_patience: int = 5  # 学习率调度器的耐心值
    scheduler_factor: float = 0.5  # 学习率调度器的衰减因子
    
    # 数据集参数
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # 训练/验证/测试集比例
    shuffle_dataset: bool = True  # 是否打乱数据集
    
    # 保存和日志参数
    checkpoint_dir: str = "./checkpoints"  # 检查点保存目录
    log_dir: str = "./logs"  # 日志保存目录
    save_every: int = 5  # 每多少个epoch保存一次检查点
    
    # 硬件配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
    num_workers: int = 4  # 数据加载器的工作线程数
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "clip_grad_norm": self.clip_grad_norm,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "train_val_test_split": self.train_val_test_split,
            "shuffle_dataset": self.shuffle_dataset,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "save_every": self.save_every,
            "device": self.device,
            "num_workers": self.num_workers
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        return cls(**config_dict)


def save_config(config: Union[ModelConfig, TrainingConfig], save_path: str) -> None:
    """
    保存配置到JSON文件
    
    参数:
        config: 要保存的配置对象
        save_path: 保存路径
    """
    import json
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存为JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(config_path: str, config_type: str = "model") -> Union[ModelConfig, TrainingConfig]:
    """
    从JSON文件加载配置
    
    参数:
        config_path: 配置文件路径
        config_type: 配置类型，"model"或"training"
        
    返回:
        加载的配置对象
    """
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    if config_type.lower() == "model":
        return ModelConfig.from_dict(config_dict)
    elif config_type.lower() == "training":
        return TrainingConfig.from_dict(config_dict)
    else:
        raise ValueError(f"未知的配置类型: {config_type}，应为'model'或'training'")


# 预定义的配置
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

# 针对不同难度的配置
EASY_CONFIG = ModelConfig(
    d_model=256,
    nhead=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1
)

NORMAL_CONFIG = ModelConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

HARD_CONFIG = ModelConfig(
    d_model=768,
    nhead=12,
    num_encoder_layers=8,
    num_decoder_layers=8,
    dim_feedforward=3072,
    dropout=0.1
)

EXPERT_CONFIG = ModelConfig(
    d_model=1024,
    nhead=16,
    num_encoder_layers=12,
    num_decoder_layers=12,
    dim_feedforward=4096,
    dropout=0.1
)

# 训练配置变体
FAST_TRAINING_CONFIG = TrainingConfig(
    batch_size=16,
    num_epochs=30,
    lr=3e-4,
    save_every=2
)

FULL_TRAINING_CONFIG = TrainingConfig(
    batch_size=32,
    num_epochs=200,
    lr=1e-4,
    save_every=10
)

# 不同应用场景的配置字典
MODEL_CONFIGS = {
    "default": DEFAULT_MODEL_CONFIG,
    "easy": EASY_CONFIG,
    "normal": NORMAL_CONFIG,
    "hard": HARD_CONFIG,
    "expert": EXPERT_CONFIG
}

TRAINING_CONFIGS = {
    "default": DEFAULT_TRAINING_CONFIG,
    "fast": FAST_TRAINING_CONFIG,
    "full": FULL_TRAINING_CONFIG
} 
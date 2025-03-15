#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练模块 - 用于训练谱面生成模型
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt

from .transformer import TransformerModel


class BeatmapDataset(Dataset):
    """
    谱面数据集
    
    用于加载和处理训练数据
    """
    
    def __init__(self, 
                 data_path: str, 
                 max_audio_seq_len: int = 1000,
                 max_beatmap_seq_len: int = 500,
                 audio_feature_keys: List[str] = None,
                 transform: Optional[Callable] = None):
        """
        初始化数据集
        
        参数:
            data_path: 数据集路径 (JSON文件)
            max_audio_seq_len: 音频序列的最大长度
            max_beatmap_seq_len: 谱面序列的最大长度
            audio_feature_keys: 要使用的音频特征键列表
            transform: 数据增强/变换函数
        """
        super().__init__()
        self.data_path = data_path
        self.max_audio_seq_len = max_audio_seq_len
        self.max_beatmap_seq_len = max_beatmap_seq_len
        self.transform = transform
        
        # 默认使用的音频特征
        self.audio_feature_keys = audio_feature_keys or [
            "beat_times", "beat_strengths", "spectral_centroids", 
            "mfccs", "onsets", "transitions"
        ]
        
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """
        加载数据集
        
        返回:
            数据集列表
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 过滤掉没有音频特征的样本
            filtered_data = []
            for item in data:
                if "analysis" in item and "audio_features" in item["analysis"]:
                    filtered_data.append(item)
            
            return filtered_data
        except Exception as e:
            raise RuntimeError(f"加载数据集失败: {str(e)}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            (audio_features, beatmap_features, audio_mask, beatmap_mask)
        """
        item = self.data[idx]
        audio_features = item["analysis"].get("audio_features", {})
        beatmap_data = item["analysis"]
        
        # 提取音频特征
        audio_feature_list = []
        audio_seq_len = 0
        
        for key in self.audio_feature_keys:
            if key in audio_features and isinstance(audio_features[key], list):
                # 确保不超过最大长度
                feature = audio_features[key][:self.max_audio_seq_len]
                audio_feature_list.append(feature)
                audio_seq_len = max(audio_seq_len, len(feature))
        
        # 提取谱面物件特征
        beatmap_objs = []
        if "heatmap" in beatmap_data:
            x_coords = beatmap_data["heatmap"].get("x", [])
            y_coords = beatmap_data["heatmap"].get("y", [])
            
            # 获取时间信息 (根据索引排序)
            hit_objects = sorted(beatmap_data.get("HitObjects", []), key=lambda obj: obj.get("time", 0))
            times = [obj.get("time", 0) for obj in hit_objects]
            types = [obj.get("type", 0) for obj in hit_objects]
            
            # 确保不超过最大长度
            max_len = min(len(x_coords), len(y_coords), len(times), len(types), self.max_beatmap_seq_len)
            
            for i in range(max_len):
                # 创建一个表示物件的特征向量 [x, y, time, type]
                obj_feature = [
                    x_coords[i] / 512.0,  # 归一化坐标
                    y_coords[i] / 384.0,  # 归一化坐标
                    times[i] / 60000.0,    # 归一化时间 (假设最大约为1分钟)
                    types[i] / 10.0        # 归一化类型
                ]
                beatmap_objs.append(obj_feature)
        
        # 将特征转换为张量
        # 对于音频特征，形状为 [序列长度, 特征数]
        audio_tensor = torch.zeros(self.max_audio_seq_len, len(self.audio_feature_keys))
        for i, feature in enumerate(audio_feature_list):
            feature_len = min(len(feature), self.max_audio_seq_len)
            audio_tensor[:feature_len, i] = torch.tensor(feature[:feature_len], dtype=torch.float32)
        
        # 对于谱面特征，形状为 [序列长度, 特征数(4)]
        beatmap_tensor = torch.zeros(self.max_beatmap_seq_len, 4)
        beatmap_len = min(len(beatmap_objs), self.max_beatmap_seq_len)
        if beatmap_len > 0:
            beatmap_tensor[:beatmap_len] = torch.tensor(beatmap_objs, dtype=torch.float32)
        
        # 创建掩码
        audio_mask = torch.zeros(self.max_audio_seq_len, dtype=torch.bool)
        audio_mask[audio_seq_len:] = True  # True表示需要掩盖的位置
        
        beatmap_mask = torch.zeros(self.max_beatmap_seq_len, dtype=torch.bool)
        beatmap_mask[beatmap_len:] = True  # True表示需要掩盖的位置
        
        # 应用数据增强/变换
        if self.transform:
            audio_tensor, beatmap_tensor = self.transform(audio_tensor, beatmap_tensor)
        
        return audio_tensor, beatmap_tensor, audio_mask, beatmap_mask


class Trainer:
    """
    训练器类
    
    用于训练、评估和保存模型
    """
    
    def __init__(self, 
                 model: TransformerModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs",
                 save_every: int = 5):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器 (可选)
            lr: 学习率
            weight_decay: 权重衰减
            device: 训练设备
            checkpoint_dir: 检查点保存目录
            log_dir: 日志保存目录
            save_every: 每多少个epoch保存一次检查点
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # 初始化早停策略
        self.early_stopping = None  # 默认不使用早停
        
        # 初始化损失列表
        self.train_losses = []
        self.val_losses = []
        
        # 创建优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 创建损失函数
        self.criterion = nn.MSELoss()
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建TensorBoard日志器
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 训练控制
        self.pause_flag = False
        self.stop_flag = False
    
    def train(self, num_epochs=50, start_epoch=1, progress_callback=None, epoch_callback=None, log_callback=None, use_mixed_precision=False):
        """
        训练模型
        
        参数:
            num_epochs: 训练轮次总数
            start_epoch: 开始训练的轮次 (用于恢复训练)
            progress_callback: 进度回调函数，接收0-100的整数表示进度
            epoch_callback: 每个轮次完成后的回调，接收当前轮次、训练损失和验证损失
            log_callback: 日志回调函数，接收日志消息字符串
            use_mixed_precision: 是否使用混合精度训练
        """
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 初始化训练状态
        self.is_running = True
        self.is_paused = False
        
        # 如果从后续轮次开始，确保损失列表已准备好
        if start_epoch > 1 and len(self.train_losses) < start_epoch - 1:
            # 填充损失列表
            if hasattr(self.model, 'temp_train_losses'):
                self.train_losses = self.model.temp_train_losses.copy()
                self.val_losses = self.model.temp_val_losses.copy() if hasattr(self.model, 'temp_val_losses') else []
            else:
                # 创建空列表，长度与开始轮次匹配
                self.train_losses = [0.0] * (start_epoch - 1)
                self.val_losses = [0.0] * (start_epoch - 1) if self.val_dataloader else []
        
        # 启用梯度缩放器(如果使用混合精度训练)
        if use_mixed_precision and self.device != 'cpu':
            if log_callback:
                log_callback("启用混合精度训练")
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        # 开始训练循环
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                if not self.is_running:
                    if log_callback:
                        log_callback("训练已停止")
                    break
                
                # 等待如果暂停
                while self.is_paused and self.is_running:
                    time.sleep(0.1)
                
                if log_callback:
                    log_callback(f"轮次 {epoch}/{num_epochs} 开始训练:")
                
                # 训练模式
                self.model.train()
                train_loss = 0
                num_batches = len(self.train_dataloader)
                
                # 迭代训练数据
                for batch_idx, batch_data in enumerate(self.train_dataloader):
                    if not self.is_running:
                        break
                    
                    # 等待如果暂停
                    while self.is_paused and self.is_running:
                        time.sleep(0.1)
                    
                    # 解包数据
                    audio_features, target_beatmap, audio_mask, beatmap_mask = batch_data
                    
                    # 将数据移动到设备
                    audio_features = audio_features.to(self.device)
                    target_beatmap = target_beatmap.to(self.device)
                    audio_mask = audio_mask.to(self.device)
                    beatmap_mask = beatmap_mask.to(self.device)
                    
                    # 清除梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    if scaler:  # 使用混合精度
                        with torch.cuda.amp.autocast():
                            # 创建目标掩码 (用于因果注意力)
                            tgt_mask = self.model._generate_square_subsequent_mask(target_beatmap.size(1)).to(self.device)
                            
                            outputs = self.model(
                                audio_features,
                                target_beatmap[:, :-1],
                                src_mask=audio_mask,
                                tgt_mask=tgt_mask[:-1, :-1],
                                memory_mask=None
                            )
                            loss = self.criterion(outputs, target_beatmap[:, 1:])
                    else:  # 正常精度
                        # 创建目标掩码 (用于因果注意力)
                        tgt_mask = self.model._generate_square_subsequent_mask(target_beatmap.size(1)).to(self.device)
                        
                        outputs = self.model(
                            audio_features,
                            target_beatmap[:, :-1],
                            src_mask=audio_mask,
                            tgt_mask=tgt_mask[:-1, :-1],
                            memory_mask=None
                        )
                        loss = self.criterion(outputs, target_beatmap[:, 1:])
                    
                    # 反向传播和优化
                    if scaler:  # 使用混合精度
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:  # 正常精度
                        loss.backward()
                        self.optimizer.step()
                    
                    # 累积损失
                    train_loss += loss.item()
                    
                    # 更新进度条
                    if progress_callback:
                        current_progress = int((((epoch - start_epoch) * num_batches + batch_idx) / 
                                              (num_epochs - start_epoch + 1) / num_batches) * 100)
                        progress_callback(min(current_progress, 99))  # 确保进度不超过99
                    
                    # 记录批次进度
                    if log_callback and batch_idx % max(1, num_batches // 10) == 0:
                        log_callback(f"  批次 {batch_idx+1}/{num_batches}, 损失: {loss.item():.6f}")
                
                # 计算平均训练损失
                avg_train_loss = train_loss / num_batches
                self.train_losses.append(avg_train_loss)
                
                # 验证
                avg_val_loss = None
                if self.val_dataloader:
                    avg_val_loss = self._validate()
                    self.val_losses.append(avg_val_loss)
                    
                    if log_callback:
                        log_callback(f"  验证损失: {avg_val_loss:.6f}")
                    
                    # 更新学习率
                    if self.scheduler:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(avg_val_loss)
                        else:
                            self.scheduler.step()
                
                # 记录轮次结果
                if log_callback:
                    val_msg = f", 验证损失: {avg_val_loss:.6f}" if avg_val_loss is not None else ""
                    log_callback(f"  轮次 {epoch} 完成, 训练损失: {avg_train_loss:.6f}{val_msg}")
                
                # 保存检查点
                if self.save_every > 0 and epoch % self.save_every == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                    if log_callback:
                        log_callback(f"  保存检查点: {checkpoint_path}")
                    self.save_checkpoint(checkpoint_path)
                
                # 触发轮次回调
                if epoch_callback:
                    epoch_callback(epoch, avg_train_loss, avg_val_loss)
                
                # 早停检查
                if hasattr(self, 'early_stopping') and self.early_stopping is not None and avg_val_loss is not None:
                    if self.early_stopping(avg_val_loss):
                        if log_callback:
                            log_callback(f"触发早停，在轮次 {epoch}")
                        break
            
            # 完成训练
            if progress_callback:
                progress_callback(100)
            
            # 保存最终模型
            final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_model.pth")
            self.save_checkpoint(final_checkpoint_path)
            
            return self.train_losses, self.val_losses
        
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            if log_callback:
                log_callback(f"训练过程出错: {str(e)}")
                log_callback(traceback_str)
            raise
    
    def _validate(self) -> float:
        """
        验证模型
        
        返回:
            验证损失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for audio_features, beatmap_features, audio_mask, beatmap_mask in self.val_dataloader:
                # 将数据移动到设备
                audio_features = audio_features.to(self.device)
                beatmap_features = beatmap_features.to(self.device)
                audio_mask = audio_mask.to(self.device)
                beatmap_mask = beatmap_mask.to(self.device)
                
                # 创建目标掩码 (用于因果注意力)
                tgt_mask = self.model._generate_square_subsequent_mask(beatmap_features.size(1)).to(self.device)
                
                # 前向传播
                outputs = self.model(
                    audio_features,
                    beatmap_features[:, :-1],
                    src_mask=audio_mask,
                    tgt_mask=tgt_mask[:-1, :-1]
                )
                
                # 计算损失
                loss = self.criterion(outputs, beatmap_features[:, 1:])
                
                # 累加损失
                total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def save_checkpoint(self, path: str) -> None:
        """
        保存检查点
        
        参数:
            path: 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        参数:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
    
    def pause(self) -> None:
        """暂停训练"""
        self.pause_flag = True
    
    def resume(self) -> None:
        """恢复训练"""
        self.pause_flag = False
    
    def stop(self) -> None:
        """停止训练"""
        self.stop_flag = True
    
    def plot_losses(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制损失曲线
        
        参数:
            save_path: 保存路径 (可选)
            
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax.plot(epochs, self.train_losses, label='训练损失', marker='o')
        if self.val_losses:
            ax.plot(epochs, self.val_losses, label='验证损失', marker='s')
        
        ax.set_title('训练和验证损失')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('损失')
        ax.legend()
        ax.grid(True)
        
        # 保存图像
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 
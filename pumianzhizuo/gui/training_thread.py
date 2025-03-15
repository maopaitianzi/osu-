#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练线程模块 - 实现模型训练的线程封装
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
import logging
import traceback
import shutil
from datetime import datetime

# 导入模型和训练组件
from models.transformer import TransformerModel
from models.training import BeatmapDataset, Trainer
from models.evaluation import Evaluator
from models.config import ModelConfig, TrainingConfig, EASY_CONFIG, NORMAL_CONFIG, HARD_CONFIG, EXPERT_CONFIG


class TrainingThread(QThread):
    """
    训练线程类
    
    用于在后台执行模型训练，避免阻塞GUI
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    log_message = pyqtSignal(str)  # 日志消息信号
    plot_updated = pyqtSignal(QPixmap)  # 绘图更新信号
    training_finished = pyqtSignal(bool, str)  # 训练完成信号(成功标志, 消息)
    epoch_completed = pyqtSignal(int, float, float)  # 轮次完成信号(轮次, 训练损失, 验证损失)
    
    def __init__(self, 
                 dataset_root: str, 
                 model_save_path: str,
                 model_type: str = "Transformer",
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 use_early_stopping: bool = True,
                 use_checkpoint: bool = True,
                 use_mixed_precision: bool = False,
                 model_config: ModelConfig = None,
                 training_config: TrainingConfig = None,
                 resume_from: str = None):
        """
        初始化训练线程
        
        参数:
            dataset_root: 数据集根目录
            model_save_path: 模型保存路径
            model_type: 模型类型
            batch_size: 批次大小
            learning_rate: 学习率
            epochs: 训练轮次
            use_gpu: 是否使用GPU
            gpu_device: GPU设备索引
            use_early_stopping: 是否使用早停
            use_checkpoint: 是否保存检查点
            use_mixed_precision: 是否使用混合精度训练
            model_config: 模型配置(可选)
            training_config: 训练配置(可选)
            resume_from: 要恢复的检查点路径(可选)
        """
        super().__init__()
        
        # 保存参数
        self.dataset_root = dataset_root
        self.model_save_path = model_save_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.use_early_stopping = use_early_stopping
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision
        self.resume_from = resume_from
        
        # 确定设备
        self.device = f"cuda:{gpu_device}" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # 训练控制标志
        self.is_paused = False
        self.is_stopped = False
        self.start_epoch = 1
        
        # 保存传入的配置或使用默认配置
        self.model_config = model_config or NORMAL_CONFIG
        
        # 如果没有提供训练配置，创建一个
        if training_config is None:
            self.training_config = TrainingConfig(
                batch_size=batch_size,
                num_epochs=epochs,
                lr=learning_rate,
                device=self.device,
                checkpoint_dir=os.path.join(model_save_path, "checkpoints"),
                log_dir=os.path.join(model_save_path, "logs"),
                save_every=5 if use_checkpoint else 0,  # 如果不使用检查点，设为0
            )
        else:
            self.training_config = training_config
        
        # 用于存储训练器和模型
        self.trainer = None
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        # 用于绘图
        self.fig = plt.figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """
        设置日志
        """
        # 创建日志目录
        log_dir = os.path.join(self.model_save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # 创建类日志器
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info("训练日志初始化完成")
    
    def run(self):
        """
        训练线程的主函数
        """
        try:
            # 日志开始信息
            self.log_message.emit("正在初始化训练环境...")
            self.status_updated.emit("初始化中")
            self.logger.info("开始初始化训练环境")
            
            # 检查目录是否存在
            if not os.path.exists(self.dataset_root):
                error_msg = f"数据集根目录不存在: {self.dataset_root}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            train_dir = os.path.join(self.dataset_root, "train")
            val_dir = os.path.join(self.dataset_root, "val")
            test_dir = os.path.join(self.dataset_root, "test")
            
            if not os.path.exists(train_dir):
                error_msg = f"训练集目录不存在: {train_dir}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 检查模型保存路径
            os.makedirs(self.model_save_path, exist_ok=True)
            os.makedirs(os.path.join(self.model_save_path, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.model_save_path, "logs"), exist_ok=True)
            
            # 打印设备信息
            device_info = f"使用设备: {self.device}"
            self.log_message.emit(device_info)
            self.logger.info(device_info)
            
            if self.use_gpu and torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(self.gpu_device)
                    gpu_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory / (1024**3)
                    gpu_info = f"GPU: {gpu_name} ({gpu_memory:.2f} GB)"
                    self.log_message.emit(gpu_info)
                    self.logger.info(gpu_info)
                except Exception as e:
                    self.logger.warning(f"获取GPU信息失败: {str(e)}")
                    self.log_message.emit(f"警告: 获取GPU信息失败: {str(e)}")
            
            # 创建或加载模型
            if self.resume_from:
                # 从检查点恢复训练
                self._resume_training()
            else:
                # 创建新模型
                self.log_message.emit(f"正在创建{self.model_type}模型...")
                self.logger.info(f"创建{self.model_type}模型")
                self.model = self._create_model()
            
            # 加载数据集
            self.log_message.emit("正在加载数据集...")
            self.logger.info("加载数据集")
            self._load_dataset()
            
            # 保存配置
            self._save_configs()
            
            # 创建训练器
            self.log_message.emit("正在设置训练器...")
            self.logger.info("设置训练器")
            self._create_trainer()
            
            # 开始训练
            self.log_message.emit(f"开始训练，总轮次: {self.epochs}，从第 {self.start_epoch} 轮开始")
            self.logger.info(f"开始训练，总轮次: {self.epochs}，从第 {self.start_epoch} 轮开始")
            self.status_updated.emit("训练中")
            
            # 训练模型，使用回调函数更新进度
            self.trainer.train(
                num_epochs=self.epochs,
                start_epoch=self.start_epoch,
                progress_callback=self._update_progress,
                epoch_callback=self._epoch_completed,
                log_callback=self._log_message,
                use_mixed_precision=self.use_mixed_precision
            )
            
            # 如果训练未被停止，则完成训练
            if not self.is_stopped:
                # 保存最终模型
                self._save_final_model()
                
                # 评估模型
                if self.test_dataloader:
                    self._evaluate_model()
                
                # 训练完成
                completion_msg = "训练成功完成"
                self.status_updated.emit("训练完成")
                self.progress_updated.emit(100)
                self.log_message.emit(completion_msg)
                self.logger.info(completion_msg)
                self.training_finished.emit(True, completion_msg)
            else:
                stop_msg = "训练已被用户停止"
                self.status_updated.emit("训练已停止")
                self.log_message.emit(stop_msg)
                self.logger.info(stop_msg)
                self.training_finished.emit(False, stop_msg)
        
        except Exception as e:
            # 详细记录异常
            tb_str = traceback.format_exc()
            error_msg = f"训练过程出错: {str(e)}"
            self.logger.error(f"{error_msg}\n{tb_str}")
            self.log_message.emit(error_msg)
            self.status_updated.emit("训练失败")
            self.training_finished.emit(False, error_msg)
    
    def _save_configs(self):
        """
        保存模型和训练配置
        """
        try:
            config_path = os.path.join(self.model_save_path, "model_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_config.to_dict(), f, indent=2)
            
            train_config_path = os.path.join(self.model_save_path, "training_config.json")
            with open(train_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_config.to_dict(), f, indent=2)
            
            self.log_message.emit(f"配置已保存到: {self.model_save_path}")
            self.logger.info(f"配置已保存到: {self.model_save_path}")
        except Exception as e:
            self.logger.warning(f"保存配置失败: {str(e)}")
            self.log_message.emit(f"警告: 保存配置失败: {str(e)}")
    
    def _create_trainer(self):
        """
        创建训练器
        """
        # 创建检查点目录
        checkpoint_dir = os.path.join(self.model_save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建日志目录
        log_dir = os.path.join(self.model_save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            lr=self.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            save_every=self.training_config.save_every
        )
        
        # 如果恢复训练，加载优化器和调度器状态
        if hasattr(self, 'optimizer_state') and self.optimizer_state:
            try:
                self.trainer.optimizer.load_state_dict(self.optimizer_state)
                self.log_message.emit("已加载优化器状态")
                self.logger.info("已加载优化器状态")
            except Exception as e:
                self.logger.warning(f"加载优化器状态失败: {str(e)}")
                self.log_message.emit(f"警告: 加载优化器状态失败: {str(e)}")
        
        if hasattr(self, 'scheduler_state') and self.scheduler_state and hasattr(self.trainer, 'scheduler'):
            try:
                self.trainer.scheduler.load_state_dict(self.scheduler_state)
                self.log_message.emit("已加载学习率调度器状态")
                self.logger.info("已加载学习率调度器状态")
            except Exception as e:
                self.logger.warning(f"加载学习率调度器状态失败: {str(e)}")
                self.log_message.emit(f"警告: 加载学习率调度器状态失败: {str(e)}")
    
    def _save_final_model(self):
        """
        保存最终模型和训练状态
        """
        try:
            # 保存最终模型
            final_model_path = os.path.join(self.model_save_path, "final_model.pth")
            final_state_path = os.path.join(self.model_save_path, "final_state.pth")
            
            # 保存模型权重
            torch.save(self.model.state_dict(), final_model_path)
            self.log_message.emit(f"最终模型已保存到: {final_model_path}")
            self.logger.info(f"最终模型已保存到: {final_model_path}")
            
            # 保存完整训练状态
            torch.save({
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'train_losses': self.trainer.train_losses,
                'val_losses': self.trainer.val_losses,
                'model_config': self.model_config.to_dict(),
                'training_config': self.training_config.to_dict(),
                'model_type': self.model_type,
                'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
            }, final_state_path)
            
            self.log_message.emit(f"完整训练状态已保存到: {final_state_path}")
            self.logger.info(f"完整训练状态已保存到: {final_state_path}")
            
            # 生成最终绘图
            self._update_plot()
        except Exception as e:
            error_msg = f"保存最终模型失败: {str(e)}"
            self.logger.error(error_msg)
            self.log_message.emit(f"错误: {error_msg}")
    
    def _evaluate_model(self):
        """
        评估模型性能
        """
        try:
            self.log_message.emit("正在评估模型...")
            self.status_updated.emit("评估中")
            self.logger.info("开始评估模型")
            
            evaluator = Evaluator(model=self.model, device=self.device)
            metrics = evaluator.evaluate_metrics(self.test_dataloader)
            
            # 打印评估结果
            self.log_message.emit("测试集评估结果:")
            self.logger.info("测试集评估结果:")
            for name, value in metrics.items():
                metric_msg = f"  {name}: {value:.6f}"
                self.log_message.emit(metric_msg)
                self.logger.info(metric_msg)
            
            # 保存评估结果
            metrics_path = os.path.join(self.model_save_path, "evaluation_metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            self.log_message.emit(f"评估结果已保存到: {metrics_path}")
            self.logger.info(f"评估结果已保存到: {metrics_path}")
        except Exception as e:
            error_msg = f"评估模型失败: {str(e)}"
            self.logger.error(error_msg)
            self.log_message.emit(f"错误: {error_msg}")
    
    def _resume_training(self):
        """
        从检查点恢复训练
        """
        try:
            self.log_message.emit(f"正在从检查点恢复训练: {self.resume_from}")
            self.logger.info(f"从检查点恢复训练: {self.resume_from}")
            
            # 检查检查点文件是否存在
            if not os.path.exists(self.resume_from):
                error_msg = f"检查点文件不存在: {self.resume_from}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 加载检查点
            checkpoint = torch.load(self.resume_from, map_location=self.device)
            
            # 获取模型类型和配置
            if 'model_type' in checkpoint:
                self.model_type = checkpoint['model_type']
                self.log_message.emit(f"加载模型类型: {self.model_type}")
            
            if 'model_config' in checkpoint:
                try:
                    # 尝试使用保存的模型配置
                    config_dict = checkpoint['model_config']
                    self.model_config = ModelConfig(**config_dict)
                    self.log_message.emit("已从检查点加载模型配置")
                    self.logger.info("已从检查点加载模型配置")
                except Exception as e:
                    self.logger.warning(f"无法从检查点加载模型配置: {str(e)}，使用当前配置代替")
                    self.log_message.emit("警告: 无法从检查点加载模型配置，使用当前配置代替")
            
            # 创建模型
            self.model = self._create_model()
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.log_message.emit("已加载模型权重")
                    self.logger.info("已加载模型权重")
                except Exception as e:
                    error_msg = f"加载模型权重失败: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = "检查点中没有模型权重"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 设置开始轮次
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                self.log_message.emit(f"将从第 {self.start_epoch} 轮继续训练")
                self.logger.info(f"将从第 {self.start_epoch} 轮继续训练")
            
            # 保存训练损失历史
            if 'train_losses' in checkpoint:
                # 创建临时对象存储损失历史
                self.temp_train_losses = checkpoint.get('train_losses', [])
                self.temp_val_losses = checkpoint.get('val_losses', [])
                self.log_message.emit(f"已加载训练历史 ({len(self.temp_train_losses)}个轮次)")
                self.logger.info(f"已加载训练历史 ({len(self.temp_train_losses)}个轮次)")
                
                # 创建训练损失图
                if len(self.temp_train_losses) > 0:
                    fig = plt.figure(figsize=(8, 5))
                    ax = fig.add_subplot(111)
                    
                    # 绘制训练损失
                    epochs = range(1, len(self.temp_train_losses) + 1)
                    ax.plot(epochs, self.temp_train_losses, 'o-', label='训练损失')
                    
                    # 绘制验证损失（如果有）
                    if self.temp_val_losses:
                        ax.plot(epochs, self.temp_val_losses, 's-', label='验证损失')
                    
                    # 设置图形属性
                    ax.set_title('之前训练的损失曲线')
                    ax.set_xlabel('轮次')
                    ax.set_ylabel('损失')
                    ax.legend()
                    ax.grid(True)
                    
                    # 转换为PyQt可用的QPixmap
                    canvas = FigureCanvasAgg(fig)
                    canvas.draw()
                    
                    buf = BytesIO()
                    canvas.print_png(buf)
                    buf.seek(0)
                    
                    qimage = QImage.fromData(buf.getvalue())
                    pixmap = QPixmap.fromImage(qimage)
                    
                    # 发送信号
                    self.plot_updated.emit(pixmap)
                    
                    plt.close(fig)
            
            # 保存优化器和调度器状态，在创建训练器后再加载
            self.optimizer_state = checkpoint.get('optimizer_state_dict', None)
            self.scheduler_state = checkpoint.get('scheduler_state_dict', None)
            
            # 如果有优化器状态，也打印日志
            if self.optimizer_state:
                self.log_message.emit("检查点包含优化器状态")
                self.logger.info("检查点包含优化器状态")
            
            # 加载训练配置
            if 'training_config' in checkpoint:
                try:
                    config_dict = checkpoint['training_config']
                    # 使用传入的一些关键参数覆盖检查点中的配置
                    config_dict['batch_size'] = self.batch_size  # 使用当前设置的批次大小
                    config_dict['lr'] = self.learning_rate  # 使用当前设置的学习率
                    
                    self.training_config = TrainingConfig(**config_dict)
                    self.log_message.emit("已从检查点加载训练配置")
                    self.logger.info("已从检查点加载训练配置")
                except Exception as e:
                    self.logger.warning(f"无法从检查点加载训练配置: {str(e)}，使用当前配置代替")
                    self.log_message.emit("警告: 无法从检查点加载训练配置，使用当前配置代替")
        
        except Exception as e:
            error_msg = f"恢复训练失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise RuntimeError(f"恢复训练失败: {str(e)}")
    
    def _create_model(self) -> nn.Module:
        """
        创建模型实例
        
        返回:
            模型实例
        """
        try:
            if self.model_type == "Transformer":
                return TransformerModel(
                    input_dim=self.model_config.input_dim,
                    d_model=self.model_config.d_model,
                    output_dim=self.model_config.output_dim,
                    nhead=self.model_config.nhead,
                    num_encoder_layers=self.model_config.num_encoder_layers,
                    num_decoder_layers=self.model_config.num_decoder_layers,
                    dim_feedforward=self.model_config.dim_feedforward,
                    dropout=self.model_config.dropout
                )
            elif self.model_type in ["LSTM", "GRU", "混合模型"]:
                # TODO: 实现其他模型类型
                self.log_message.emit(f"注意: {self.model_type}模型尚未实现，使用默认的Transformer模型")
                self.logger.warning(f"{self.model_type}模型尚未实现，使用默认的Transformer模型")
                return TransformerModel(
                    input_dim=self.model_config.input_dim,
                    d_model=self.model_config.d_model,
                    output_dim=self.model_config.output_dim,
                    nhead=self.model_config.nhead,
                    num_encoder_layers=self.model_config.num_encoder_layers,
                    num_decoder_layers=self.model_config.num_decoder_layers,
                    dim_feedforward=self.model_config.dim_feedforward,
                    dropout=self.model_config.dropout
                )
            else:
                error_msg = f"不支持的模型类型: {self.model_type}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"创建模型失败: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_dataset(self):
        """
        加载数据集并创建DataLoader
        """
        # 查找数据集JSON文件
        train_json = self._find_dataset_json(os.path.join(self.dataset_root, "train"))
        val_json = self._find_dataset_json(os.path.join(self.dataset_root, "val"))
        test_json = self._find_dataset_json(os.path.join(self.dataset_root, "test"))
        
        if not train_json:
            raise FileNotFoundError("在训练集目录中未找到数据集JSON文件")
        
        self.log_message.emit(f"加载训练集: {train_json}")
        
        # 创建数据集
        train_dataset = BeatmapDataset(
            data_path=train_json,
            max_audio_seq_len=self.model_config.max_audio_seq_len,
            max_beatmap_seq_len=self.model_config.max_beatmap_seq_len,
            audio_feature_keys=self.model_config.audio_feature_keys
        )
        
        self.log_message.emit(f"训练集样本数: {len(train_dataset)}")
        
        # 创建验证集（如果有）
        val_dataset = None
        if val_json:
            self.log_message.emit(f"加载验证集: {val_json}")
            val_dataset = BeatmapDataset(
                data_path=val_json,
                max_audio_seq_len=self.model_config.max_audio_seq_len,
                max_beatmap_seq_len=self.model_config.max_beatmap_seq_len,
                audio_feature_keys=self.model_config.audio_feature_keys
            )
            self.log_message.emit(f"验证集样本数: {len(val_dataset)}")
        
        # 创建测试集（如果有）
        test_dataset = None
        if test_json:
            self.log_message.emit(f"加载测试集: {test_json}")
            test_dataset = BeatmapDataset(
                data_path=test_json,
                max_audio_seq_len=self.model_config.max_audio_seq_len,
                max_beatmap_seq_len=self.model_config.max_beatmap_seq_len,
                audio_feature_keys=self.model_config.audio_feature_keys
            )
            self.log_message.emit(f"测试集样本数: {len(test_dataset)}")
        
        # 创建数据加载器
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.training_config.shuffle_dataset,
            num_workers=self.training_config.num_workers,
            pin_memory=self.use_gpu  # 如果使用GPU，使用pin_memory加速数据传输
        )
        
        if val_dataset:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers,
                pin_memory=self.use_gpu
            )
        
        if test_dataset:
            self.test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers,
                pin_memory=self.use_gpu
            )
    
    def _find_dataset_json(self, directory: str) -> str:
        """
        在目录中查找数据集JSON文件
        
        参数:
            directory: 要搜索的目录
            
        返回:
            找到的JSON文件路径，如果未找到则返回None
        """
        if not os.path.exists(directory):
            self.logger.warning(f"目录不存在: {directory}")
            return None
        
        # 首先查找dataset.json
        dataset_path = os.path.join(directory, "dataset.json")
        if os.path.exists(dataset_path):
            return dataset_path
        
        # 查找其他可能的文件名
        for filename in os.listdir(directory):
            if filename.endswith(".json") and "dataset" in filename.lower():
                return os.path.join(directory, filename)
        
        self.logger.warning(f"在目录中未找到数据集JSON文件: {directory}")
        return None
    
    def _update_progress(self, progress: int):
        """
        更新进度条
        
        参数:
            progress: 进度百分比(0-100)
        """
        self.progress_updated.emit(progress)
        
        # 处理暂停
        while self.is_paused and not self.is_stopped:
            time.sleep(0.1)
        
        # 处理停止
        if self.is_stopped:
            raise InterruptedError("训练已被用户停止")
    
    def _epoch_completed(self, epoch: int, train_loss: float, val_loss: float):
        """
        轮次完成回调
        
        参数:
            epoch: 当前轮次
            train_loss: 训练损失
            val_loss: 验证损失
        """
        self.epoch_completed.emit(epoch, train_loss, val_loss)
        
        # 更新绘图
        self._update_plot()
    
    def _log_message(self, message: str):
        """
        日志消息回调
        
        参数:
            message: 日志消息
        """
        self.log_message.emit(message)
    
    def _update_plot(self):
        """
        更新损失曲线图
        """
        if not self.trainer or not self.trainer.train_losses:
            return
        
        # 清除当前图形
        self.ax.clear()
        
        # 绘制训练损失
        epochs = range(1, len(self.trainer.train_losses) + 1)
        self.ax.plot(epochs, self.trainer.train_losses, 'o-', label='训练损失')
        
        # 绘制验证损失（如果有）
        if self.trainer.val_losses:
            self.ax.plot(epochs, self.trainer.val_losses, 's-', label='验证损失')
        
        # 设置图形属性
        self.ax.set_title('训练过程中的损失变化')
        self.ax.set_xlabel('轮次')
        self.ax.set_ylabel('损失')
        self.ax.legend()
        self.ax.grid(True)
        
        # 转换为PyQt可用的QPixmap
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        qimage = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        # 发送信号
        self.plot_updated.emit(pixmap)
    
    def pause_training(self):
        """
        暂停训练
        """
        if self.trainer:
            self.is_paused = True
            self.trainer.pause()
            self.status_updated.emit("已暂停")
            self.log_message.emit("训练已暂停")
    
    def resume_training(self):
        """
        恢复训练
        """
        if self.trainer:
            self.is_paused = False
            self.trainer.resume()
            self.status_updated.emit("训练中")
            self.log_message.emit("训练已恢复")
    
    def stop_training(self):
        """
        停止训练
        """
        if self.trainer:
            self.is_stopped = True
            self.is_paused = False
            self.trainer.stop()
            self.status_updated.emit("正在停止")
            self.log_message.emit("正在停止训练...")
    
    def is_training_paused(self) -> bool:
        """
        检查训练是否暂停
        
        返回:
            True如果训练已暂停，否则False
        """
        return self.is_paused

    def export_model(self, format: str = "pytorch") -> str:
        """
        导出训练好的模型
        
        参数:
            format: 导出格式，支持 "pytorch", "onnx", "torchscript"
            
        返回:
            导出文件的路径
        """
        if not self.model:
            error_msg = "没有可导出的模型"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            export_dir = os.path.join(self.model_save_path, "exported")
            os.makedirs(export_dir, exist_ok=True)
            
            self.log_message.emit(f"正在导出模型为{format}格式...")
            self.logger.info(f"导出模型为{format}格式")
            
            # 确保模型处于评估模式
            self.model.eval()
            
            if format.lower() == "pytorch":
                # 导出PyTorch模型
                export_path = os.path.join(export_dir, "model.pth")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "model_config": self.model_config.to_dict(),
                    "model_type": self.model_type,
                    "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(self.device)
                }, export_path)
                
                # 保存模型信息
                info_path = os.path.join(export_dir, "model_info.json")
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model_type": self.model_type,
                        "input_dim": self.model_config.input_dim,
                        "output_dim": self.model_config.output_dim,
                        "d_model": self.model_config.d_model,
                        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "format": "pytorch"
                    }, f, indent=2)
                
            elif format.lower() == "onnx":
                # 导出为ONNX格式
                export_path = os.path.join(export_dir, "model.onnx")
                
                try:
                    # 创建示例输入
                    dummy_input_audio = torch.randn(1, self.model_config.max_audio_seq_len, self.model_config.input_dim, device=self.device)
                    dummy_input_beatmap = torch.randn(1, 1, self.model_config.output_dim, device=self.device)
                    
                    # 导出模型
                    torch.onnx.export(
                        self.model,
                        (dummy_input_audio, dummy_input_beatmap),
                        export_path,
                        input_names=["audio_features", "initial_beatmap"],
                        output_names=["beatmap_output"],
                        dynamic_axes={
                            "audio_features": {0: "batch_size", 1: "audio_seq_len"},
                            "initial_beatmap": {0: "batch_size", 1: "beatmap_seq_len"},
                            "beatmap_output": {0: "batch_size", 1: "beatmap_seq_len"}
                        },
                        opset_version=12,
                        verbose=False
                    )
                    
                    # 保存模型信息
                    info_path = os.path.join(export_dir, "model_info.json")
                    with open(info_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "model_type": self.model_type,
                            "input_dim": self.model_config.input_dim,
                            "output_dim": self.model_config.output_dim,
                            "input_names": ["audio_features", "initial_beatmap"],
                            "output_names": ["beatmap_output"],
                            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "format": "onnx",
                            "opset_version": 12
                        }, f, indent=2)
                    
                except Exception as e:
                    error_msg = f"ONNX导出失败: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
            elif format.lower() == "torchscript":
                # 导出为TorchScript格式
                export_path = os.path.join(export_dir, "model.pt")
                
                try:
                    # 使用JIT跟踪
                    dummy_input_audio = torch.randn(1, self.model_config.max_audio_seq_len, self.model_config.input_dim, device=self.device)
                    dummy_input_beatmap = torch.randn(1, 1, self.model_config.output_dim, device=self.device)
                    
                    # 确保模型处于评估模式
                    self.model.eval()
                    
                    # 跟踪模型
                    with torch.no_grad():
                        traced_model = torch.jit.trace(self.model, (dummy_input_audio, dummy_input_beatmap))
                        
                    # 保存模型
                    traced_model.save(export_path)
                    
                    # 保存模型信息
                    info_path = os.path.join(export_dir, "model_info.json")
                    with open(info_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "model_type": self.model_type,
                            "input_dim": self.model_config.input_dim,
                            "output_dim": self.model_config.output_dim,
                            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "format": "torchscript"
                        }, f, indent=2)
                    
                except Exception as e:
                    error_msg = f"TorchScript导出失败: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"不支持的导出格式: {format}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            success_msg = f"模型已成功导出到: {export_path}"
            self.log_message.emit(success_msg)
            self.logger.info(success_msg)
            
            return export_path
            
        except Exception as e:
            error_msg = f"导出模型失败: {str(e)}"
            self.logger.error(error_msg)
            self.log_message.emit(f"错误: {error_msg}")
            raise RuntimeError(error_msg)

    def save_checkpoint(self, custom_name=None):
        """
        保存当前检查点
        
        参数:
            custom_name: 自定义检查点名称(可选)
        
        返回:
            保存的检查点路径
        """
        if not self.trainer or not self.model:
            self.log_message.emit("警告: 没有可保存的模型")
            self.logger.warning("没有可保存的模型")
            return None
        
        try:
            # 确定检查点名称
            checkpoint_dir = os.path.join(self.model_save_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if custom_name:
                checkpoint_name = f"{custom_name}.pth"
            else:
                current_epoch = len(self.trainer.train_losses)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_epoch_{current_epoch}_{timestamp}.pth"
            
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # 保存检查点
            torch.save({
                'epoch': len(self.trainer.train_losses),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'train_losses': self.trainer.train_losses,
                'val_losses': self.trainer.val_losses,
                'model_config': self.model_config.to_dict(),
                'training_config': self.training_config.to_dict(),
                'model_type': self.model_type,
                'scheduler_state_dict': self.trainer.scheduler.state_dict() if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler else None,
            }, checkpoint_path)
            
            self.log_message.emit(f"检查点已保存到: {checkpoint_path}")
            self.logger.info(f"检查点已保存到: {checkpoint_path}")
            
            return checkpoint_path
        
        except Exception as e:
            error_msg = f"保存检查点失败: {str(e)}"
            self.logger.error(error_msg)
            self.log_message.emit(f"错误: {error_msg}")
            return None

    def get_checkpoints(self):
        """
        获取所有可用的检查点
        
        返回:
            检查点文件列表 [(path, description), ...]
        """
        checkpoint_dir = os.path.join(self.model_save_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return []
        
        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith(".pth"):
                path = os.path.join(checkpoint_dir, filename)
                # 尝试提取轮次信息
                epoch_info = "未知轮次"
                try:
                    if "epoch_" in filename:
                        epoch_num = filename.split("epoch_")[1].split("_")[0]
                        epoch_info = f"轮次 {epoch_num}"
                except:
                    pass
                
                # 获取文件修改时间
                time_info = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                
                description = f"{filename} ({epoch_info}, {time_info})"
                checkpoints.append((path, description))
        
        # 按修改时间排序，最新的在前面
        return sorted(checkpoints, key=lambda x: os.path.getmtime(x[0]), reverse=True)

    def backup_checkpoint(self, checkpoint_path):
        """
        备份检查点
        
        参数:
            checkpoint_path: 要备份的检查点路径
            
        返回:
            备份路径
        """
        try:
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"要备份的检查点不存在: {checkpoint_path}")
                return None
            
            # 创建备份目录
            backup_dir = os.path.join(self.model_save_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # 备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{os.path.basename(checkpoint_path)}_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # 复制文件
            shutil.copy2(checkpoint_path, backup_path)
            
            self.logger.info(f"检查点已备份到: {backup_path}")
            return backup_path
        
        except Exception as e:
            self.logger.error(f"备份检查点失败: {str(e)}")
            return None 
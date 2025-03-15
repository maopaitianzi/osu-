#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估模块 - 用于评估和推理谱面生成模型
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from .transformer import TransformerModel


class Evaluator:
    """
    评估器类
    
    用于评估模型性能和生成谱面
    """
    
    def __init__(self, 
                 model: TransformerModel,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化评估器
        
        参数:
            model: 要评估的模型
            device: 推理设备
        """
        self.model = model.to(device)
        self.device = device
        
        # 确保模型处于评估模式
        self.model.eval()
    
    def evaluate_metrics(self, 
                         dataloader: torch.utils.data.DataLoader, 
                         metrics: List[str] = ["mse", "mae"]) -> Dict[str, float]:
        """
        评估模型性能指标
        
        参数:
            dataloader: 测试数据加载器
            metrics: 要计算的指标列表
            
        返回:
            包含评估结果的字典
        """
        results = {}
        total_mse = 0.0
        total_mae = 0.0
        
        # 跟踪每个特征维度的误差
        dimension_errors = {"x": [], "y": [], "time": [], "type": []}
        
        with torch.no_grad():
            for audio_features, beatmap_features, audio_mask, beatmap_mask in tqdm(dataloader, desc="评估中"):
                # 将数据移到设备上
                audio_features = audio_features.to(self.device)
                beatmap_features = beatmap_features.to(self.device)
                audio_mask = audio_mask.to(self.device)
                beatmap_mask = beatmap_mask.to(self.device)
                
                # 自回归生成
                generated = []
                
                # 初始输入（第一个时间步的物件）
                current_input = beatmap_features[:, 0:1, :]
                
                # 逐步生成后续物件
                for i in range(1, beatmap_features.size(1)):
                    # 排除掩码位置
                    if beatmap_mask[:, i].any():
                        break
                    
                    # 创建目标掩码
                    tgt_mask = self.model._generate_square_subsequent_mask(len(generated) + 1).to(self.device)
                    
                    # 预测下一个物件
                    tgt_input = torch.cat([current_input] + generated, dim=1)
                    output = self.model(
                        audio_features,
                        tgt_input,
                        src_mask=audio_mask,
                        tgt_mask=tgt_mask
                    )
                    
                    # 取最后一个预测结果
                    prediction = output[:, -1:, :]
                    generated.append(prediction)
                    
                    # 计算与真实值的误差
                    true_val = beatmap_features[:, i:i+1, :]
                    
                    # MSE
                    mse = ((prediction - true_val) ** 2).mean().item()
                    total_mse += mse
                    
                    # MAE
                    mae = torch.abs(prediction - true_val).mean().item()
                    total_mae += mae
                    
                    # 各维度误差
                    for j, dim_name in enumerate(["x", "y", "time", "type"]):
                        dim_error = torch.abs(prediction[:, :, j] - true_val[:, :, j]).mean().item()
                        dimension_errors[dim_name].append(dim_error)
                
                # 合并生成结果，形状为 [batch_size, seq_len-1, feature_dim]
                if generated:
                    predicted_sequence = torch.cat(generated, dim=1)
                
        # 计算平均指标
        avg_mse = total_mse / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        
        # 各维度平均误差
        dim_avg_errors = {}
        for dim_name, errors in dimension_errors.items():
            if errors:
                dim_avg_errors[f"{dim_name}_error"] = sum(errors) / len(errors)
        
        # 构建结果字典
        if "mse" in metrics:
            results["mse"] = avg_mse
        if "mae" in metrics:
            results["mae"] = avg_mae
        
        # 添加维度误差
        results.update(dim_avg_errors)
        
        return results
    
    def generate_beatmap(self, 
                         audio_features: torch.Tensor,
                         start_obj: Optional[torch.Tensor] = None,
                         max_length: int = 500,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        生成谱面
        
        参数:
            audio_features: 音频特征，形状为 [batch_size, seq_length, feature_dim]
            start_obj: 起始物件 (可选)，形状为 [batch_size, 1, 4]
            max_length: 生成的最大物件数
            temperature: 生成温度
            
        返回:
            生成的谱面，形状为 [batch_size, seq_length, 4]
        """
        # 确保输入是批次形式
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        
        batch_size = audio_features.size(0)
        
        # 移动到设备上
        audio_features = audio_features.to(self.device)
        
        # 如果没有提供起始物件，则创建一个默认的起始物件
        if start_obj is None:
            start_obj = torch.zeros(batch_size, 1, 4, device=self.device)
            # 默认值: x=0.5, y=0.5, time=0, type=1 (圆圈)
            start_obj[:, 0, 0] = 0.5  # x坐标 (归一化)
            start_obj[:, 0, 1] = 0.5  # y坐标 (归一化)
            start_obj[:, 0, 2] = 0.0  # 时间 (归一化)
            start_obj[:, 0, 3] = 0.1  # 类型 (归一化，1表示圆圈)
        else:
            start_obj = start_obj.to(self.device)
        
        with torch.no_grad():
            # 编码音频特征
            memory = self.model.encoder(audio_features)
            
            # 初始化生成序列
            generated = [start_obj]
            
            # 自回归生成剩余物件
            for i in range(max_length - 1):
                # 创建掩码
                tgt_mask = self.model._generate_square_subsequent_mask(i + 1).to(self.device)
                
                # 目标序列嵌入
                tgt = torch.cat(generated, dim=1)
                tgt_emb = self.model.tgt_embedding(tgt)
                
                # 解码
                output = self.model.decoder(tgt_emb, memory, tgt_mask)
                
                # 获取下一个物件预测
                next_obj = output[:, -1:, :]
                
                # 应用温度
                if temperature != 1.0:
                    next_obj = next_obj / temperature
                
                # 确保物件类型有效 (归一化后的类型值应在0-0.4之间，对应OSU的类型0-4)
                next_obj[:, :, 3] = torch.clamp(next_obj[:, :, 3], 0.0, 0.4)
                
                # 确保坐标在有效范围内
                next_obj[:, :, 0] = torch.clamp(next_obj[:, :, 0], 0.0, 1.0)  # x坐标
                next_obj[:, :, 1] = torch.clamp(next_obj[:, :, 1], 0.0, 1.0)  # y坐标
                
                # 确保时间是递增的
                if i > 0:
                    prev_time = generated[-1][:, :, 2]
                    next_obj[:, :, 2] = torch.clamp(next_obj[:, :, 2], min=prev_time + 0.001)
                
                # 添加到生成序列
                generated.append(next_obj)
        
        # 合并生成结果
        beatmap_sequence = torch.cat(generated, dim=1)
        
        return beatmap_sequence
    
    def convert_to_osu_format(self, 
                             beatmap_sequence: torch.Tensor,
                             audio_path: str,
                             artist: str = "Unknown",
                             title: str = "Generated Beatmap",
                             creator: str = "AI",
                             version: str = "AI Generated",
                             base_template: Optional[str] = None) -> str:
        """
        将生成的谱面序列转换为OSU文件格式
        
        参数:
            beatmap_sequence: 生成的谱面序列，形状为 [batch_size, seq_length, 4]
            audio_path: 音频文件路径
            artist: 艺术家名称
            title: 歌曲标题
            creator: 谱面创建者
            version: 谱面难度版本
            base_template: 基础OSU文件模板 (可选)
            
        返回:
            OSU文件内容
        """
        # 默认模板
        if base_template is None:
            base_template = f"""osu file format v14

[General]
AudioFilename: {os.path.basename(audio_path)}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 0
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 0.8
BeatDivisor: 4
GridSize: 32
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:{artist}
ArtistUnicode:{artist}
Creator:{creator}
Version:{version}
Source:
Tags:AI generated
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Layer 4 (Overlay)
//Storyboard Sound Samples

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
"""
        
        # 确保是numpy数组
        if isinstance(beatmap_sequence, torch.Tensor):
            beatmap_sequence = beatmap_sequence.cpu().numpy()
        
        # 取第一个批次（如果有多个）
        if beatmap_sequence.ndim == 3:
            beatmap_sequence = beatmap_sequence[0]
        
        # 构建HitObjects部分
        hit_objects = []
        
        for i in range(len(beatmap_sequence)):
            obj = beatmap_sequence[i]
            
            # 反归一化
            x = int(obj[0] * 512)  # x坐标
            y = int(obj[1] * 384)  # y坐标
            time = int(obj[2] * 60000)  # 时间（毫秒）
            obj_type = int(obj[3] * 10)  # 物件类型
            
            # 确保坐标在有效范围内
            x = max(0, min(512, x))
            y = max(0, min(384, y))
            
            # 确保时间非负
            time = max(0, time)
            
            # 确保类型有效 (1: 圆圈, 2: 滑条, 8: 转盘)
            if obj_type not in [1, 2, 8]:
                obj_type = 1  # 默认为圆圈
            
            # 构建物件字符串
            if obj_type == 1:  # 圆圈
                hit_obj_str = f"{x},{y},{time},{obj_type},0,0:0:0:0:"
            elif obj_type == 2:  # 滑条
                # 添加简单的直线滑条
                end_x = min(512, x + 100)
                end_y = y
                hit_obj_str = f"{x},{y},{time},{obj_type},0,L|{end_x}:{end_y},1,100,0,0:0:0:0:"
            elif obj_type == 8:  # 转盘
                # 转盘需要结束时间
                end_time = time + 1000  # 1秒转盘
                hit_obj_str = f"{x},{y},{time},{obj_type},0,{end_time},0:0:0:0:"
            
            hit_objects.append(hit_obj_str)
        
        # 合并OSU文件内容
        osu_content = base_template + "\n".join(hit_objects)
        
        return osu_content
    
    def save_osu_file(self, 
                      osu_content: str, 
                      output_path: str) -> None:
        """
        保存OSU文件
        
        参数:
            osu_content: OSU文件内容
            output_path: 输出路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(osu_content)
    
    def visualize_beatmap(self,
                         beatmap_sequence: torch.Tensor,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        可视化谱面
        
        参数:
            beatmap_sequence: 谱面序列，形状为 [batch_size, seq_length, 4]
            save_path: 保存路径 (可选)
            
        返回:
            matplotlib Figure对象
        """
        # 确保是numpy数组
        if isinstance(beatmap_sequence, torch.Tensor):
            beatmap_sequence = beatmap_sequence.cpu().numpy()
        
        # 取第一个批次（如果有多个）
        if beatmap_sequence.ndim == 3:
            beatmap_sequence = beatmap_sequence[0]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 提取坐标
        x_coords = beatmap_sequence[:, 0] * 512  # 反归一化
        y_coords = beatmap_sequence[:, 1] * 384  # 反归一化
        times = beatmap_sequence[:, 2] * 60000   # 反归一化（毫秒）
        types = beatmap_sequence[:, 3] * 10      # 反归一化
        
        # 物件类型颜色映射
        colors = []
        for t in types:
            if t < 1.5:  # 圆圈
                colors.append('blue')
            elif t < 2.5:  # 滑条
                colors.append('green')
            else:  # 转盘或其他
                colors.append('red')
        
        # 绘制物件分布
        scatter = ax1.scatter(x_coords, y_coords, c=colors, alpha=0.7)
        ax1.set_xlim(0, 512)
        ax1.set_ylim(384, 0)  # 反转y轴以匹配OSU坐标系
        ax1.set_title('物件分布')
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.grid(True)
        
        # 添加连接线以显示物件顺序
        for i in range(len(x_coords) - 1):
            ax1.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 'gray', alpha=0.3)
        
        # 绘制时间分布
        ax2.plot(range(len(times)), times, 'o-', color='purple')
        ax2.set_title('时间分布')
        ax2.set_xlabel('物件索引')
        ax2.set_ylabel('时间 (毫秒)')
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_beatmaps(self,
                         generated: torch.Tensor,
                         reference: torch.Tensor,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        比较生成的谱面和参考谱面
        
        参数:
            generated: 生成的谱面序列
            reference: 参考谱面序列
            save_path: 保存路径 (可选)
            
        返回:
            matplotlib Figure对象
        """
        # 确保是numpy数组
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()
        
        # 取第一个批次（如果有多个）
        if generated.ndim == 3:
            generated = generated[0]
        if reference.ndim == 3:
            reference = reference[0]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取生成谱面的数据
        gen_x = generated[:, 0] * 512
        gen_y = generated[:, 1] * 384
        gen_times = generated[:, 2] * 60000
        
        # 提取参考谱面的数据
        ref_x = reference[:, 0] * 512
        ref_y = reference[:, 1] * 384
        ref_times = reference[:, 2] * 60000
        
        # 绘制生成谱面的物件分布
        axes[0, 0].scatter(gen_x, gen_y, color='blue', alpha=0.7)
        for i in range(len(gen_x) - 1):
            axes[0, 0].plot([gen_x[i], gen_x[i+1]], [gen_y[i], gen_y[i+1]], 'gray', alpha=0.3)
        axes[0, 0].set_xlim(0, 512)
        axes[0, 0].set_ylim(384, 0)
        axes[0, 0].set_title('生成谱面物件分布')
        axes[0, 0].grid(True)
        
        # 绘制参考谱面的物件分布
        axes[0, 1].scatter(ref_x, ref_y, color='green', alpha=0.7)
        for i in range(len(ref_x) - 1):
            axes[0, 1].plot([ref_x[i], ref_x[i+1]], [ref_y[i], ref_y[i+1]], 'gray', alpha=0.3)
        axes[0, 1].set_xlim(0, 512)
        axes[0, 1].set_ylim(384, 0)
        axes[0, 1].set_title('参考谱面物件分布')
        axes[0, 1].grid(True)
        
        # 绘制生成谱面的时间分布
        axes[1, 0].plot(range(len(gen_times)), gen_times, 'o-', color='blue')
        axes[1, 0].set_title('生成谱面时间分布')
        axes[1, 0].set_xlabel('物件索引')
        axes[1, 0].set_ylabel('时间 (毫秒)')
        axes[1, 0].grid(True)
        
        # 绘制参考谱面的时间分布
        axes[1, 1].plot(range(len(ref_times)), ref_times, 'o-', color='green')
        axes[1, 1].set_title('参考谱面时间分布')
        axes[1, 1].set_xlabel('物件索引')
        axes[1, 1].set_ylabel('时间 (毫秒)')
        axes[1, 1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 
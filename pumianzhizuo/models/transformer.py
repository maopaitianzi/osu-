#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
谱面生成器Transformer模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List

from .positional_encoding import PositionalEncoding


class FeatureEncoder(nn.Module):
    """
    音频特征编码器
    
    将原始音频特征转换为模型可处理的隐藏表示。
    
    Args:
        feature_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        dropout (float): Dropout率
    """
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_encoder = PositionalEncoding(hidden_dim, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 形状为 [batch_size, seq_length, feature_dim] 的输入特征
            
        Returns:
            torch.Tensor: 形状为 [batch_size, seq_length, hidden_dim] 的编码特征
        """
        x = self.feature_projection(x)
        x = self.position_encoder(x)
        return x


class BeatmapDecoder(nn.Module):
    """
    谱面解码器
    
    将Transformer的输出转换为谱面元素的参数。
    
    Args:
        hidden_dim (int): 隐藏层维度
        n_object_types (int): 物件类型数量
        n_positions (int): 离散化的位置数量
        dropout (float): Dropout率
    """
    def __init__(
        self, 
        hidden_dim: int, 
        n_object_types: int = 4, 
        n_positions: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 物件类型预测头
        self.object_type_head = nn.Linear(hidden_dim // 2, n_object_types)
        
        # 位置预测头 - 预测离散化的x坐标和y坐标
        self.position_head = nn.Linear(hidden_dim // 2, n_positions * 2)
        
        # 时间偏移预测头 - 相对于节拍的时间偏移
        self.time_offset_head = nn.Linear(hidden_dim // 2, 1)
        
        # 滑条长度预测头 (对于滑条类型的物件)
        self.slider_length_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 形状为 [batch_size, seq_length, hidden_dim] 的输入特征
            
        Returns:
            Dict[str, torch.Tensor]: 包含各种预测结果的字典
                - object_type: 物件类型预测，形状为 [batch_size, seq_length, n_object_types]
                - position: 位置预测，形状为 [batch_size, seq_length, n_positions*2]
                - time_offset: 时间偏移预测，形状为 [batch_size, seq_length, 1]
                - slider_length: 滑条长度预测，形状为 [batch_size, seq_length, 1]
        """
        features = self.feature_extractor(x)
        
        return {
            "object_type": self.object_type_head(features),
            "position": self.position_head(features),
            "time_offset": self.time_offset_head(features),
            "slider_length": self.slider_length_head(features)
        }


class BeatmapTransformer(nn.Module):
    """
    谱面生成Transformer模型
    
    基于Transformer架构的谱面生成模型，将音频特征序列映射到谱面元素序列。
    
    Args:
        feature_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        n_object_types (int): 物件类型数量
        n_positions (int): 离散化的位置数量
        n_encoder_layers (int): 编码器层数
        n_decoder_layers (int): 解码器层数
        nhead (int): 多头注意力中的头数
        dropout (float): Dropout率
    """
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        n_object_types: int = 4,
        n_positions: int = 100,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1
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
        self.beatmap_decoder = BeatmapDecoder(
            hidden_dim, 
            n_object_types, 
            n_positions, 
            dropout
        )
        
        # 目标嵌入层
        self.object_type_embedding = nn.Embedding(n_object_types, hidden_dim // 4)
        self.position_embedding = nn.Embedding(n_positions * 2, hidden_dim // 4)
        self.time_embedding = nn.Linear(1, hidden_dim // 4)
        self.slider_embedding = nn.Linear(1, hidden_dim // 4)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成方形上三角掩码，用于解码器自注意力"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def process_targets(self, target_seq: torch.Tensor) -> torch.Tensor:
        """
        处理目标序列，转换为嵌入向量
        
        Args:
            target_seq (torch.Tensor): 形状为 [batch_size, seq_length, feature_count] 的目标序列
                feature_count应该包含：[object_type, pos_x, pos_y, time, slider_length]
                
        Returns:
            torch.Tensor: 形状为 [batch_size, seq_length, hidden_dim] 的嵌入向量
        """
        # 提取各特征
        object_type = target_seq[:, :, 0].long()
        pos_x = target_seq[:, :, 1].long()
        pos_y = target_seq[:, :, 2].long()
        time = target_seq[:, :, 3:4]
        slider_length = target_seq[:, :, 4:5]
        
        # 嵌入各特征
        object_embeds = self.object_type_embedding(object_type)
        pos_x_embeds = self.position_embedding(pos_x)
        pos_y_embeds = self.position_embedding(pos_y + self.position_embedding.num_embeddings // 2)  # 偏移y坐标的嵌入索引
        time_embeds = self.time_embedding(time)
        slider_embeds = self.slider_embedding(slider_length)
        
        # 组合所有嵌入
        combined_embeds = torch.cat([
            object_embeds, pos_x_embeds, pos_y_embeds, time_embeds, slider_embeds
        ], dim=-1)
        
        return combined_embeds
    
    def forward(
        self, 
        audio_features: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            audio_features (torch.Tensor): 形状为 [batch_size, seq_length, feature_dim] 的音频特征
            target_seq (Optional[torch.Tensor]): 训练时的目标序列，形状为 [batch_size, tgt_length, feature_count]
            target_padding_mask (Optional[torch.Tensor]): 目标序列的填充掩码
            
        Returns:
            Dict[str, torch.Tensor]: 模型预测结果
        """
        # 编码音频特征
        memory_key_padding_mask = None  # 可选：为音频特征添加填充掩码
        
        # 特征编码
        memory = self.feature_encoder(audio_features)
        
        # Transformer编码器
        memory = self.transformer_encoder(
            memory, 
            src_key_padding_mask=memory_key_padding_mask
        )
        
        if self.training and target_seq is not None:
            # 训练模式：Teacher forcing
            
            # 处理目标序列
            tgt = self.process_targets(target_seq)
            
            # 创建目标序列掩码（上三角掩码，防止看到未来信息）
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Transformer解码器
            output = self.transformer_decoder(
                tgt, 
                memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 输出投影
            output = self.output_projection(output)
            
            # 谱面解码器
            return self.beatmap_decoder(output)
        else:
            # 推理模式：自回归生成
            return self.generate(memory, memory_key_padding_mask)
    
    def generate(
        self, 
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        max_length: int = 1000, 
        temperature: float = 1.0,
        eos_token: int = 0  # 假设0为结束符
    ) -> Dict[str, torch.Tensor]:
        """
        自回归生成谱面序列
        
        Args:
            memory (torch.Tensor): 编码器输出，形状为 [batch_size, src_length, hidden_dim]
            memory_key_padding_mask (Optional[torch.Tensor]): 记忆填充掩码
            max_length (int): 生成的最大长度
            temperature (float): 采样温度，值越大随机性越强
            eos_token (int): 结束符的token ID
            
        Returns:
            Dict[str, torch.Tensor]: 生成的谱面序列
        """
        batch_size = memory.size(0)
        device = memory.device
        
        # 初始化结果存储
        results = {
            "object_type": [],
            "position": [],
            "time_offset": [],
            "slider_length": []
        }
        
        # 初始化首个token（起始符或特殊符号）
        # 这个设计需要根据具体应用场景调整
        object_type = torch.ones(batch_size, 1).long().to(device)  # 假设1为起始符
        pos_x = torch.zeros(batch_size, 1).long().to(device)
        pos_y = torch.zeros(batch_size, 1).long().to(device)
        time = torch.zeros(batch_size, 1, 1).to(device)
        slider_length = torch.zeros(batch_size, 1, 1).to(device)
        
        # 构建初始序列
        current_seq = torch.cat([
            object_type.unsqueeze(-1), 
            pos_x.unsqueeze(-1), 
            pos_y.unsqueeze(-1), 
            time, 
            slider_length
        ], dim=-1)
        
        # 生成掩码
        tgt_mask = self._generate_square_subsequent_mask(1).to(device)
        
        # 是否继续生成的标志，每个样本独立
        is_generating = torch.ones(batch_size, dtype=torch.bool).to(device)
        
        for i in range(max_length):
            # 只处理仍在生成的样本
            if not is_generating.any():
                break
                
            # 处理当前序列
            tgt_embeddings = self.process_targets(current_seq)
            
            # 使用Transformer解码器
            output = self.transformer_decoder(
                tgt_embeddings, 
                memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 输出投影
            output = self.output_projection(output)
            
            # 谱面解码
            predictions = self.beatmap_decoder(output)
            
            # 仅获取最后一个时间步的预测
            last_predictions = {k: v[:, -1:, :] for k, v in predictions.items()}
            
            # 对预测进行采样或argmax
            object_type_probs = F.softmax(last_predictions["object_type"] / temperature, dim=-1)
            sampled_object_type = torch.multinomial(object_type_probs.view(batch_size, -1), 1)
            
            # 根据当前生成的物件类型选择合适的位置和时间
            # 简化实现：取argmax
            position_logits = last_predictions["position"].view(batch_size, 1, -1)
            sampled_position = torch.argmax(position_logits, dim=-1)
            pos_x = sampled_position % self.position_embedding.num_embeddings
            pos_y = sampled_position // self.position_embedding.num_embeddings
            
            time_offset = last_predictions["time_offset"]
            slider_length = last_predictions["slider_length"]
            
            # 将新生成的token添加到结果中
            results["object_type"].append(sampled_object_type)
            results["position"].append(position_logits)
            results["time_offset"].append(time_offset)
            results["slider_length"].append(slider_length)
            
            # 更新生成状态
            is_generating = is_generating & (sampled_object_type.squeeze(-1) != eos_token)
            
            # 准备下一个时间步的输入
            new_seq = torch.cat([
                sampled_object_type.unsqueeze(-1),
                pos_x.unsqueeze(-1),
                pos_y.unsqueeze(-1),
                time_offset,
                slider_length
            ], dim=-1)
            
            current_seq = torch.cat([current_seq, new_seq], dim=1)
            
            # 更新掩码大小
            tgt_mask = self._generate_square_subsequent_mask(current_seq.size(1)).to(device)
        
        # 将结果转换为张量
        for k in results:
            results[k] = torch.cat(results[k], dim=1)
            
        return results


# 简单测试代码
if __name__ == "__main__":
    # 创建测试输入
    batch_size = 2
    src_length = 100
    feature_dim = 128
    tgt_length = 20
    hidden_dim = 256
    
    audio_features = torch.rand(batch_size, src_length, feature_dim)
    target_seq = torch.rand(batch_size, tgt_length, 5)  # [object_type, pos_x, pos_y, time, slider_length]
    target_seq[:, :, 0] = torch.randint(0, 4, (batch_size, tgt_length))  # 物件类型
    target_seq[:, :, 1] = torch.randint(0, 100, (batch_size, tgt_length))  # x坐标
    target_seq[:, :, 2] = torch.randint(0, 100, (batch_size, tgt_length))  # y坐标
    
    # 创建模型
    model = BeatmapTransformer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim
    )
    
    # 测试训练模式
    print("Testing training mode...")
    outputs = model(audio_features, target_seq)
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
    
    # 测试推理模式
    print("\nTesting inference mode...")
    outputs = model(audio_features)
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
        
    print("\nAll tests passed!") 
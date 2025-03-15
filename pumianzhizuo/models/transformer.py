#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer模型 - 用于谱面生成的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    将序列中元素的位置信息编码为向量，然后与元素的特征向量相加
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        初始化位置编码模块
        
        参数:
            d_model: 模型的维度
            max_seq_length: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        返回:
            添加了位置编码的张量，形状为 [batch_size, seq_length, d_model]
        """
        return x + self.pe[:x.size(1)]


class AudioEncoder(nn.Module):
    """
    音频特征编码器
    
    将音频特征编码为隐藏状态
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 d_model: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        """
        初始化音频编码器
        
        参数:
            input_dim: 输入特征的维度
            d_model: Transformer的维度
            nhead: 多头注意力的头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈神经网络的维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 输入特征投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            src: 输入特征，形状为 [batch_size, seq_length, input_dim]
            src_mask: 掩码，形状为 [batch_size, seq_length]
            
        返回:
            编码后的特征，形状为 [batch_size, seq_length, d_model]
        """
        # 投影到模型维度
        src = self.input_projection(src)
        
        # 应用位置编码
        src = self.pos_encoder(src)
        
        # 应用Dropout
        src = self.dropout(src)
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        return output


class BeatmapDecoder(nn.Module):
    """
    谱面解码器
    
    将编码的特征解码为谱面物件序列
    """
    
    def __init__(self, 
                 d_model: int = 512, 
                 output_dim: int = 64, 
                 nhead: int = 8, 
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        """
        初始化谱面解码器
        
        参数:
            d_model: Transformer的维度
            output_dim: 输出特征的维度
            nhead: 多头注意力的头数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈神经网络的维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            tgt: 目标序列，形状为 [batch_size, tgt_seq_length, d_model]
            memory: 编码器的输出，形状为 [batch_size, src_seq_length, d_model]
            tgt_mask: 目标序列的掩码，形状为 [tgt_seq_length, tgt_seq_length]
            memory_mask: 记忆的掩码，形状为 [tgt_seq_length, src_seq_length]
            
        返回:
            解码后的特征，形状为 [batch_size, tgt_seq_length, output_dim]
        """
        # 应用位置编码
        tgt = self.pos_encoder(tgt)
        
        # 应用Dropout
        tgt = self.dropout(tgt)
        
        # 通过Transformer解码器
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # 投影到输出维度
        output = self.output_projection(output)
        
        return output


class TransformerModel(nn.Module):
    """
    完整的Transformer模型，用于谱面生成
    
    包含一个音频编码器和一个谱面解码器
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 d_model: int = 512, 
                 output_dim: int = 64,
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        """
        初始化Transformer模型
        
        参数:
            input_dim: 输入特征的维度
            d_model: Transformer的维度
            output_dim: 输出特征的维度
            nhead: 多头注意力的头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈神经网络的维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 音频编码器
        self.encoder = AudioEncoder(
            input_dim=input_dim, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        
        # 谱面解码器
        self.decoder = BeatmapDecoder(
            d_model=d_model, 
            output_dim=output_dim, 
            nhead=nhead, 
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        
        # 目标嵌入层
        self.tgt_embedding = nn.Linear(output_dim, d_model)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """
        初始化模型参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            src: 源序列，形状为 [batch_size, src_seq_length, input_dim]
            tgt: 目标序列，形状为 [batch_size, tgt_seq_length, output_dim]
            src_mask: 源序列的掩码，形状为 [batch_size, src_seq_length]
            tgt_mask: 目标序列的掩码，形状为 [tgt_seq_length, tgt_seq_length]
            memory_mask: 记忆的掩码，形状为 [tgt_seq_length, src_seq_length]
            
        返回:
            输出序列，形状为 [batch_size, tgt_seq_length, output_dim]
        """
        # 编码
        memory = self.encoder(src, src_mask)
        
        # 目标序列嵌入
        tgt = self.tgt_embedding(tgt)
        
        # 解码
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        
        return output
    
    def generate(self, 
                 src: torch.Tensor, 
                 max_length: int = 1000, 
                 temperature: float = 1.0,
                 src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成谱面
        
        参数:
            src: 源序列，形状为 [batch_size, src_seq_length, input_dim]
            max_length: 生成的最大长度
            temperature: 温度参数，控制采样的随机性
            src_mask: 源序列的掩码，形状为 [batch_size, src_seq_length]
            
        返回:
            生成的序列，形状为 [batch_size, gen_seq_length, output_dim]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码
        memory = self.encoder(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.zeros(batch_size, 1, self.decoder.output_projection.out_features, device=device)
        
        # 生成序列
        for i in range(max_length - 1):
            # 创建三角形掩码
            tgt_mask = self._generate_square_subsequent_mask(i + 1).to(device)
            
            # 目标序列嵌入
            tgt_emb = self.tgt_embedding(tgt)
            
            # 解码
            output = self.decoder(tgt_emb, memory, tgt_mask)
            
            # 获取下一个词的预测
            next_item = output[:, -1:, :]
            
            # 应用温度
            if temperature != 1.0:
                next_item = next_item / temperature
            
            # 连接到目标序列
            tgt = torch.cat([tgt, next_item], dim=1)
        
        return tgt[:, 1:]  # 去掉初始的零向量
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成方形的后续掩码
        
        参数:
            sz: 掩码的大小
            
        返回:
            形状为 [sz, sz] 的掩码
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 
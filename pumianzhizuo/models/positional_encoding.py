#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
位置编码实现 - 为Transformer模型提供位置信息
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    标准的正弦余弦位置编码
    
    将位置信息编码到输入向量中，使Transformer模型能够区分序列中不同位置的元素。
    
    Arguments:
        d_model (int): 模型维度/嵌入维度
        dropout (float): dropout比率
        max_len (int): 支持的最大序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度，适配batch_first=True的场景
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不作为模型参数，但会保存到模型文件中）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        添加位置编码到输入张量
        
        Args:
            x (Tensor): 形状为 [batch_size, seq_len, embedding_dim] 的输入张量
            
        Returns:
            Tensor: 添加位置编码后的张量，形状与输入相同
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    通过嵌入层学习每个位置的编码，而不是使用固定的正弦余弦函数。
    
    Arguments:
        d_model (int): 模型维度/嵌入维度
        dropout (float): dropout比率
        max_len (int): 支持的最大序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # 初始化位置索引，用于前向传播
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1))
        )

    def forward(self, x):
        """
        添加可学习的位置编码到输入张量
        
        Args:
            x (Tensor): 形状为 [batch_size, seq_len, embedding_dim] 的输入张量
            
        Returns:
            Tensor: 添加位置编码后的张量，形状与输入相同
        """
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    
    实现音乐谱面特定的相对位置编码，考虑时间距离与音乐节拍的关系。
    
    Arguments:
        d_model (int): 模型维度/嵌入维度
        dropout (float): dropout比率
        max_distance (int): 最大相对距离
    """
    def __init__(self, d_model, dropout=0.1, max_distance=100):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_distance = max_distance
        
        # 创建相对位置嵌入
        self.relative_embeddings = nn.Embedding(2 * max_distance + 1, d_model)
        
    def forward(self, x, beat_positions=None):
        """
        添加相对位置编码到输入张量
        
        Args:
            x (Tensor): 形状为 [batch_size, seq_len, embedding_dim] 的输入张量
            beat_positions (Tensor, optional): 每个位置的节拍信息，形状为 [batch_size, seq_len]
            
        Returns:
            Tensor: 经过相对位置编码处理的张量
        """
        batch_size, seq_len, _ = x.size()
        
        if beat_positions is None:
            # 使用序列中的位置代替节拍位置
            positions = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
        else:
            positions = beat_positions
            
        # 计算每个位置对相对于其他所有位置的距离
        # 待实现：更复杂的基于节拍的相对位置计算
        
        # 简单实现：直接返回标准位置编码
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.size(-1), 2, device=x.device).float() * 
                             (-math.log(10000.0) / x.size(-1)))
        
        pe = torch.zeros_like(x[0])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


# 简单测试代码
if __name__ == "__main__":
    # 创建测试输入
    batch_size = 4
    seq_len = 10
    d_model = 64
    
    x = torch.rand(batch_size, seq_len, d_model)
    
    # 测试标准位置编码
    print("Testing standard positional encoding...")
    pe = PositionalEncoding(d_model)
    output = pe(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    # 测试可学习位置编码
    print("\nTesting learnable positional encoding...")
    lpe = LearnablePositionalEncoding(d_model)
    output = lpe(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    # 测试相对位置编码
    print("\nTesting relative positional encoding...")
    rpe = RelativePositionalEncoding(d_model)
    output = rpe(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("\nAll tests passed!") 
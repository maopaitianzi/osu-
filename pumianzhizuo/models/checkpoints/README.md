# 模型检查点目录

此目录用于存储训练过程中保存的模型检查点文件。

## 目录结构

检查点文件将使用以下命名规则：
```
{model_name}_{date}_{epoch}.pth
```

例如：
```
beatmap_transformer_20230320_epoch50.pth
```

## 检查点内容

每个检查点文件包含以下内容：

1. 模型参数（state_dict）
2. 优化器状态
3. 训练配置
4. 训练统计信息（损失值、准确率等）
5. 当前轮次和步数

## 使用方法

### 加载检查点

可以使用以下代码加载检查点：

```python
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_path: 检查点文件路径
        
    Returns:
        epoch: 训练轮次
        stats: 训练统计信息
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['stats']
```

### 保存检查点

可以使用以下代码保存检查点：

```python
def save_checkpoint(model, optimizer, epoch, stats, config, checkpoint_dir):
    """
    保存模型检查点
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前训练轮次
        stats: 训练统计信息
        config: 训练配置
        checkpoint_dir: 检查点保存目录
    """
    from datetime import datetime
    
    date_str = datetime.now().strftime("%Y%m%d")
    checkpoint_path = f"{checkpoint_dir}/{config['model_name']}_{date_str}_epoch{epoch}.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
```

## 注意事项

- 定期删除旧的检查点以节省磁盘空间
- 始终保留最新的和最佳的检查点
- 对于重要的里程碑模型，考虑使用更具描述性的文件名 
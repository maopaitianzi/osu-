# 高级音频分析模块

这个模块提供了高级音频分析功能，专门为osu!谱面生成优化。模块可以分析音频文件，提取丰富的特征，并提供直观的可视化界面。

## 主要特性

### AudioAnalyzer 类

这个类提供了强大的音频分析功能：

- **多种BPM检测算法**：使用多种算法确保准确检测BPM
- **节拍检测和网格生成**：检测节拍位置并创建规范的节拍网格
- **节拍强度分析**：标记歌曲中的强拍和弱拍
- **音量/能量变化分析**：检测能量变化点，适合放置物件
- **丰富的频谱特征**：提取频谱质心、色度图等特征
- **音频段落检测**：自动检测歌曲的段落变化
- **过渡点检测**：找出适合放置关键物件的过渡点
- **谱面参数推荐**：根据音频特征推荐合适的谱面参数

### AudioVisualizer 类

这个类提供了丰富的音频可视化功能：

- **波形图**：显示音频波形，并标记节拍和段落
- **频谱图**：显示音频的频谱变化
- **梅尔频谱图**：展示音频的梅尔频谱
- **色度图**：显示音频的音高/和弦变化
- **音量变化图**：展示音频能量变化

## 使用方法

### 基本分析流程

```python
# 创建分析器
analyzer = AudioAnalyzer()

# 加载音频文件
analyzer.load_audio("your_audio.mp3")

# 执行分析
features = analyzer.analyze()

# 获取结果
bpm = analyzer.get_bpm()
beat_times = analyzer.get_beat_times()
energy_points = analyzer.get_energy_points(threshold=0.8)
osu_points = analyzer.get_osu_timing_points()
```

### 与可视化器集成

```python
# 创建可视化器
visualizer = AudioVisualizer()

# 设置音频数据
visualizer.set_audio_data(analyzer.y, analyzer.sr)

# 设置分析特征
visualizer.set_audio_features(analyzer.features)
```

## 导出功能

分析结果可以导出为JSON文件，便于谱面生成器使用：

```python
output_path = analyzer.export_analysis_to_json("analysis_output.json")
```

## 依赖要求

- librosa (音频分析核心库)
- numpy (数值计算)
- matplotlib (可视化)
- PyQt5 (UI集成)

## 技术细节

- **BPM检测**：使用动态规划节拍跟踪和调谐范围检测
- **段落检测**：使用自相似矩阵和谱聚类
- **频谱特征**：包括MFCC、色度图、频谱质心等
- **过渡点检测**：结合节拍和音量变化检测 
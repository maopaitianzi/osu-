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
- **GPU加速支持**：需要CUDA环境
- **人声分离功能**：使用Demucs模型将音频分离为人声、鼓声、贝斯和其他乐器

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
analyzer = AudioAnalyzer(use_gpu=True)

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

### 人声分离功能

新增的人声分离功能允许将音频分离为不同的声音组件，并对特定组件进行分析。

```python
from audio.analyzer import AudioAnalyzer

# 创建分析器实例
analyzer = AudioAnalyzer(use_gpu=True)  # 使用GPU加速推荐用于音频分离

# 启用人声分离
analyzer.set_use_source_separation(True)

# 设置音频源优先级（默认为["vocals", "piano", "drums", "other"]）
# 可以调整顺序以更改优先级
analyzer.set_source_priority(["vocals", "drums", "bass", "other"])

# 加载音频文件
analyzer.load_audio("path/to/audio.mp3")

# 执行分析（会自动进行音频分离）
results = analyzer.analyze()

# 获取可用的分离音频源
available_sources = analyzer.get_available_sources()  # 例如：["vocals", "drums", "bass", "other", "original"]

# 设置当前活跃的音频源（用于后续分析）
analyzer.set_active_source("vocals")  # 切换到人声轨道

# 导出分离的音频源
exported_files = analyzer.export_separated_audio("path/to/output/directory")
```

### 可视化

使用可视化器显示音频特征：

```python
from audio.analyzer import AudioAnalyzer
from audio.visualizer import AudioVisualizer
from PyQt5 import QtWidgets
import sys

# 创建应用
app = QtWidgets.QApplication(sys.argv)

# 创建可视化器
viz = AudioVisualizer()

# 创建分析器并启用人声分离
analyzer = AudioAnalyzer(use_gpu=True)
analyzer.set_use_source_separation(True)
analyzer.load_audio("path/to/audio.mp3")

# 执行分析
results = analyzer.analyze()

# 设置音频数据和特征
viz.set_audio_data(analyzer.y, analyzer.sr)
viz.set_audio_features(results)

# 将分离的音频源设置给可视化器
viz.separated_sources = analyzer.separated_sources

# 显示可视化器
viz.show()
sys.exit(app.exec_())
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
- **音频源分离**：支持多种先进模型，包括Demucs v4、MelBand RoFormer和SCNet XL

## 注意事项

1. 音频分离是一个计算密集型任务，建议在有GPU的环境中使用
2. 首次使用时会下载模型权重（约1-2GB），请确保网络连接正常
3. 分离大文件可能需要较长时间，请耐心等待
4. 分离结果的质量取决于原始音频的质量和内容以及选择的模型
5. 高级模型通常提供更好的分离质量，但可能需要更多计算资源

## 音频源分离

本系统支持多种先进的音频源分离模型：

### 1. Demucs v4 系列

Demucs是Meta Research开发的音频分离模型，可将音频分离为以下组件：

- **vocals**: 人声部分
- **drums**: 鼓声/打击乐部分
- **bass**: 贝斯部分
- **other**: 其他乐器（包括钢琴、吉他等）

可选的Demucs变体：
- **demucs_v4**: 标准版Demucs v4
- **htdemucs**: 混合变体，结合了时域和频域处理（推荐）
- **htdemucs_ft**: 微调版本，针对特定类型音乐进行了优化

> **更新说明**：**系统已完全修复音频源标签映射问题**。现在所有模型输出均已标准化，保证标签与实际内容一致：
> - vocals 标签 → 输出人声内容
> - drums 标签 → 输出鼓声内容
> - bass 标签 → 输出贝斯内容
> - other 标签 → 输出其他乐器内容
>
> 同时，导出文件名将包含明确的内容描述，例如"歌曲名_人声(vocals).wav"，确保文件名与实际内容匹配。

### 2. MelBand RoFormer（高性能）

MelBand RoFormer是新一代的高性能音频分离模型，基于Transformer架构，具有以下特点：
- 更高的人声分离清晰度，SDR可达11.28
- 更好的背景噪音抑制
- 更准确的声音细节保留

### 3. SCNet XL（高质量）

SCNet XL是目前最先进的音频分离模型之一，提供：
- 超高质量的人声分离（SDR 10.96）
- 优秀的乐器分离（SDR 17.27）
- 更好的相位一致性和空间感

分离的音频可以：

1. 用于更精确的音频分析（例如，仅分析人声或鼓声部分）
2. 导出为单独的音频文件
3. 在可视化器中单独查看每个音轨
4. 根据优先级选择性地应用于谱面生成 
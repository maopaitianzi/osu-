# 音乐游戏谱面生成器

基于深度学习的音乐游戏谱面自动生成系统，可以分析音频特征并生成符合游戏规则的谱面文件。

## 项目概述

本项目旨在通过人工智能技术，实现音乐游戏谱面的自动生成。系统能够分析音频文件的节拍、音量、频谱等特征，并结合已有的谱面设计模式，训练深度学习模型，最终自动生成符合游戏节奏和难度要求的谱面。目前支持生成OSU格式的谱面文件。

### 核心功能

- **音频分析**：检测BPM、节拍位置、音量变化、频谱特征等
- **谱面学习**：分析现有谱面的设计模式和规则
- **AI生成**：基于Transformer模型的谱面自动生成
- **难度调整**：支持Easy、Normal、Hard、Expert四个难度级别
- **风格定制**：支持不同音乐风格的谱面生成
- **可视化预览**：生成前预览谱面效果

## 当前开发状态

本项目目前处于开发阶段，已完成的组件包括：

1. **GUI前端框架**：✅
   - 完整的用户界面与交互功能
   - 音频分析、谱面分析、数据集处理、模型训练等功能模块

2. **音频分析模块**：✅
   - 音频加载与BPM检测
   - 频谱特征提取与节拍分析
   - 音频可视化功能

3. **谱面分析模块**：✅
   - OSU文件格式解析
   - 谱面统计与分布分析
   - 谱面模式识别

4. **模型架构**：✅
   - 基于Transformer的编码器-解码器结构
   - 特征编码器与谱面解码器
   - 位置编码实现

正在开发的功能：

1. **数据处理模块**：⚠️
   - 数据集构建与预处理
   - 特征提取与标准化

2. **模型训练流程**：🔄
   - 训练循环与损失函数
   - 模型评估与检查点保存

待开发的功能：

1. **谱面生成模块**：❌
   - 基于训练模型的谱面生成
   - 规则约束与后处理

更多详细信息，请参阅[开发计划文档.md](开发计划文档.md)。

## 开发流程

本项目的开发流程分为以下几个主要阶段：

1. **GUI程序前端开发**：创建用户友好的图形界面
2. **音频分析模块**：实现音频特征提取算法
3. **谱面分析模块**：开发谱面解析和模式识别
4. **模型训练阶段**：设计并训练生成模型
5. **谱面生成模块**：实现基于模型的谱面生成
6. **项目打包成EXE**：构建可分发的应用程序

详细开发流程请参考[项目开发流程图.md](项目开发流程图.md)。

## 系统要求

- **操作系统**：Windows 10/11、macOS 10.14+、Linux
- **Python版本**：Python 3.8+
- **GPU**：推荐NVIDIA GPU (用于模型训练和加速生成)
- **RAM**：最低4GB，推荐8GB以上
- **存储空间**：至少1GB可用空间

## 安装指南

### 从源代码安装

1. 克隆项目仓库：
   ```
   git clone https://github.com/your-username/beatmap-generator.git
   cd beatmap-generator
   ```

2. 创建虚拟环境（推荐）：
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

4. 运行应用：
   ```
   python main.py
   ```

### 使用预编译版本

1. 从[发布页面](https://github.com/your-username/beatmap-generator/releases)下载最新版本的安装包
2. 运行安装程序并按照指引完成安装
3. 从开始菜单或桌面快捷方式启动应用

## 使用指南

### 基本操作流程

1. **选择音频文件**：点击"浏览..."按钮选择一个音频文件（支持MP3、WAV、OGG、FLAC格式）
2. **分析音频**：点击"分析音频"按钮，系统将自动分析音频特征
3. **设置参数**：选择难度级别和谱面风格
4. **生成谱面**：点击"生成谱面"按钮，系统将自动生成谱面文件
5. **保存与导出**：系统会自动保存谱面到与音频文件相同的目录

### 高级设置

在应用的"设置"菜单中，可以调整更多高级参数：

- **模型选择**：切换不同的训练模型
- **生成参数**：调整生成算法参数
- **音频处理**：配置音频分析的详细参数
- **自定义难度**：创建自定义难度配置

## 开发者指南

### 项目结构

```
.
├── main.py                 # 主程序入口
├── gui/                    # GUI相关代码
├── audio/                  # 音频分析模块
│   ├── analyzer.py         # 音频分析器
│   └── features.py         # 特征提取
├── beatmap/                # 谱面相关代码
│   ├── parser.py           # 谱面解析器
│   ├── generator.py        # 谱面生成器
│   └── renderer.py         # 谱面渲染器
├── models/                 # 深度学习模型
│   ├── transformer.py      # Transformer模型
│   ├── training.py         # 训练代码
│   └── evaluation.py       # 评估代码
├── data/                   # 数据处理
│   ├── dataset.py          # 数据集类
│   └── preprocessing.py    # 预处理函数
├── utils/                  # 工具函数
├── resources/              # 资源文件
│   ├── models/             # 预训练模型
│   └── samples/            # 示例文件
└── tests/                  # 测试代码
```

### 扩展开发

如果您想要扩展项目功能，这里有一些建议的方向：

1. **添加新的谱面格式支持**：在`beatmap/parser.py`和`beatmap/generator.py`中添加新格式处理
2. **改进音频分析算法**：在`audio/analyzer.py`中实现更精确的特征提取
3. **优化模型架构**：在`models/transformer.py`中尝试不同的网络结构
4. **添加更多评估指标**：在`models/evaluation.py`中实现新的评估方法

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 提交问题和功能请求
- 修复bug和改进代码
- 添加新功能和测试
- 改进文档和示例

请遵循以下步骤提交贡献：

1. Fork项目仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 详细信息请参阅[LICENSE](LICENSE)文件。

## 联系方式

项目维护者：Your Name - your.email@example.com

项目链接：[https://github.com/your-username/beatmap-generator](https://github.com/your-username/beatmap-generator) 
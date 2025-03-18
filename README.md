# OSUMAP - OSU!谱面生成器

基于深度学习的osu!游戏谱面自动生成工具。通过音频分析和Transformer模型，根据音乐特征生成匹配的osu!谱面。

## 功能特点

- 🎵 **高级音频分析**: 提取BPM、节拍、频谱特征等音频信息
- 🤖 **AI谱面生成**: 基于Transformer的深度学习模型生成谱面
- 🎮 **多样难度支持**: 生成从Easy到Expert的多种难度谱面
- 🔊 **音频源分离**: 可选分离人声、鼓点等音轨用于精细分析
- 📊 **可视化工具**: 直观展示音频特征和谱面结构
- 🛠 **自定义选项**: 灵活调整生成参数和模型设置

## 安装说明

### 前置要求

- Python 3.8+
- [可选] NVIDIA GPU (用于加速模型训练和推理)

### 安装步骤

1. 克隆仓库:
   ```bash
   git clone https://github.com/yourusername/osumap.git
   cd osumap
   ```

2. 创建并激活虚拟环境:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

4. [可选] 安装音频处理系统依赖:
   ```bash
   # Linux
   sudo apt-get install ffmpeg libsndfile1

   # macOS
   brew install ffmpeg libsndfile

   # Windows
   # 建议通过Chocolatey安装ffmpeg
   choco install ffmpeg
   ```

## 快速开始

### 启动程序

```bash
# 在项目根目录下运行
python -m osu.pumianzhizuo.launcher
```

### 基本使用流程

1. 加载音频文件
2. 分析音频特征
3. 设置谱面生成参数
4. 生成谱面
5. 预览及导出谱面

## 使用教程

详细的使用教程请参阅[使用指南](docs/使用指南.md)。

## 开发者文档

- [开发文档](docs/开发文档.md)
- [依赖库文档](docs/依赖库文档.md)
- [贡献指南](docs/贡献指南.md)

## 技术架构

OSUMAP基于以下技术构建:

- **PyTorch**: 深度学习框架
- **librosa**: 音频分析库
- **PyQt5**: 图形用户界面
- **Transformer**: 自注意力深度学习架构

项目的核心是基于编码器-解码器结构的Transformer模型，将音频特征编码后，自回归生成谱面物件序列。

## 项目结构

```
pumianzhizuo/
├── audio/          # 音频分析模块
├── beatmap/        # 谱面生成与分析模块
├── models/         # 深度学习模型模块
├── gui/            # 图形用户界面模块
└── main.py         # 程序入口
```

## 贡献代码

欢迎贡献代码！请阅读[贡献指南](docs/贡献指南.md)了解详情。

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 致谢

- 感谢osu!社区提供的灵感和支持
- 感谢开源社区的各种优秀工具和库

## 联系方式

- **项目维护者**: [维护者姓名]
- **联系邮箱**: [邮箱地址]
- **项目主页**: [GitHub项目链接] 
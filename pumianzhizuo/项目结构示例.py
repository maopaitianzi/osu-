#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音乐游戏谱面生成系统 - 项目结构示例
"""

import os
import sys
import torch
import numpy as np
import librosa
from PyQt5 import QtWidgets, QtCore, QtGui


# ============= GUI前端模块 =============
class MainWindow(QtWidgets.QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音乐游戏谱面生成器")
        self.setMinimumSize(1024, 768)
        
        # 初始化UI
        self.init_ui()
        
        # 初始化模块
        self.audio_analyzer = AudioAnalyzer()
        self.beatmap_analyzer = BeatmapAnalyzer()
        self.model_manager = ModelManager()
        self.beatmap_generator = BeatmapGenerator()
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 顶部工具栏
        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)
        
        # 文件选择部分
        file_group = QtWidgets.QGroupBox("音频文件")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        
        self.file_path = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_audio)
        
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse_btn)
        
        # 参数设置部分
        params_group = QtWidgets.QGroupBox("谱面参数")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        self.difficulty_combo = QtWidgets.QComboBox()
        self.difficulty_combo.addItems(["Easy", "Normal", "Hard", "Expert"])
        
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(["标准", "流行", "电子", "古典"])
        
        params_layout.addRow("难度级别:", self.difficulty_combo)
        params_layout.addRow("曲风风格:", self.style_combo)
        
        # 操作按钮
        button_layout = QtWidgets.QHBoxLayout()
        
        analyze_btn = QtWidgets.QPushButton("分析音频")
        analyze_btn.clicked.connect(self.analyze_audio)
        
        generate_btn = QtWidgets.QPushButton("生成谱面")
        generate_btn.clicked.connect(self.generate_beatmap)
        
        button_layout.addWidget(analyze_btn)
        button_layout.addWidget(generate_btn)
        
        # 添加到主布局
        main_layout.addWidget(file_group)
        main_layout.addWidget(params_group)
        main_layout.addLayout(button_layout)
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
    
    def browse_audio(self):
        """浏览并选择音频文件"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.mp3 *.wav *.ogg *.flac)"
        )
        if file_path:
            self.file_path.setText(file_path)
    
    def analyze_audio(self):
        """分析音频文件"""
        file_path = self.file_path.text()
        if not file_path or not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的音频文件")
            return
        
        self.statusBar().showMessage("正在分析音频...")
        
        # 调用音频分析模块
        try:
            features = self.audio_analyzer.analyze(file_path)
            self.statusBar().showMessage(f"音频分析完成: BPM={features['bpm']:.1f}")
        except Exception as e:
            self.statusBar().showMessage(f"音频分析失败: {str(e)}")
    
    def generate_beatmap(self):
        """生成谱面"""
        file_path = self.file_path.text()
        if not file_path or not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的音频文件")
            return
        
        difficulty = self.difficulty_combo.currentText()
        style = self.style_combo.currentText()
        
        self.statusBar().showMessage("正在生成谱面...")
        
        # 调用谱面生成流程
        try:
            # 1. 分析音频
            audio_features = self.audio_analyzer.analyze(file_path)
            
            # 2. 加载模型
            model = self.model_manager.load_model(difficulty, style)
            
            # 3. 生成谱面
            beatmap = self.beatmap_generator.generate(audio_features, model, difficulty)
            
            # 4. 保存谱面
            save_path = os.path.splitext(file_path)[0] + ".osu"
            self.beatmap_generator.save(beatmap, save_path)
            
            self.statusBar().showMessage(f"谱面生成完成: {save_path}")
            
            # 显示成功消息
            QtWidgets.QMessageBox.information(
                self, "成功", f"谱面已生成并保存为:\n{save_path}"
            )
        except Exception as e:
            self.statusBar().showMessage(f"谱面生成失败: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "错误", f"谱面生成失败: {str(e)}")


# ============= 音频分析模块 =============
class AudioAnalyzer:
    """音频分析器类"""
    
    def __init__(self):
        self.sr = 22050  # 采样率
    
    def analyze(self, audio_path):
        """分析音频文件并提取特征"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # 提取特征
        features = {}
        
        # BPM检测
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features["bpm"] = tempo
        
        # 节拍定位
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        features["beat_times"] = beat_times
        
        # 音量曲线
        rms = librosa.feature.rms(y=y)[0]
        features["volume_curve"] = rms
        
        # 频谱特征
        spec = np.abs(librosa.stft(y))
        features["spectrogram"] = spec
        
        # 音高特征(如果有人声)
        if self._has_vocals(spec):
            f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=800, sr=sr)
            features["pitch"] = f0
            features["voiced"] = voiced_flag
        
        return features
    
    def _has_vocals(self, spectrogram):
        """简单判断是否包含人声"""
        # 这只是一个示例判断逻辑，实际项目中需要更复杂的人声检测算法
        return True


# ============= 谱面分析模块 =============
class BeatmapAnalyzer:
    """谱面分析器类"""
    
    def __init__(self):
        self.dataset = []
    
    def parse_osu_file(self, osu_path):
        """解析OSU谱面文件"""
        beatmap = {"hit_objects": []}
        
        with open(osu_path, "r", encoding="utf-8") as f:
            current_section = None
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith("//"):
                    continue
                
                # 检测区块
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    beatmap[current_section] = []
                    continue
                
                # 处理当前区块内容
                if current_section == "General":
                    if ":" in line:
                        key, value = [x.strip() for x in line.split(":", 1)]
                        beatmap[key] = value
                
                elif current_section == "Difficulty":
                    if ":" in line:
                        key, value = [x.strip() for x in line.split(":", 1)]
                        beatmap[key] = float(value)
                
                elif current_section == "TimingPoints":
                    beatmap.setdefault("TimingPoints", []).append(line.split(","))
                
                elif current_section == "HitObjects":
                    parts = line.split(",")
                    obj = {
                        "x": int(parts[0]),
                        "y": int(parts[1]),
                        "time": int(parts[2]),
                        "type": int(parts[3]),
                        "hitSound": int(parts[4])
                    }
                    beatmap["hit_objects"].append(obj)
        
        return beatmap
    
    def extract_patterns(self, beatmap):
        """从谱面中提取常见模式"""
        patterns = []
        hit_objects = beatmap["hit_objects"]
        
        # 按时间排序
        hit_objects.sort(key=lambda x: x["time"])
        
        # 提取连续的几个物件作为模式
        for i in range(len(hit_objects) - 3):
            pattern = hit_objects[i:i+4]
            time_diffs = [pattern[j+1]["time"] - pattern[j]["time"] for j in range(len(pattern)-1)]
            
            # 计算相对位置
            positions = [(obj["x"], obj["y"]) for obj in pattern]
            rel_positions = []
            for j in range(1, len(positions)):
                dx = positions[j][0] - positions[j-1][0]
                dy = positions[j][1] - positions[j-1][1]
                rel_positions.append((dx, dy))
            
            patterns.append({
                "time_diffs": time_diffs,
                "rel_positions": rel_positions,
                "types": [obj["type"] for obj in pattern]
            })
        
        return patterns
    
    def build_dataset(self, osu_folder, audio_analyzer):
        """构建训练数据集"""
        for root, _, files in os.walk(osu_folder):
            for file in files:
                if file.endswith(".osu"):
                    osu_path = os.path.join(root, file)
                    beatmap = self.parse_osu_file(osu_path)
                    
                    # 找到对应的音频文件
                    audio_file = beatmap.get("AudioFilename")
                    if audio_file:
                        audio_path = os.path.join(os.path.dirname(osu_path), audio_file)
                        if os.path.exists(audio_path):
                            # 分析音频
                            audio_features = audio_analyzer.analyze(audio_path)
                            
                            # 提取谱面模式
                            patterns = self.extract_patterns(beatmap)
                            
                            # 添加到数据集
                            self.dataset.append({
                                "audio_features": audio_features,
                                "beatmap_patterns": patterns,
                                "difficulty": {
                                    "AR": beatmap.get("ApproachRate"),
                                    "OD": beatmap.get("OverallDifficulty"),
                                    "CS": beatmap.get("CircleSize"),
                                    "HP": beatmap.get("HPDrainRate")
                                }
                            })
        
        return self.dataset


# ============= 模型训练模块 =============
class ModelManager:
    """模型管理器类"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_model(self):
        """构建深度学习模型"""
        # 这里使用一个简单的Transformer模型示例
        class BeatmapTransformer(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
                super().__init__()
                self.input_encoder = torch.nn.Linear(input_dim, hidden_dim)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=8, batch_first=True
                )
                self.transformer_encoder = torch.nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers
                )
                self.output_decoder = torch.nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x, mask=None):
                x = self.input_encoder(x)
                x = self.transformer_encoder(x, src_key_padding_mask=mask)
                return self.output_decoder(x)
        
        # 创建模型实例
        input_dim = 128  # 音频特征维度
        hidden_dim = 256  # 隐藏层维度
        output_dim = 64   # 输出维度(谱面参数)
        
        model = BeatmapTransformer(input_dim, hidden_dim, output_dim)
        model.to(self.device)
        
        return model
    
    def train(self, dataset, epochs=50, batch_size=32):
        """训练模型"""
        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 准备数据加载器
        # 这里简化处理，实际项目中需要正确实现数据集和数据加载
        train_loader = self._prepare_data_loader(dataset, batch_size)
        
        # 训练循环
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # 获取输入和目标
                inputs = batch["audio_features"].to(self.device)
                targets = batch["beatmap_patterns"].to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 每个epoch结束后更新学习率
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return model
    
    def _prepare_data_loader(self, dataset, batch_size):
        """准备数据加载器"""
        # 这是一个简化的示例，实际项目中需要实现完整的Dataset和DataLoader
        return [{"audio_features": torch.randn(batch_size, 128),
                 "beatmap_patterns": torch.randn(batch_size, 64)}
                for _ in range(10)]
    
    def save_model(self, model, difficulty, style):
        """保存模型"""
        model_name = f"model_{difficulty}_{style}.pth"
        torch.save(model.state_dict(), model_name)
        self.models[(difficulty, style)] = model_name
    
    def load_model(self, difficulty, style):
        """加载模型"""
        model_name = self.models.get((difficulty, style))
        if not model_name or not os.path.exists(model_name):
            # 如果模型不存在，使用默认模型
            model_name = "model_default.pth"
        
        model = self.build_model()
        model.load_state_dict(torch.load(model_name))
        model.eval()
        
        return model


# ============= 谱面生成模块 =============
class BeatmapGenerator:
    """谱面生成器类"""
    
    def __init__(self):
        pass
    
    def generate(self, audio_features, model, difficulty):
        """生成谱面"""
        # 将音频特征转换为模型输入格式
        input_features = self._prepare_model_input(audio_features)
        
        # 使用模型预测谱面参数
        with torch.no_grad():
            predicted_params = model(input_features)
        
        # 根据预测参数生成谱面
        beatmap = self._create_beatmap_from_params(predicted_params, audio_features, difficulty)
        
        # 应用难度调整
        beatmap = self._adjust_difficulty(beatmap, difficulty)
        
        # 应用规则约束
        beatmap = self._apply_constraints(beatmap)
        
        return beatmap
    
    def _prepare_model_input(self, audio_features):
        """准备模型输入"""
        # 提取关键特征并转换为张量
        bpm = audio_features["bpm"]
        beat_times = audio_features["beat_times"]
        volume_curve = audio_features["volume_curve"]
        
        # 创建特征序列
        features_seq = []
        for i, beat_time in enumerate(beat_times):
            # 找到对应时间点的音量
            time_idx = int(beat_time * 22050 / 512)  # 假设使用了512帧的STFT
            if time_idx < len(volume_curve):
                volume = volume_curve[time_idx]
            else:
                volume = 0
            
            # 添加到特征序列
            features_seq.append([beat_time, volume, bpm])
        
        # 转换为张量
        input_tensor = torch.tensor(features_seq, dtype=torch.float32)
        # 扩展为批次维度
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def _create_beatmap_from_params(self, params, audio_features, difficulty):
        """根据参数生成谱面"""
        # 初始化谱面
        beatmap = {
            "version": 14,
            "general": {
                "AudioFilename": os.path.basename(audio_features.get("file_path", "audio.mp3")),
                "AudioLeadIn": 0,
                "PreviewTime": -1,
                "Countdown": 0,
                "SampleSet": "Normal",
                "StackLeniency": 0.7,
                "Mode": 0,
                "LetterboxInBreaks": 0,
                "WidescreenStoryboard": 1
            },
            "metadata": {
                "Title": "Generated Beatmap",
                "Artist": "AI Generator",
                "Creator": "Beatmap Generator AI",
                "Version": difficulty
            },
            "difficulty": {
                "HPDrainRate": self._get_difficulty_param("HP", difficulty),
                "CircleSize": self._get_difficulty_param("CS", difficulty),
                "OverallDifficulty": self._get_difficulty_param("OD", difficulty),
                "ApproachRate": self._get_difficulty_param("AR", difficulty),
                "SliderMultiplier": 1.6,
                "SliderTickRate": 1
            },
            "timing_points": [],
            "hit_objects": []
        }
        
        # 设置timing points
        bpm = audio_features["bpm"]
        beatmap["timing_points"].append([
            0,                  # 开始时间
            60000 / bpm,        # 毫秒每拍
            4,                  # 拍子数
            1,                  # 样本集
            0,                  # 样本索引
            100,                # 音量
            1,                  # 不继承
            0                   # 效果
        ])
        
        # 使用模型参数生成hit objects
        beat_times = audio_features["beat_times"]
        params = params.squeeze(0).cpu().numpy()  # 去掉批次维度并转为numpy
        
        for i, beat_time in enumerate(beat_times):
            if i >= len(params):
                break
                
            # 解析参数
            param = params[i]
            obj_type = int(param[0]) % 4  # 0-3
            if obj_type == 0:
                obj_type = 1  # 圆圈
            elif obj_type == 1:
                obj_type = 2  # 滑条
            elif obj_type == 2:
                obj_type = 6  # 滑条+新连击
            else:
                obj_type = 5  # 圆圈+新连击
            
            # 位置
            x = int(((param[1] + 1) / 2) * 512)  # 归一化到0-512
            y = int(((param[2] + 1) / 2) * 384)  # 归一化到0-384
            
            # 时间
            time = int(beat_time * 1000)  # 转换为毫秒
            
            # 添加物件
            if obj_type == 1 or obj_type == 5:  # 圆圈
                hit_object = [x, y, time, obj_type, 0, "0:0:0:0:"]
            else:  # 滑条
                # 简单的直线滑条
                curve_type = "L"
                end_x = int(((param[3] + 1) / 2) * 512)
                end_y = int(((param[4] + 1) / 2) * 384)
                slides = 1
                length = 100
                
                hit_object = [
                    x, y, time, obj_type, 0,
                    f"{curve_type}|{end_x}:{end_y},{slides},{length},0:0|0:0,0:0:0:0:"
                ]
            
            beatmap["hit_objects"].append(hit_object)
        
        return beatmap
    
    def _get_difficulty_param(self, param, difficulty):
        """获取难度参数"""
        params = {
            "HP": {"Easy": 3, "Normal": 5, "Hard": 7, "Expert": 8},
            "CS": {"Easy": 3, "Normal": 4, "Hard": 4.5, "Expert": 5},
            "OD": {"Easy": 3, "Normal": 5, "Hard": 7, "Expert": 9},
            "AR": {"Easy": 4, "Normal": 6, "Hard": 8, "Expert": 9.5}
        }
        
        return params.get(param, {}).get(difficulty, 5)
    
    def _adjust_difficulty(self, beatmap, difficulty):
        """调整谱面难度"""
        # 根据难度调整物件密度
        density_factor = {
            "Easy": 0.5,
            "Normal": 0.7,
            "Hard": 0.9,
            "Expert": 1.0
        }.get(difficulty, 0.7)
        
        # 按密度因子过滤物件
        hit_objects = beatmap["hit_objects"]
        filtered_objects = []
        
        for i, obj in enumerate(hit_objects):
            # 保留第一个和最后一个物件
            if i == 0 or i == len(hit_objects) - 1:
                filtered_objects.append(obj)
                continue
                
            # 根据密度随机保留物件
            if np.random.random() < density_factor:
                filtered_objects.append(obj)
        
        beatmap["hit_objects"] = filtered_objects
        
        return beatmap
    
    def _apply_constraints(self, beatmap):
        """应用谱面约束规则"""
        hit_objects = beatmap["hit_objects"]
        
        # 按时间排序
        hit_objects.sort(key=lambda x: x[2])
        
        # 检查物件间距
        for i in range(len(hit_objects) - 1):
            current = hit_objects[i]
            next_obj = hit_objects[i + 1]
            
            # 获取位置
            current_pos = (current[0], current[1])
            next_pos = (next_obj[0], next_obj[1])
            
            # 计算距离
            distance = ((next_pos[0] - current_pos[0]) ** 2 + 
                        (next_pos[1] - current_pos[1]) ** 2) ** 0.5
            
            # 计算时间差
            time_diff = next_obj[2] - current[2]
            
            # 约束1: 如果时间非常接近但距离太远，调整位置
            if time_diff < 100 and distance > 200:
                # 将下一个物件位置调整为当前位置附近
                direction = np.random.random() * 2 * np.pi
                max_distance = min(200, distance * 0.8)
                new_x = int(current_pos[0] + np.cos(direction) * max_distance)
                new_y = int(current_pos[1] + np.sin(direction) * max_distance)
                
                # 确保在屏幕范围内
                new_x = max(0, min(512, new_x))
                new_y = max(0, min(384, new_y))
                
                next_obj[0] = new_x
                next_obj[1] = new_y
        
        beatmap["hit_objects"] = hit_objects
        return beatmap
    
    def save(self, beatmap, output_path):
        """保存谱面为.osu文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            # 写入文件头
            f.write(f"osu file format v{beatmap['version']}\n\n")
            
            # 写入General区块
            f.write("[General]\n")
            for key, value in beatmap["general"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 写入Metadata区块
            f.write("[Metadata]\n")
            for key, value in beatmap["metadata"].items():
                f.write(f"{key}:{value}\n")
            f.write("\n")
            
            # 写入Difficulty区块
            f.write("[Difficulty]\n")
            for key, value in beatmap["difficulty"].items():
                f.write(f"{key}:{value}\n")
            f.write("\n")
            
            # 写入TimingPoints区块
            f.write("[TimingPoints]\n")
            for point in beatmap["timing_points"]:
                f.write(",".join(map(str, point)) + "\n")
            f.write("\n")
            
            # 写入HitObjects区块
            f.write("[HitObjects]\n")
            for obj in beatmap["hit_objects"]:
                if isinstance(obj, list):
                    f.write(",".join(map(str, obj)) + "\n")
                elif isinstance(obj, dict):
                    # 将字典格式转换为osu格式字符串
                    obj_parts = [
                        str(obj.get("x", 0)),
                        str(obj.get("y", 0)),
                        str(obj.get("time", 0)),
                        str(obj.get("type", 1)),
                        str(obj.get("hitSound", 0))
                    ]
                    
                    # 根据类型添加额外参数
                    obj_type = obj.get("type", 1)
                    if (obj_type & 2) > 0:  # 滑条
                        curve_type = obj.get("curveType", "L")
                        curve_points = "|".join([f"{x}:{y}" for x, y in obj.get("curvePoints", [])])
                        slides = obj.get("slides", 1)
                        length = obj.get("length", 100)
                        obj_parts.append(f"{curve_type}|{curve_points},{slides},{length}")
                    else:  # 圆圈
                        obj_parts.append("0:0:0:0:")
                    
                    f.write(",".join(obj_parts) + "\n")


# ============= 主函数 =============
def main():
    """主函数"""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
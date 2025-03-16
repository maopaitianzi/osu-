#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频可视化器 - 用于在GUI中可视化音频特征
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Dict, List, Optional, Tuple


class AudioVisualizerCanvas(FigureCanvas):
    """音频可视化器画布，用于在GUI中嵌入matplotlib图像"""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        """
        初始化可视化器画布
        
        参数:
            parent: 父级Qt控件
            width: 图像宽度（英寸）
            height: 图像高度（英寸）
            dpi: 分辨率（点/英寸）
        """
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # 设置图像的背景色以匹配osu风格
        self.fig.patch.set_facecolor('#1A1A1A')
        
        # 添加子图
        self.axes = self.fig.add_subplot(111)
        
        # 初始化FigureCanvas
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 自动调整大小
        FigureCanvas.setSizePolicy(
            self, 
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
    
    def clear(self):
        """清除当前图像"""
        self.axes.clear()
        self.draw()


class AudioVisualizer(QtWidgets.QWidget):
    """
    音频可视化器类 - 用于在GUI中显示各种音频分析结果
    
    支持的可视化类型:
    - 波形图 (带节拍标记)
    - 频谱图
    - 梅尔频谱图
    - 色度图
    - BPM图
    - 音量变化图
    """
    
    # 定义信号
    visualization_changed = QtCore.pyqtSignal(str)  # 可视化类型改变信号
    source_changed = QtCore.pyqtSignal(str)  # 音频源改变信号
    
    def __init__(self, parent=None):
        """初始化可视化器"""
        super().__init__(parent)
        self.y = None  # 音频数据
        self.sr = None  # 采样率
        self.features = {}  # 音频特征
        self.current_view = "waveform"  # 当前可视化类型
        self.available_sources = ["original"]  # 可用音频源
        self.current_source = "original"  # 当前音频源
        self.separated_sources = {}  # 分离的音频源
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 顶部工具栏
        toolbar = QtWidgets.QHBoxLayout()
        
        # 可视化类型选择
        self.view_selector = QtWidgets.QComboBox()
        self.view_selector.addItems([
            "波形图 (Waveform)", 
            "频谱图 (Spectrum)", 
            "梅尔频谱图 (Mel Spectrogram)",
            "音高色度图 (Chroma)",
            "音量包络 (Volume)"
        ])
        self.view_selector.currentIndexChanged.connect(self._on_view_changed)
        
        # 源选择下拉框
        self.source_selector = QtWidgets.QComboBox()
        self.source_selector.addItem("原始音频 (Original)")
        self.source_selector.currentIndexChanged.connect(self._on_source_changed)
        
        # 添加标签
        toolbar.addWidget(QtWidgets.QLabel("可视化类型:"))
        toolbar.addWidget(self.view_selector)
        toolbar.addSpacing(20)
        toolbar.addWidget(QtWidgets.QLabel("音频源:"))
        toolbar.addWidget(self.source_selector)
        toolbar.addStretch()
        
        # 导出按钮
        self.export_btn = QtWidgets.QPushButton("导出当前视图")
        self.export_btn.clicked.connect(self.export_current_view)
        toolbar.addWidget(self.export_btn)
        
        # 可视化画布
        self.canvas = AudioVisualizerCanvas(self)
        
        # 信息面板
        self.info_panel = QtWidgets.QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setMaximumHeight(100)
        self.info_panel.setStyleSheet("""
            background-color: #1A1A1A; 
            color: #FFFFFF;
            border: 1px solid #333;
            font-family: "Consolas", monospace;
        """)
        
        # 将各部分添加到主布局
        main_layout.addLayout(toolbar)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(self.info_panel)
        
        # 设置默认内容
        self.update_info_panel()
    
    def set_audio_data(self, y: np.ndarray, sr: int):
        """设置音频数据"""
        self.y = y
        self.sr = sr
        self.update_visualization()
    
    def set_audio_features(self, features: Dict):
        """设置音频特征数据"""
        self.features = features
        
        # 更新音频源选项
        if "available_sources" in features:
            self.available_sources = features["available_sources"]
            self._update_source_selector()
            
        self.update_info_panel()
        self.update_visualization()
    
    def _update_source_selector(self):
        """更新音频源选择器的选项"""
        # 保存当前选中的音频源
        current_source = self.current_source
        
        # 清空并重新添加选项
        self.source_selector.clear()
        
        # 添加原始音频选项
        if "original" in self.available_sources:
            self.source_selector.addItem("原始音频 (Original)", "original")
            
        # 添加分离的音频源选项，使用正确的音频源映射
        source_display_names = {
            "vocals": "人声 (Vocals)",
            "drums": "鼓声 (Drums)",
            "bass": "贝斯 (Bass)",
            "other": "其他乐器 (Other)"
        }
        
        for source in self.available_sources:
            if source != "original" and source in source_display_names:
                self.source_selector.addItem(source_display_names[source], source)
        
        # 尝试恢复之前选中的音频源
        index = self.source_selector.findData(current_source)
        if index >= 0:
            self.source_selector.setCurrentIndex(index)
        else:
            # 如果找不到之前的音频源，默认选择第一个
            if self.source_selector.count() > 0:
                self.current_source = self.source_selector.itemData(0)
    
    def set_active_source(self, source: str):
        """设置当前活跃的音频源"""
        if source != self.current_source:
            self.current_source = source
            
            # 更新选择器
            index = self.source_selector.findData(source)
            if index >= 0:
                self.source_selector.setCurrentIndex(index)
            
            # 更新可视化
            self.update_visualization()
            self.update_info_panel()
            
            # 发送信号
            self.source_changed.emit(source)
    
    def _on_source_changed(self, index: int):
        """当音频源选择改变时触发"""
        if index >= 0:
            source = self.source_selector.itemData(index)
            if source and source != self.current_source:
                self.set_active_source(source)
                
    def update_info_panel(self):
        """更新信息面板"""
        if not self.features:
            self.info_panel.setText("未加载音频数据")
            return
            
        # 获取基本信息
        info_text = []
        
        # 显示当前活跃音频源
        if "active_source" in self.features:
            # 使用正确的音频源映射显示名称
            source_display = {
                "original": "原始音频",
                "vocals": "人声",
                "drums": "鼓声",
                "bass": "贝斯",
                "other": "其他乐器"
            }
            active_source = self.features["active_source"]
            info_text.append(f"活跃音频源: {source_display.get(active_source, active_source)}")
        
        # 显示基本音频信息
        if "sample_rate" in self.features and "duration" in self.features:
            sr = self.features["sample_rate"]
            duration = self.features["duration"]
            info_text.append(f"采样率: {sr} Hz | 时长: {duration:.2f} 秒")
            
        # 显示BPM信息
        if "bpm" in self.features and "beat_source" in self.features:
            bpm = self.features["bpm"]
            source = self.features["beat_source"]
            source_display = {
                "default": "自动检测",
                "librosa": "Librosa算法",
                "tempo": "节拍检测",
                "manual": "手动设置",
                "beatmap": "谱面导入"
            }
            info_text.append(f"BPM: {bpm:.1f} | 来源: {source_display.get(source, source)}")
            
        # 显示当前可视化类型
        view_display = {
            "waveform": "波形图",
            "spectrum": "频谱图",
            "melspectrogram": "梅尔频谱图",
            "chroma": "音高色度图",
            "volume": "音量包络"
        }
        info_text.append(f"当前视图: {view_display.get(self.current_view, self.current_view)}")
        
        # 设置文本
        self.info_panel.setText(" | ".join(info_text))
    
    def update_visualization(self):
        """更新可视化"""
        if self.y is None or self.sr is None:
            return
            
        # 清除画布
        self.canvas.clear()
        
        # 根据当前可视化类型绘制
        if self.current_view == "waveform":
            self._draw_waveform()
        elif self.current_view == "spectrum":
            self._draw_spectrum()
        elif self.current_view == "melspectrogram":
            self._draw_mel_spectrogram()
        elif self.current_view == "chroma":
            self._draw_chroma()
        elif self.current_view == "volume":
            self._draw_volume()
            
        # 绘制
        self.canvas.draw()
    
    def _get_current_audio_data(self):
        """获取当前选中的音频源数据"""
        if self.current_source == "original" or not hasattr(self, 'separated_sources') or not self.separated_sources:
            return self.y
            
        # 如果分离后的音频源可用，使用它
        if self.current_source in self.separated_sources:
            return self.separated_sources[self.current_source]
            
        # 否则使用原始音频
        return self.y
    
    def _on_view_changed(self, index: int):
        """当可视化类型选择改变时触发"""
        view_types = ["waveform", "spectrum", "melspectrogram", "chroma", "volume"]
        if index >= 0 and index < len(view_types):
            self.current_view = view_types[index]
            self.visualization_changed.emit(self.current_view)
            self.update_visualization()
            self.update_info_panel()
    
    def _draw_waveform(self):
        """绘制波形图"""
        if self.y is None:
            return
        
        y = self._get_current_audio_data()
        sr = self.sr
        
        # 创建时间轴
        times = np.linspace(0, len(y) / sr, len(y))
        
        # 绘制波形
        self.canvas.axes.plot(times, y, color='#FF66AA', linewidth=0.5)
        
        # 设置坐标轴范围
        self.canvas.axes.set_xlim(0, len(y) / sr)
        self.canvas.axes.set_ylim(-1.1, 1.1)
        
        # 设置标签
        self.canvas.axes.set_title("音频波形", fontsize=14, color='white')
        self.canvas.axes.set_xlabel("时间 (秒)", fontsize=12, color='white')
        self.canvas.axes.set_ylabel("振幅", fontsize=12, color='white')
        
        # 绘制节拍线
        if self.features and "beat_times" in self.features:
            for beat_time in self.features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
    
    def _draw_spectrum(self):
        """绘制频谱图"""
        if self.y is None:
            return
        
        import librosa
        
        y = self._get_current_audio_data()
        sr = self.sr
        
        # 计算短时傅里叶变换 (STFT)
        D = librosa.stft(y)
        
        # 转换为分贝
        magnitude = np.abs(D)
        power_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # 绘制频谱图
        img = librosa.display.specshow(
            power_db, 
            sr=sr, 
            x_axis='time', 
            y_axis='log',
            ax=self.canvas.axes,
            cmap='magma'
        )
        
        # 设置标签
        self.canvas.axes.set_title("频谱图", fontsize=14, color='white')
        self.canvas.axes.set_xlabel("时间 (秒)", fontsize=12, color='white')
        self.canvas.axes.set_ylabel("频率 (Hz)", fontsize=12, color='white')
        
        # 添加颜色条 - 检查images列表是否为空
        if len(self.canvas.axes.images) > 0:
            cbar = self.canvas.fig.colorbar(
                self.canvas.axes.images[0], 
                ax=self.canvas.axes,
                format="%+2.0f dB"
            )
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.label.set_color('white')
        else:
            # 使用刚刚创建的img对象作为替代
            cbar = self.canvas.fig.colorbar(
                img, 
                ax=self.canvas.axes,
                format="%+2.0f dB"
            )
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.label.set_color('white')
        
        # 绘制节拍线
        if self.features and "beat_times" in self.features:
            for beat_time in self.features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
    
    def _draw_mel_spectrogram(self):
        """绘制梅尔频谱图"""
        # 在方法开头导入librosa，确保所有代码路径都能访问到它
        import librosa
        
        if self.features and "visualization" in self.features and "mel_spec_db" in self.features["visualization"]:
            # 使用预计算的梅尔频谱
            mel_spec_db = np.array(self.features["visualization"]["mel_spec_db"])
            
            librosa.display.specshow(
                mel_spec_db, 
                x_axis='time', 
                y_axis='mel',
                ax=self.canvas.axes,
                cmap='magma'
            )
        elif self.y is not None:
            # 实时计算梅尔频谱
            y = self._get_current_audio_data()
            sr = self.sr
            
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            librosa.display.specshow(
                mel_spec_db, 
                sr=sr, 
                x_axis='time', 
                y_axis='mel',
                ax=self.canvas.axes,
                cmap='magma'
            )
        else:
            return
        
        # 设置标签
        self.canvas.axes.set_title("梅尔频谱图", fontsize=14, color='white')
        self.canvas.axes.set_xlabel("时间 (秒)", fontsize=12, color='white')
        self.canvas.axes.set_ylabel("梅尔频率", fontsize=12, color='white')
        
        # 添加颜色条
        if len(self.canvas.axes.images) > 0:
            cbar = self.canvas.fig.colorbar(
                self.canvas.axes.images[0], 
                ax=self.canvas.axes,
                format="%+2.0f dB"
            )
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.label.set_color('white')
        
        # 绘制节拍线
        if self.features and "beat_times" in self.features:
            for beat_time in self.features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
    
    def _draw_chroma(self):
        """绘制色度图"""
        # 在方法开头导入librosa，确保所有代码路径都能访问到它
        import librosa
        
        if self.features and "visualization" in self.features and "chroma" in self.features["visualization"]:
            # 使用预计算的色度图
            chroma = np.array(self.features["visualization"]["chroma"])
            
            librosa.display.specshow(
                chroma, 
                x_axis='time', 
                y_axis='chroma',
                ax=self.canvas.axes,
                cmap='plasma'
            )
        elif self.y is not None:
            # 实时计算色度图
            y = self._get_current_audio_data()
            sr = self.sr
            
            # 计算色度图
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            librosa.display.specshow(
                chroma, 
                sr=sr, 
                x_axis='time', 
                y_axis='chroma',
                ax=self.canvas.axes,
                cmap='plasma'
            )
        else:
            return
        
        # 设置标签
        self.canvas.axes.set_title("色度图", fontsize=14, color='white')
        self.canvas.axes.set_xlabel("时间 (秒)", fontsize=12, color='white')
        self.canvas.axes.set_ylabel("音高", fontsize=12, color='white')
        self.canvas.axes.set_yticks(np.arange(12))
        self.canvas.axes.set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        
        # 绘制节拍线
        if self.features and "beat_times" in self.features:
            for beat_time in self.features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
    
    def _draw_volume(self):
        """绘制音量/能量变化图"""
        # 在方法开头导入librosa，确保所有代码路径都能访问到它
        import librosa
        
        if self.features and "volume" in self.features:
            volume_data = self.features["volume"]
            times = volume_data["times"]
            rms = volume_data["rms"]
            
            # 绘制音量曲线
            self.canvas.axes.plot(times, rms, color='#FF66AA', linewidth=1.5)
            self.canvas.axes.fill_between(times, 0, rms, color='#FF66AA', alpha=0.3)
            
            # 设置坐标轴范围
            self.canvas.axes.set_xlim(0, max(times))
            self.canvas.axes.set_ylim(0, max(rms) * 1.1)
            
            # 绘制音量变化点
            if "volume_changes" in self.features:
                change_times = self.features["volume_changes"]
                
                for time in change_times:
                    # 找到最接近的索引
                    idx = np.argmin(np.abs(np.array(times) - time))
                    if idx < len(rms):
                        self.canvas.axes.plot(
                            time, rms[idx], 'o', 
                            color='#FFFF00', 
                            markersize=5
                        )
        elif self.y is not None:
            # 实时计算音量
            y = self._get_current_audio_data()
            sr = self.sr
            
            # 计算RMS能量
            rms = librosa.feature.rms(y=y)[0]
            
            # 创建时间轴
            frames = np.arange(len(rms))
            times = librosa.frames_to_time(frames, sr=sr)
            
            # 绘制音量曲线
            self.canvas.axes.plot(times, rms, color='#FF66AA', linewidth=1.5)
            self.canvas.axes.fill_between(times, 0, rms, color='#FF66AA', alpha=0.3)
            
            # 设置坐标轴范围
            self.canvas.axes.set_xlim(0, max(times))
            self.canvas.axes.set_ylim(0, max(rms) * 1.1)
        else:
            return
        
        # 设置标签
        self.canvas.axes.set_title("音量/能量变化", fontsize=14, color='white')
        self.canvas.axes.set_xlabel("时间 (秒)", fontsize=12, color='white')
        self.canvas.axes.set_ylabel("能量", fontsize=12, color='white')
        
        # 绘制节拍线
        if self.features and "beat_times" in self.features:
            for beat_time in self.features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
    
    @property
    def fig(self):
        """获取Figure对象"""
        return self.canvas.fig
    
    def export_current_view(self):
        """导出当前视图为图像文件"""
        # 检查是否有数据可以导出
        if self.y is None:
            QtWidgets.QMessageBox.warning(
                self, "警告", "没有可导出的数据，请先加载音频文件"
            )
            return
        
        # 获取要保存的文件路径
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG图像 (*.png);;JPG图像 (*.jpg);;PDF文档 (*.pdf);;SVG图像 (*.svg)"
        )
        
        if not file_path:
            return  # 用户取消了保存
        
        try:
            # 保存当前图像
            self.canvas.fig.savefig(
                file_path, 
                dpi=300,
                bbox_inches='tight',
                facecolor=self.canvas.fig.get_facecolor(),
                edgecolor='none'
            )
            
            # 显示成功消息
            QtWidgets.QMessageBox.information(
                self, "导出成功", 
                f"图像已成功导出至:\n{file_path}"
            )
            
        except Exception as e:
            # 显示错误消息
            QtWidgets.QMessageBox.critical(
                self, "导出错误", 
                f"导出图像时出错:\n{str(e)}"
            ) 
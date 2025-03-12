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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.audio_features = None
        
        # 创建UI
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # 创建工具栏
        self.toolbar = QtWidgets.QHBoxLayout()
        
        # 可视化类型选择组
        self.viz_type_group = QtWidgets.QButtonGroup(self)
        
        # 波形按钮
        self.waveform_btn = QtWidgets.QRadioButton("波形图")
        self.waveform_btn.setChecked(True)
        self.viz_type_group.addButton(self.waveform_btn, 1)
        
        # 频谱按钮
        self.spectrum_btn = QtWidgets.QRadioButton("频谱图")
        self.viz_type_group.addButton(self.spectrum_btn, 2)
        
        # 梅尔频谱按钮
        self.mel_btn = QtWidgets.QRadioButton("梅尔频谱")
        self.viz_type_group.addButton(self.mel_btn, 3)
        
        # 色度图按钮
        self.chroma_btn = QtWidgets.QRadioButton("色度图")
        self.viz_type_group.addButton(self.chroma_btn, 4)
        
        # 音量变化按钮
        self.volume_btn = QtWidgets.QRadioButton("音量/能量")
        self.viz_type_group.addButton(self.volume_btn, 5)
        
        # 添加到工具栏
        self.toolbar.addWidget(self.waveform_btn)
        self.toolbar.addWidget(self.spectrum_btn)
        self.toolbar.addWidget(self.mel_btn)
        self.toolbar.addWidget(self.chroma_btn)
        self.toolbar.addWidget(self.volume_btn)
        
        # 添加显示特征复选框
        self.show_beats_cb = QtWidgets.QCheckBox("显示节拍")
        self.show_beats_cb.setChecked(True)
        
        self.show_sections_cb = QtWidgets.QCheckBox("显示段落")
        self.show_sections_cb.setChecked(True)
        
        # 添加到工具栏
        self.toolbar.addStretch()
        self.toolbar.addWidget(self.show_beats_cb)
        self.toolbar.addWidget(self.show_sections_cb)
        
        # 添加导出按钮
        self.export_btn = QtWidgets.QPushButton("导出当前视图")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #6666FF;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #8888FF;
            }
            QPushButton:pressed {
                background-color: #4444CC;
            }
        """)
        self.export_btn.clicked.connect(self.export_current_view)
        self.toolbar.addWidget(self.export_btn)
        
        # 创建可视化画布
        self.canvas = AudioVisualizerCanvas(self, width=10, height=4)
        
        # 添加到主布局
        self.layout.addLayout(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # 信息面板
        self.info_panel = QtWidgets.QLabel("未加载音频")
        self.info_panel.setStyleSheet("""
            background-color: #333333;
            color: #FFFFFF;
            padding: 5px;
            border-radius: 3px;
        """)
        
        self.layout.addWidget(self.info_panel)
        
        # 连接信号
        self.viz_type_group.buttonClicked.connect(self.update_visualization)
        self.show_beats_cb.stateChanged.connect(self.update_visualization)
        self.show_sections_cb.stateChanged.connect(self.update_visualization)
    
    def set_audio_data(self, y: np.ndarray, sr: int):
        """设置音频数据"""
        self.audio_data = {"y": y, "sr": sr}
        self.update_visualization()
    
    def set_audio_features(self, features: Dict):
        """设置音频特征数据"""
        self.audio_features = features
        self.update_info_panel()
        self.update_visualization()
    
    def update_info_panel(self):
        """更新信息面板"""
        if self.audio_features is None:
            self.info_panel.setText("未加载音频")
            return
        
        info_text = ""
        
        # 基本信息
        if "duration" in self.audio_features:
            duration_mins = int(self.audio_features["duration"] // 60)
            duration_secs = int(self.audio_features["duration"] % 60)
            info_text += f"时长: {duration_mins}分{duration_secs}秒 | "
        
        # BPM
        if "bpm" in self.audio_features:
            info_text += f"BPM: {self.audio_features['bpm']} | "
        
        # 节拍数
        if "beat_times" in self.audio_features:
            info_text += f"节拍数: {len(self.audio_features['beat_times'])} | "
        
        # 规律性
        if "beat_regularity" in self.audio_features:
            regularity = self.audio_features["beat_regularity"] * 100
            info_text += f"节奏规律性: {regularity:.1f}% | "
        
        # 段落数
        if "sections" in self.audio_features:
            info_text += f"段落数: {len(self.audio_features['sections'])} | "
        
        # 过渡点数
        if "transitions" in self.audio_features:
            info_text += f"过渡点数: {len(self.audio_features['transitions'])}"
        
        self.info_panel.setText(info_text)
    
    def update_visualization(self):
        """根据选择的类型更新可视化"""
        # 清除当前图像
        self.canvas.axes.clear()
        
        # 获取当前选中的可视化类型
        viz_type = self.viz_type_group.checkedId()
        
        # 如果没有数据可视化，则显示提示
        if self.audio_data is None and self.audio_features is None:
            self.canvas.axes.text(
                0.5, 0.5, "请先加载音频文件进行分析", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='#FF66AA'
            )
            self.canvas.draw()
            return
        
        # 根据不同的可视化类型进行绘制
        if viz_type == 1:  # 波形图
            self._draw_waveform()
        elif viz_type == 2:  # 频谱图
            self._draw_spectrum()
        elif viz_type == 3:  # 梅尔频谱
            self._draw_mel_spectrogram()
        elif viz_type == 4:  # 色度图
            self._draw_chroma()
        elif viz_type == 5:  # 音量变化
            self._draw_volume()
        
        # 绘制
        self.canvas.draw()
    
    def _draw_waveform(self):
        """绘制波形图"""
        if self.audio_data is None:
            return
        
        y = self.audio_data["y"]
        sr = self.audio_data["sr"]
        
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
        if self.show_beats_cb.isChecked() and self.audio_features and "beat_times" in self.audio_features:
            for beat_time in self.audio_features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
        
        # 绘制段落标记
        if self.show_sections_cb.isChecked() and self.audio_features and "sections" in self.audio_features:
            for section_time in self.audio_features["sections"]:
                self.canvas.axes.axvline(x=section_time, color='#00FFFF', alpha=0.5, linewidth=1.0)
        
        # 绘制过渡点
        if self.audio_features and "transitions" in self.audio_features:
            for transition_time in self.audio_features["transitions"]:
                self.canvas.axes.axvline(x=transition_time, color='#FFFF00', alpha=0.5, linewidth=1.0)
    
    def _draw_spectrum(self):
        """绘制频谱图"""
        if self.audio_data is None:
            return
        
        import librosa
        
        y = self.audio_data["y"]
        sr = self.audio_data["sr"]
        
        # 计算短时傅里叶变换 (STFT)
        D = librosa.stft(y)
        
        # 转换为分贝
        magnitude = np.abs(D)
        power_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # 绘制频谱图
        librosa.display.specshow(
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
        
        # 添加颜色条
        cbar = self.canvas.fig.colorbar(
            self.canvas.axes.images[0], 
            ax=self.canvas.axes,
            format="%+2.0f dB"
        )
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        
        # 绘制节拍线
        if self.show_beats_cb.isChecked() and self.audio_features and "beat_times" in self.audio_features:
            for beat_time in self.audio_features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
        
        # 绘制段落标记
        if self.show_sections_cb.isChecked() and self.audio_features and "sections" in self.audio_features:
            for section_time in self.audio_features["sections"]:
                self.canvas.axes.axvline(x=section_time, color='#00FFFF', alpha=0.5, linewidth=1.0)
    
    def _draw_mel_spectrogram(self):
        """绘制梅尔频谱图"""
        if self.audio_features and "visualization" in self.audio_features and "mel_spec_db" in self.audio_features["visualization"]:
            # 使用预计算的梅尔频谱
            mel_spec_db = np.array(self.audio_features["visualization"]["mel_spec_db"])
            
            librosa.display.specshow(
                mel_spec_db, 
                x_axis='time', 
                y_axis='mel',
                ax=self.canvas.axes,
                cmap='magma'
            )
        elif self.audio_data is not None:
            # 实时计算梅尔频谱
            import librosa
            
            y = self.audio_data["y"]
            sr = self.audio_data["sr"]
            
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
        cbar = self.canvas.fig.colorbar(
            self.canvas.axes.images[0], 
            ax=self.canvas.axes,
            format="%+2.0f dB"
        )
        
        # 绘制节拍线
        if self.show_beats_cb.isChecked() and self.audio_features and "beat_times" in self.audio_features:
            for beat_time in self.audio_features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
        
        # 绘制段落标记
        if self.show_sections_cb.isChecked() and self.audio_features and "sections" in self.audio_features:
            for section_time in self.audio_features["sections"]:
                self.canvas.axes.axvline(x=section_time, color='#00FFFF', alpha=0.5, linewidth=1.0)
    
    def _draw_chroma(self):
        """绘制色度图"""
        if self.audio_features and "visualization" in self.audio_features and "chroma" in self.audio_features["visualization"]:
            # 使用预计算的色度图
            chroma = np.array(self.audio_features["visualization"]["chroma"])
            
            librosa.display.specshow(
                chroma, 
                x_axis='time', 
                y_axis='chroma',
                ax=self.canvas.axes,
                cmap='plasma'
            )
        elif self.audio_data is not None:
            # 实时计算色度图
            import librosa
            
            y = self.audio_data["y"]
            sr = self.audio_data["sr"]
            
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
        if self.show_beats_cb.isChecked() and self.audio_features and "beat_times" in self.audio_features:
            for beat_time in self.audio_features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
        
        # 绘制段落标记
        if self.show_sections_cb.isChecked() and self.audio_features and "sections" in self.audio_features:
            for section_time in self.audio_features["sections"]:
                self.canvas.axes.axvline(x=section_time, color='#00FFFF', alpha=0.5, linewidth=1.0)
    
    def _draw_volume(self):
        """绘制音量/能量变化图"""
        if self.audio_features and "volume" in self.audio_features:
            volume_data = self.audio_features["volume"]
            times = volume_data["times"]
            rms = volume_data["rms"]
            
            # 绘制音量曲线
            self.canvas.axes.plot(times, rms, color='#FF66AA', linewidth=1.5)
            self.canvas.axes.fill_between(times, 0, rms, color='#FF66AA', alpha=0.3)
            
            # 设置坐标轴范围
            self.canvas.axes.set_xlim(0, max(times))
            self.canvas.axes.set_ylim(0, max(rms) * 1.1)
            
            # 绘制音量变化点
            if "volume_changes" in self.audio_features:
                change_times = self.audio_features["volume_changes"]
                
                for time in change_times:
                    # 找到最接近的索引
                    idx = np.argmin(np.abs(np.array(times) - time))
                    if idx < len(rms):
                        self.canvas.axes.plot(
                            time, rms[idx], 'o', 
                            color='#FFFF00', 
                            markersize=5
                        )
        elif self.audio_data is not None:
            # 实时计算音量
            import librosa
            
            y = self.audio_data["y"]
            sr = self.audio_data["sr"]
            
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
        if self.show_beats_cb.isChecked() and self.audio_features and "beat_times" in self.audio_features:
            for beat_time in self.audio_features["beat_times"]:
                self.canvas.axes.axvline(x=beat_time, color='white', alpha=0.3, linewidth=0.5)
        
        # 绘制段落标记
        if self.show_sections_cb.isChecked() and self.audio_features and "sections" in self.audio_features:
            for section_time in self.audio_features["sections"]:
                self.canvas.axes.axvline(x=section_time, color='#00FFFF', alpha=0.5, linewidth=1.0)
        
        # 绘制过渡点
        if self.audio_features and "transitions" in self.audio_features:
            for transition_time in self.audio_features["transitions"]:
                self.canvas.axes.axvline(x=transition_time, color='#FFFF00', alpha=0.5, linewidth=1.0)
    
    @property
    def fig(self):
        """获取Figure对象"""
        return self.canvas.fig
    
    def export_current_view(self):
        """导出当前视图为图像文件"""
        # 检查是否有数据可以导出
        if self.audio_data is None and self.audio_features is None:
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
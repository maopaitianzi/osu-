#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
osu!风格的谱面生成器主窗口
"""

import os
import sys
import json
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# 导入PyTorch的GPU支持 - 使用try/except处理导入错误
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: 未找到PyTorch库，GPU加速将不可用。如需GPU加速，请安装PyTorch: pip install torch")

# 导入音频分析模块
from audio.analyzer import AudioAnalyzer
from audio.visualizer import AudioVisualizer
# 导入谱面分析模块
from beatmap.analyzer import BeatmapAnalyzer

# 在import部分添加以下导入
from gui.training_thread import TrainingThread

# 导入视频生成模块
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shipingshengcheng"))
    from osu_to_vsrg import OsuParser, VSRGRenderer, create_vsrg_video
    VIDEO_GEN_AVAILABLE = True
except ImportError:
    VIDEO_GEN_AVAILABLE = False
    print("警告: 未找到视频生成模块，视频生成功能将不可用")


class OsuStyleMainWindow(QtWidgets.QMainWindow):
    """osu!风格的主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("osu!谱面生成器")
        self.setMinimumSize(1200, 800)
        
        # 设置应用图标和样式
        self.setup_appearance()
        
        # 初始化音频分析器
        self.audio_analyzer = AudioAnalyzer()
        
        # 设置人声分离默认启用
        self.audio_analyzer.set_use_source_separation(True)
        
        # 初始化谱面分析器
        self.beatmap_analyzer = BeatmapAnalyzer()
        
        # 连接信号
        self.audio_analyzer.analysis_progress.connect(self.update_analysis_progress)
        self.audio_analyzer.analysis_complete.connect(self.handle_analysis_complete)
        self.audio_analyzer.analysis_error.connect(self.handle_analysis_error)
        
        # 连接谱面分析器信号
        self.beatmap_analyzer.analysis_progress.connect(self.update_beatmap_analysis_progress)
        self.beatmap_analyzer.analysis_complete.connect(self.handle_beatmap_analysis_complete)
        self.beatmap_analyzer.analysis_error.connect(self.handle_beatmap_analysis_error)
        
        # 初始化UI
        self.init_ui()
        
        # 记住上次使用的导出格式
        self.last_export_format = "JSON文件 (*.json)"
        
        # 添加对init_training_signals的调用
        self.init_training_signals()

    def setup_appearance(self):
        """设置外观样式"""
        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OSUMAP.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        
        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                color: #000000;
            }
            QGroupBox {
                background-color: #F0F0F0;
                border: 2px solid #FF66AA;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #FF66AA;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #FF66AA;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #FF99CC;
            }
            QPushButton:pressed {
                background-color: #CC5588;
            }
            QLabel {
                color: #000000;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 5px;
                color: #000000;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
                background-color: #FF66AA;
            }
            QTabWidget::pane {
                border: 2px solid #FF66AA;
                border-radius: 5px;
                background-color: #F0F0F0;
            }
            QTabBar::tab {
                background-color: #333333;
                color: white;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #FF66AA;
            }
            QTabBar::tab:hover:!selected {
                background-color: #444444;
            }
            QSlider::groove:horizontal {
                border: 1px solid #CCCCCC;
                height: 8px;
                background: #F0F0F0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #FF66AA;
                border: none;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #FF99CC;
            }
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                background-color: #F0F0F0;
                color: #000000;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF66AA;
                border-radius: 4px;
            }
        """)
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建标题标签
        title_label = QtWidgets.QLabel("osu!谱面生成器")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #FF66AA;
            margin-bottom: 20px;
        """)
        main_layout.addWidget(title_label)
        
        # 创建选项卡控件
        tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 创建"音频分析"选项卡
        generate_tab = QtWidgets.QWidget()
        tab_widget.addTab(generate_tab, "音频分析")
        
        # 创建"谱面分析"选项卡
        beatmap_analysis_tab = QtWidgets.QWidget()
        tab_widget.addTab(beatmap_analysis_tab, "谱面分析")
        
        # 创建"谱面生成"选项卡
        beatmap_generate_tab = QtWidgets.QWidget()
        tab_widget.addTab(beatmap_generate_tab, "谱面生成")
        
        # 创建"谱面预览"选项卡
        preview_tab = QtWidgets.QWidget()
        tab_widget.addTab(preview_tab, "谱面预览")
        
        # 创建"数据集处理"选项卡
        dataset_tab = QtWidgets.QWidget()
        tab_widget.addTab(dataset_tab, "数据集处理")
        
        # 创建"模型训练"选项卡
        model_training_tab = QtWidgets.QWidget()
        tab_widget.addTab(model_training_tab, "模型训练")
        
        # 创建"字幕处理"选项卡
        subtitle_tab = QtWidgets.QWidget()
        tab_widget.addTab(subtitle_tab, "字幕处理")
        
        # 创建"视频生成"选项卡
        video_generate_tab = QtWidgets.QWidget()
        tab_widget.addTab(video_generate_tab, "视频生成")
        
        # 创建"设置"选项卡
        settings_tab = QtWidgets.QWidget()
        tab_widget.addTab(settings_tab, "设置")
        
        # 设置"生成谱面"选项卡的布局
        generate_layout = QtWidgets.QVBoxLayout(generate_tab)
        generate_layout.setSpacing(15)
        
        # 文件选择部分
        file_group = QtWidgets.QGroupBox("音频文件")
        file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        file_layout = QtWidgets.QVBoxLayout(file_group)
        
        # 文件选择行
        file_row = QtWidgets.QHBoxLayout()
        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setPlaceholderText("请选择.mp3或.wav文件...")
        
        browse_btn = QtWidgets.QPushButton("浏览")
        try:
            browse_btn.setIcon(QtGui.QIcon("gui/resources/folder_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        browse_btn.clicked.connect(self.browse_audio)
        
        file_row.addWidget(self.file_path, 3)
        file_row.addWidget(browse_btn, 1)
        
        file_layout.addLayout(file_row)
        
        # BPM设置行
        bpm_row = QtWidgets.QHBoxLayout()
        
        # 自动检测BPM选项
        self.auto_bpm_rb = QtWidgets.QRadioButton("自动检测BPM")
        self.auto_bpm_rb.setChecked(True)
        bpm_row.addWidget(self.auto_bpm_rb)
        
        # 手动设置BPM选项
        self.manual_bpm_rb = QtWidgets.QRadioButton("手动设置BPM:")
        bpm_row.addWidget(self.manual_bpm_rb)
        
        # BPM输入框
        self.bpm_input = QtWidgets.QDoubleSpinBox()
        self.bpm_input.setRange(20, 300)
        self.bpm_input.setValue(120)
        self.bpm_input.setSingleStep(0.1)
        self.bpm_input.setDecimals(1)
        self.bpm_input.setEnabled(False)
        bpm_row.addWidget(self.bpm_input)
        
        # 从谱面导入BPM
        self.import_bpm_btn = QtWidgets.QPushButton("从谱面导入BPM")
        self.import_bpm_btn.setEnabled(False)
        self.import_bpm_btn.clicked.connect(self.import_bpm_from_beatmap)
        bpm_row.addWidget(self.import_bpm_btn)
        
        # 添加一些弹性空间
        bpm_row.addStretch(1)
        
        # 连接按钮组的信号
        self.auto_bpm_rb.toggled.connect(self.toggle_bpm_mode)
        self.manual_bpm_rb.toggled.connect(self.toggle_bpm_mode)
        
        file_layout.addLayout(bpm_row)
        
        # GPU设置选项
        gpu_row = QtWidgets.QHBoxLayout()
        
        # 检测GPU是否可用
        gpu_available = 'TORCH_AVAILABLE' in globals() and TORCH_AVAILABLE and torch.cuda.is_available()
        
        # 添加GPU加速复选框
        self.use_gpu_cb = QtWidgets.QCheckBox("使用GPU加速分析")
        self.use_gpu_cb.setChecked(gpu_available)
        self.use_gpu_cb.setEnabled(gpu_available)
        
        if not gpu_available:
            if 'TORCH_AVAILABLE' not in globals() or not TORCH_AVAILABLE:
                gpu_status = "未安装PyTorch库，GPU加速不可用"
            else:
                gpu_status = "未检测到可用GPU"
            self.use_gpu_cb.setToolTip(gpu_status)
        else:
            self.use_gpu_cb.setToolTip(f"检测到 {torch.cuda.device_count()} 个可用GPU")
        
        gpu_row.addWidget(self.use_gpu_cb)
        
        # 自动导出复选框
        self.auto_export_cb = QtWidgets.QCheckBox("自动导出分析结果")
        self.auto_export_cb.setChecked(False)
        gpu_row.addWidget(self.auto_export_cb)
        
        # 添加弹性空间
        gpu_row.addStretch(1)
        
        file_layout.addLayout(gpu_row)
        
        # 添加输出设置
        output_row = QtWidgets.QHBoxLayout()
        output_row.setSpacing(10)
        
        output_label = QtWidgets.QLabel("输出目录:")
        
        self.output_path = QtWidgets.QLineEdit()
        self.output_path.setPlaceholderText("默认：与音频文件相同目录")
        
        browse_output_btn = QtWidgets.QPushButton("浏览")
        browse_output_btn.setFixedWidth(100)
        browse_output_btn.clicked.connect(self.browse_output_directory)
        
        output_row.addWidget(output_label)
        output_row.addWidget(self.output_path, 3)
        output_row.addWidget(browse_output_btn)
        
        file_layout.addLayout(output_row)
        
        generate_layout.addWidget(file_group)
        
        # 操作区域
        actions_layout = QtWidgets.QHBoxLayout()
        
        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        
        # 操作按钮
        self.analyze_btn = QtWidgets.QPushButton("分析音频")
        try:
            self.analyze_btn.setIcon(QtGui.QIcon("gui/resources/analyze_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        self.analyze_btn.clicked.connect(self.analyze_audio)
        
        # 删除生成谱面按钮
        
        preview_btn = QtWidgets.QPushButton("预览谱面")
        try:
            preview_btn.setIcon(QtGui.QIcon("gui/resources/preview_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        preview_btn.clicked.connect(self.preview_beatmap)
        
        # 导出按钮
        export_btn = QtWidgets.QPushButton("导出分析")
        try:
            export_btn.setIcon(QtGui.QIcon("gui/resources/export_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        export_btn.clicked.connect(self.export_analysis)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #6666FF;
            }
            QPushButton:hover {
                background-color: #8888FF;
            }
            QPushButton:pressed {
                background-color: #4444CC;
            }
        """)
        
        # 添加到布局
        actions_layout.addWidget(self.progress_bar, 3)
        actions_layout.addWidget(self.analyze_btn, 1)
        actions_layout.addWidget(preview_btn, 1)
        actions_layout.addWidget(export_btn, 1)
        
        generate_layout.addLayout(actions_layout)
        
        # 状态区域
        status_layout = QtWidgets.QHBoxLayout()
        
        self.status_label = QtWidgets.QLabel("准备就绪")
        self.status_label.setStyleSheet("color: #666666;")
        
        status_layout.addWidget(self.status_label)
        
        generate_layout.addLayout(status_layout)
        
        # 设置"谱面分析"选项卡的布局
        beatmap_analysis_layout = QtWidgets.QVBoxLayout(beatmap_analysis_tab)
        beatmap_analysis_layout.setSpacing(15)
        
        # 谱面文件选择部分
        beatmap_file_group = QtWidgets.QGroupBox("谱面文件")
        beatmap_file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        beatmap_file_layout = QtWidgets.QHBoxLayout(beatmap_file_group)
        
        self.beatmap_file_path = QtWidgets.QLineEdit()
        self.beatmap_file_path.setPlaceholderText("请选择.osu谱面文件...")
        
        browse_beatmap_btn = QtWidgets.QPushButton("浏览")
        try:
            browse_beatmap_btn.setIcon(QtGui.QIcon("gui/resources/folder_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        browse_beatmap_btn.clicked.connect(self.browse_beatmap)
        
        beatmap_file_layout.addWidget(self.beatmap_file_path, 3)
        beatmap_file_layout.addWidget(browse_beatmap_btn, 1)
        
        beatmap_analysis_layout.addWidget(beatmap_file_group)
        
        # 谱面分析操作按钮区域
        beatmap_actions_layout = QtWidgets.QHBoxLayout()
        
        self.beatmap_progress_bar = QtWidgets.QProgressBar()
        self.beatmap_progress_bar.setRange(0, 100)
        self.beatmap_progress_bar.setValue(0)
        
        analyze_beatmap_btn = QtWidgets.QPushButton("分析谱面")
        try:
            analyze_beatmap_btn.setIcon(QtGui.QIcon("gui/resources/analyze_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        analyze_beatmap_btn.clicked.connect(self.analyze_beatmap)
        
        export_beatmap_analysis_btn = QtWidgets.QPushButton("导出分析")
        try:
            export_beatmap_analysis_btn.setIcon(QtGui.QIcon("gui/resources/export_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        export_beatmap_analysis_btn.clicked.connect(self.export_beatmap_analysis)
        
        beatmap_actions_layout.addWidget(self.beatmap_progress_bar, 3)
        beatmap_actions_layout.addWidget(analyze_beatmap_btn, 1)
        beatmap_actions_layout.addWidget(export_beatmap_analysis_btn, 1)
        
        beatmap_analysis_layout.addLayout(beatmap_actions_layout)
        
        # 结果显示区域 - 选项卡
        beatmap_results_tabs = QtWidgets.QTabWidget()
        beatmap_analysis_layout.addWidget(beatmap_results_tabs)
        
        # 谱面概要选项卡
        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)
        self.beatmap_summary_text = QtWidgets.QTextEdit()
        self.beatmap_summary_text.setReadOnly(True)
        summary_layout.addWidget(self.beatmap_summary_text)
        beatmap_results_tabs.addTab(summary_tab, "谱面概要")
        
        # 难度分析选项卡
        difficulty_tab = QtWidgets.QWidget()
        difficulty_layout = QtWidgets.QVBoxLayout(difficulty_tab)
        self.difficulty_analysis_text = QtWidgets.QTextEdit()
        self.difficulty_analysis_text.setReadOnly(True)
        difficulty_layout.addWidget(self.difficulty_analysis_text)
        beatmap_results_tabs.addTab(difficulty_tab, "难度分析")
        
        # 物件分布选项卡
        distribution_tab = QtWidgets.QWidget()
        distribution_layout = QtWidgets.QVBoxLayout(distribution_tab)
        self.distribution_widget = QtWidgets.QWidget()
        distribution_layout.addWidget(self.distribution_widget)
        beatmap_results_tabs.addTab(distribution_tab, "物件分布")
        
        # 热图选项卡
        heatmap_tab = QtWidgets.QWidget()
        heatmap_layout = QtWidgets.QVBoxLayout(heatmap_tab)
        self.heatmap_widget = QtWidgets.QWidget()
        heatmap_layout.addWidget(self.heatmap_widget)
        beatmap_results_tabs.addTab(heatmap_tab, "热图分析")
        
        # 模式识别选项卡
        pattern_tab = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_tab)
        self.pattern_analysis_text = QtWidgets.QTextEdit()
        self.pattern_analysis_text.setReadOnly(True)
        pattern_layout.addWidget(self.pattern_analysis_text)
        beatmap_results_tabs.addTab(pattern_tab, "模式识别")
        
        # 状态区域
        beatmap_status_layout = QtWidgets.QHBoxLayout()
        self.beatmap_status_label = QtWidgets.QLabel("请选择要分析的谱面文件")
        self.beatmap_status_label.setStyleSheet("color: #666666;")
        beatmap_status_layout.addWidget(self.beatmap_status_label)
        beatmap_analysis_layout.addLayout(beatmap_status_layout)
        
        # 设置预览选项卡内容
        preview_layout = QtWidgets.QVBoxLayout(preview_tab)
        
        # 创建音频可视化器但默认不显示
        self.audio_visualizer = AudioVisualizer()
        self.audio_visualizer.setVisible(False)  # 默认隐藏可视化器
        preview_layout.addWidget(self.audio_visualizer)
        
        # 添加一个提示标签，当可视化关闭时显示
        self.visualization_disabled_label = QtWidgets.QLabel("音频可视化功能已禁用。\n您可以在设置选项卡中启用此功能。")
        self.visualization_disabled_label.setAlignment(QtCore.Qt.AlignCenter)
        self.visualization_disabled_label.setStyleSheet("""
            font-size: 16px;
            color: #666666;
            margin: 50px;
        """)
        preview_layout.addWidget(self.visualization_disabled_label)
        
        preview_controls_layout = QtWidgets.QHBoxLayout()
        
        play_btn = QtWidgets.QPushButton("播放")
        try:
            play_btn.setIcon(QtGui.QIcon("gui/resources/play_icon.png"))
        except:
            pass
        
        pause_btn = QtWidgets.QPushButton("暂停")
        try:
            pause_btn.setIcon(QtGui.QIcon("gui/resources/pause_icon.png"))
        except:
            pass
        
        stop_btn = QtWidgets.QPushButton("停止")
        try:
            stop_btn.setIcon(QtGui.QIcon("gui/resources/stop_icon.png"))
        except:
            pass
        
        self.playback_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        
        preview_controls_layout.addWidget(play_btn)
        preview_controls_layout.addWidget(pause_btn)
        preview_controls_layout.addWidget(stop_btn)
        preview_controls_layout.addWidget(self.playback_slider, 2)
        
        preview_layout.addLayout(preview_controls_layout)
        
        # 设置数据集处理选项卡内容
        dataset_layout = QtWidgets.QVBoxLayout(dataset_tab)
        
        # 文件夹选择部分
        dataset_folder_group = QtWidgets.QGroupBox("谱面文件夹")
        dataset_folder_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        dataset_folder_layout = QtWidgets.QHBoxLayout(dataset_folder_group)
        
        self.dataset_folder_path = QtWidgets.QLineEdit()
        self.dataset_folder_path.setPlaceholderText("请选择包含谱面文件的文件夹...")
        
        browse_dataset_btn = QtWidgets.QPushButton("浏览")
        try:
            browse_dataset_btn.setIcon(QtGui.QIcon("gui/resources/folder_icon.png"))
        except:
            pass
        browse_dataset_btn.clicked.connect(self.browse_dataset_folder)
        
        dataset_folder_layout.addWidget(self.dataset_folder_path, 3)
        dataset_folder_layout.addWidget(browse_dataset_btn, 1)
        
        dataset_layout.addWidget(dataset_folder_group)
        
        # 数据集参数设置
        dataset_params_group = QtWidgets.QGroupBox("数据集参数")
        dataset_params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        dataset_params_layout = QtWidgets.QGridLayout(dataset_params_group)
        
        # 模式选择
        mode_label = QtWidgets.QLabel("游戏模式:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["std", "taiko", "catch", "mania"])
        dataset_params_layout.addWidget(mode_label, 0, 0)
        dataset_params_layout.addWidget(self.mode_combo, 0, 1)
        
        # 难度选择
        difficulty_label = QtWidgets.QLabel("难度选择:")
        self.difficulty_combo = QtWidgets.QComboBox()
        self.difficulty_combo.addItems(["所有难度", "Easy", "Normal", "Hard", "Insane", "Expert", "Expert+"])
        dataset_params_layout.addWidget(difficulty_label, 1, 0)
        dataset_params_layout.addWidget(self.difficulty_combo, 1, 1)
        
        # 文件数量限制
        files_limit_label = QtWidgets.QLabel("文件数量限制:")
        self.files_limit_spin = QtWidgets.QSpinBox()
        self.files_limit_spin.setRange(1, 10000)
        self.files_limit_spin.setValue(100)
        dataset_params_layout.addWidget(files_limit_label, 2, 0)
        dataset_params_layout.addWidget(self.files_limit_spin, 2, 1)
        
        # 数据集分割设置
        dataset_split_label = QtWidgets.QLabel("数据集分割:")
        dataset_split_layout = QtWidgets.QHBoxLayout()
        
        self.enable_split_check = QtWidgets.QCheckBox("启用分割")
        dataset_split_layout.addWidget(self.enable_split_check)
        
        train_label = QtWidgets.QLabel("训练集:")
        self.train_percent_spin = QtWidgets.QSpinBox()
        self.train_percent_spin.setRange(10, 90)
        self.train_percent_spin.setValue(70)
        self.train_percent_spin.setSuffix("%")
        dataset_split_layout.addWidget(train_label)
        dataset_split_layout.addWidget(self.train_percent_spin)
        
        val_label = QtWidgets.QLabel("验证集:")
        self.val_percent_spin = QtWidgets.QSpinBox()
        self.val_percent_spin.setRange(5, 30)
        self.val_percent_spin.setValue(15)
        self.val_percent_spin.setSuffix("%")
        dataset_split_layout.addWidget(val_label)
        dataset_split_layout.addWidget(self.val_percent_spin)
        
        test_label = QtWidgets.QLabel("测试集:")
        self.test_percent_spin = QtWidgets.QSpinBox()
        self.test_percent_spin.setRange(5, 30)
        self.test_percent_spin.setValue(15)
        self.test_percent_spin.setSuffix("%")
        dataset_split_layout.addWidget(test_label)
        dataset_split_layout.addWidget(self.test_percent_spin)
        
        # 确保百分比总和为100%
        self.train_percent_spin.valueChanged.connect(self.adjust_split_percentages)
        self.val_percent_spin.valueChanged.connect(self.adjust_split_percentages)
        self.test_percent_spin.valueChanged.connect(self.adjust_split_percentages)
        
        dataset_params_layout.addWidget(dataset_split_label, 3, 0)
        dataset_params_layout.addLayout(dataset_split_layout, 3, 1)
        
        # 分割方式选择
        split_method_label = QtWidgets.QLabel("分割方式:")
        self.split_method_combo = QtWidgets.QComboBox()
        self.split_method_combo.addItems(["按文件夹分割", "随机分割"])
        self.split_method_combo.setToolTip("按文件夹分割会保持同一文件夹中的谱面在同一数据集中")
        
        dataset_params_layout.addWidget(split_method_label, 4, 0)
        dataset_params_layout.addWidget(self.split_method_combo, 4, 1)
        
        # 批处理设置
        batch_size_label = QtWidgets.QLabel("批处理大小:")
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(10)
        self.batch_size_spin.setToolTip("每批处理的文件数量，较小的值可以减少内存占用但会增加处理时间")
        
        dataset_params_layout.addWidget(batch_size_label, 5, 0)
        dataset_params_layout.addWidget(self.batch_size_spin, 5, 1)
        
        # 添加GPU加速选项
        gpu_accel_layout = QtWidgets.QHBoxLayout()
        self.dataset_use_gpu_checkbox = QtWidgets.QCheckBox("使用GPU加速处理")
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.dataset_use_gpu_checkbox.setEnabled(gpu_available)
        self.dataset_use_gpu_checkbox.setChecked(gpu_available)
        if not gpu_available:
            if not TORCH_AVAILABLE:
                gpu_status = "未安装PyTorch库，GPU加速不可用"
            else:
                gpu_status = "未检测到可用GPU"
            self.dataset_use_gpu_checkbox.setToolTip(gpu_status)
        else:
            self.dataset_use_gpu_checkbox.setToolTip(f"检测到 {torch.cuda.device_count()} 个可用GPU")
        
        # GPU设备选择
        gpu_device_label = QtWidgets.QLabel("GPU设备:")
        self.dataset_gpu_device_combo = QtWidgets.QComboBox()
        if gpu_available:
            self.dataset_gpu_device_combo.addItems([f"GPU {i}: {torch.cuda.get_device_name(i)}" 
                                         for i in range(torch.cuda.device_count())])
        else:
            self.dataset_gpu_device_combo.addItem("无可用设备")
        self.dataset_gpu_device_combo.setEnabled(gpu_available and self.dataset_use_gpu_checkbox.isChecked())
        self.dataset_use_gpu_checkbox.toggled.connect(lambda state: self.dataset_gpu_device_combo.setEnabled(state))
        
        gpu_accel_layout.addWidget(self.dataset_use_gpu_checkbox)
        gpu_accel_layout.addWidget(gpu_device_label)
        gpu_accel_layout.addWidget(self.dataset_gpu_device_combo, 1)
        
        dataset_params_layout.addWidget(QtWidgets.QLabel("GPU加速:"), 6, 0)
        dataset_params_layout.addLayout(gpu_accel_layout, 6, 1)
        
        # 输出文件夹设置
        output_folder_label = QtWidgets.QLabel("输出文件夹:")
        self.dataset_output_path = QtWidgets.QLineEdit()
        self.dataset_output_path.setPlaceholderText("请选择数据集输出文件夹...")
        
        browse_output_btn = QtWidgets.QPushButton("浏览")
        try:
            browse_output_btn.setIcon(QtGui.QIcon("gui/resources/folder_icon.png"))
        except:
            pass
        browse_output_btn.clicked.connect(self.browse_dataset_output)
        
        output_folder_layout = QtWidgets.QHBoxLayout()
        output_folder_layout.addWidget(self.dataset_output_path, 3)
        output_folder_layout.addWidget(browse_output_btn, 1)
        
        dataset_params_layout.addWidget(output_folder_label, 7, 0)
        dataset_params_layout.addLayout(output_folder_layout, 7, 1)
        
        dataset_layout.addWidget(dataset_params_group)
        
        # 扫描与处理按钮
        dataset_actions_group = QtWidgets.QGroupBox("处理操作")
        dataset_actions_layout = QtWidgets.QVBoxLayout(dataset_actions_group)
        
        # 处理进度条
        self.dataset_progress_bar = QtWidgets.QProgressBar()
        self.dataset_progress_bar.setRange(0, 100)
        self.dataset_progress_bar.setValue(0)
        
        # 操作按钮布局
        dataset_buttons_layout = QtWidgets.QHBoxLayout()
        
        scan_btn = QtWidgets.QPushButton("扫描文件夹")
        scan_btn.clicked.connect(self.scan_dataset_folder)
        
        process_btn = QtWidgets.QPushButton("处理数据集")
        process_btn.clicked.connect(self.process_dataset)
        
        export_btn = QtWidgets.QPushButton("导出数据集")
        export_btn.clicked.connect(self.export_dataset)
        
        dataset_buttons_layout.addWidget(scan_btn)
        dataset_buttons_layout.addWidget(process_btn)
        dataset_buttons_layout.addWidget(export_btn)
        
        dataset_actions_layout.addWidget(self.dataset_progress_bar)
        dataset_actions_layout.addLayout(dataset_buttons_layout)
        
        dataset_layout.addWidget(dataset_actions_group)
        
        # 扫描结果显示区域
        dataset_results_group = QtWidgets.QGroupBox("扫描结果")
        dataset_results_layout = QtWidgets.QVBoxLayout(dataset_results_group)
        
        self.dataset_files_list = QtWidgets.QListWidget()
        self.dataset_status_label = QtWidgets.QLabel("请选择要扫描的谱面文件夹")
        
        dataset_results_layout.addWidget(self.dataset_files_list)
        dataset_results_layout.addWidget(self.dataset_status_label)
        
        dataset_layout.addWidget(dataset_results_group)
        
        # 设置"模型训练"选项卡的布局
        self.setup_model_training_tab(model_training_tab)
        
        # 设置"设置"选项卡的布局
        self.setup_settings_tab(settings_tab)
        
        # 设置"谱面生成"选项卡的布局
        self.setup_beatmap_generate_tab(beatmap_generate_tab)
        
        # 设置"字幕处理"选项卡的布局
        self.setup_subtitle_tab(subtitle_tab)
        
        # 设置"视频生成"选项卡的布局
        self.setup_video_generate_tab(video_generate_tab)
        
        # 设置状态栏
        self.status_bar = self.statusBar()
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)
        
        # 设置初始状态和信号连接
        self.reset_source_priority()
        self.init_training_signals()
        
        # 确保可视化器状态与复选框一致
        self.toggle_visualization(QtCore.Qt.Unchecked)
    
    def setup_model_training_tab(self, tab):
        """设置模型训练选项卡的布局"""
        # 创建滚动区域，确保在窗口较小时能显示所有内容
        training_scroll_area = QtWidgets.QScrollArea()
        training_scroll_area.setWidgetResizable(True)
        training_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域内容窗口
        training_scroll_content = QtWidgets.QWidget()
        training_layout = QtWidgets.QVBoxLayout(training_scroll_content)
        training_layout.setSpacing(15)
        
        # 训练数据设置
        training_data_group = QtWidgets.QGroupBox("训练数据设置")
        training_data_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        training_data_layout = QtWidgets.QGridLayout(training_data_group)
        
        # 数据集目录选择
        dataset_root_label = QtWidgets.QLabel("数据集根目录:")
        self.dataset_root_path = QtWidgets.QLineEdit()
        self.dataset_root_path.setPlaceholderText("请选择包含train/val/test子文件夹的数据集目录...")
        
        browse_dataset_root_btn = QtWidgets.QPushButton("浏览")
        browse_dataset_root_btn.clicked.connect(self.browse_dataset_root)
        
        training_data_layout.addWidget(dataset_root_label, 0, 0)
        dataset_root_path_layout = QtWidgets.QHBoxLayout()
        dataset_root_path_layout.addWidget(self.dataset_root_path, 3)
        dataset_root_path_layout.addWidget(browse_dataset_root_btn, 1)
        training_data_layout.addLayout(dataset_root_path_layout, 0, 1)
        
        # 展示检测到的子文件夹信息
        subfolders_info_label = QtWidgets.QLabel("检测到的子文件夹:")
        self.train_folder_label = QtWidgets.QLabel("训练集: 未检测")
        self.val_folder_label = QtWidgets.QLabel("验证集: 未检测")
        self.test_folder_label = QtWidgets.QLabel("测试集: 未检测")
        
        # 设置状态标签样式
        status_style_detected = "color: green; font-weight: bold;"
        status_style_missing = "color: red;"
        self.train_folder_label.setStyleSheet(status_style_missing)
        self.val_folder_label.setStyleSheet(status_style_missing)
        self.test_folder_label.setStyleSheet(status_style_missing)
        
        training_data_layout.addWidget(subfolders_info_label, 1, 0)
        subfolders_info_layout = QtWidgets.QVBoxLayout()
        subfolders_info_layout.addWidget(self.train_folder_label)
        subfolders_info_layout.addWidget(self.val_folder_label)
        subfolders_info_layout.addWidget(self.test_folder_label)
        training_data_layout.addLayout(subfolders_info_layout, 1, 1)
        
        # 模型输出路径
        model_output_label = QtWidgets.QLabel("模型保存路径:")
        self.model_output_path = QtWidgets.QLineEdit()
        self.model_output_path.setPlaceholderText("请选择模型保存路径...")
        
        browse_model_output_btn = QtWidgets.QPushButton("浏览")
        browse_model_output_btn.clicked.connect(self.browse_model_output)
        
        training_data_layout.addWidget(model_output_label, 2, 0)
        model_output_path_layout = QtWidgets.QHBoxLayout()
        model_output_path_layout.addWidget(self.model_output_path, 3)
        model_output_path_layout.addWidget(browse_model_output_btn, 1)
        training_data_layout.addLayout(model_output_path_layout, 2, 1)
        
        training_layout.addWidget(training_data_group)
        
        # 模型训练参数
        training_params_group = QtWidgets.QGroupBox("训练参数")
        training_params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        training_params_layout = QtWidgets.QGridLayout(training_params_group)
        training_params_layout.setVerticalSpacing(10)  # 增加垂直间距，使布局更清晰
        training_params_layout.setHorizontalSpacing(10)  # 增加水平间距
        
        # 模型架构选择
        model_architecture_label = QtWidgets.QLabel("模型架构:")
        self.model_architecture_combo = QtWidgets.QComboBox()
        self.model_architecture_combo.addItems(["Transformer", "LSTM", "GRU", "混合模型"])
        self.model_architecture_combo.setMinimumWidth(120)  # 设置最小宽度
        training_params_layout.addWidget(model_architecture_label, 0, 0)
        training_params_layout.addWidget(self.model_architecture_combo, 0, 1)
        
        # 批次大小
        batch_size_label = QtWidgets.QLabel("批次大小:")
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setSingleStep(2)
        self.batch_size_spin.setMinimumWidth(80)  # 设置最小宽度
        training_params_layout.addWidget(batch_size_label, 1, 0)
        training_params_layout.addWidget(self.batch_size_spin, 1, 1)
        
        # 学习率
        learning_rate_label = QtWidgets.QLabel("学习率:")
        self.learning_rate_combo = QtWidgets.QComboBox()
        self.learning_rate_combo.addItems(["0.0001", "0.0005", "0.001", "0.003", "0.01"])
        self.learning_rate_combo.setCurrentIndex(2)  # 默认选择0.001
        self.learning_rate_combo.setMinimumWidth(80)  # 设置最小宽度
        training_params_layout.addWidget(learning_rate_label, 2, 0)
        training_params_layout.addWidget(self.learning_rate_combo, 2, 1)
        
        # 训练周期
        epochs_label = QtWidgets.QLabel("训练周期:")
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumWidth(80)  # 设置最小宽度
        training_params_layout.addWidget(epochs_label, 3, 0)
        training_params_layout.addWidget(self.epochs_spin, 3, 1)
        
        # GPU设置
        gpu_group = QtWidgets.QGroupBox("GPU设置")
        gpu_layout = QtWidgets.QVBoxLayout(gpu_group)
        gpu_layout.setSpacing(8)  # 减小控件间距
        
        # 检测可用GPU
        self.available_gpus = []
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.available_gpus = [f"GPU {i}: {torch.cuda.get_device_name(i)}" 
                                  for i in range(torch.cuda.device_count())]
            gpu_status = f"检测到 {torch.cuda.device_count()} 个可用GPU"
        else:
            if not TORCH_AVAILABLE:
                gpu_status = "未安装PyTorch库，GPU加速不可用"
            else:
                gpu_status = "未检测到可用GPU"
            
        # GPU状态标签
        self.gpu_status_label = QtWidgets.QLabel(gpu_status)
        self.gpu_status_label.setStyleSheet("color: #0066CC;")
        self.gpu_status_label.setWordWrap(True)  # 允许文字自动换行
        gpu_layout.addWidget(self.gpu_status_label)
        
        # 使用GPU复选框
        self.use_gpu_checkbox = QtWidgets.QCheckBox("使用GPU加速")
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.use_gpu_checkbox.setChecked(gpu_available)
        self.use_gpu_checkbox.setEnabled(gpu_available)
        self.use_gpu_checkbox.toggled.connect(self.toggle_gpu_options)
        gpu_layout.addWidget(self.use_gpu_checkbox)
        
        # GPU设备选择
        gpu_device_layout = QtWidgets.QHBoxLayout()
        gpu_device_layout.setSpacing(5)  # 减小控件间距
        gpu_device_label = QtWidgets.QLabel("GPU设备:")
        self.gpu_device_combo = QtWidgets.QComboBox()
        self.gpu_device_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)  # 根据内容调整大小
        if self.available_gpus:
            self.gpu_device_combo.addItems(self.available_gpus)
        else:
            self.gpu_device_combo.addItem("无可用设备")
        self.gpu_device_combo.setEnabled(gpu_available and self.use_gpu_checkbox.isChecked())
        
        gpu_device_layout.addWidget(gpu_device_label)
        gpu_device_layout.addWidget(self.gpu_device_combo, 1)  # 给combobox分配更多空间
        gpu_layout.addLayout(gpu_device_layout)
        
        # 添加到训练参数布局
        training_params_layout.addWidget(gpu_group, 4, 0, 1, 2)
        
        # 高级选项
        self.use_early_stopping = QtWidgets.QCheckBox("使用早停(根据验证集损失)")
        self.use_early_stopping.setChecked(True)
        training_params_layout.addWidget(self.use_early_stopping, 5, 0, 1, 2)
        
        self.use_checkpoint = QtWidgets.QCheckBox("保存检查点")
        self.use_checkpoint.setChecked(True)
        training_params_layout.addWidget(self.use_checkpoint, 6, 0, 1, 2)
        
        self.use_mixed_precision = QtWidgets.QCheckBox("使用混合精度训练")
        self.use_mixed_precision.setChecked(gpu_available)
        self.use_mixed_precision.setEnabled(gpu_available and self.use_gpu_checkbox.isChecked())
        training_params_layout.addWidget(self.use_mixed_precision, 7, 0, 1, 2)
        
        training_layout.addWidget(training_params_group)
        
        # 训练操作区
        training_actions_group = QtWidgets.QGroupBox("训练操作")
        training_actions_layout = QtWidgets.QVBoxLayout(training_actions_group)
        training_actions_layout.setSpacing(10)  # 设置布局间距
        
        # 训练进度条
        self.training_progress_bar = QtWidgets.QProgressBar()
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setMinimumHeight(20)  # 设置最小高度
        
        # 训练状态显示
        self.training_status_label = QtWidgets.QLabel("准备训练")
        self.training_status_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # 训练日志区域
        self.training_log = QtWidgets.QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(120)  # 减少最小高度以适应小窗口
        self.training_log.setMaximumHeight(200)  # 设置最大高度，避免占用过多空间
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #F8F8F8;
                color: #333333;
                font-family: Consolas, Monospace;
                font-size: 12px;
                border: 1px solid #CCCCCC;
            }
        """)
        
        # 添加初始日志信息
        self.training_log.append("请选择数据集根目录，系统将自动检测训练、验证和测试集子文件夹")
        self.training_log.append("数据集应包含train、val和test子文件夹")
        
        # 训练损失曲线图
        self.training_plot_widget = QtWidgets.QWidget()
        self.training_plot_layout = QtWidgets.QVBoxLayout(self.training_plot_widget)
        self.training_plot_widget.setMinimumHeight(150)  # 设置最小高度
        self.training_plot_widget.setMaximumHeight(250)  # 设置最大高度
        
        # 操作按钮
        training_buttons_layout = QtWidgets.QHBoxLayout()
        training_buttons_layout.setSpacing(8)  # 减小按钮间距
        
        self.start_training_btn = QtWidgets.QPushButton("开始训练")
        self.start_training_btn.clicked.connect(self.start_training)
        self.start_training_btn.setMinimumWidth(80)  # 设置最小宽度
        self.start_training_btn.setEnabled(False)  # 初始状态下禁用，直到检测到有效的数据集
        
        self.pause_resume_btn = QtWidgets.QPushButton("暂停训练")
        self.pause_resume_btn.clicked.connect(self.toggle_training_pause)
        self.pause_resume_btn.setEnabled(False)
        self.pause_resume_btn.setMinimumWidth(80)  # 设置最小宽度
        
        self.stop_training_btn = QtWidgets.QPushButton("停止训练")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setMinimumWidth(80)  # 设置最小宽度
        
        self.export_model_btn = QtWidgets.QPushButton("导出模型")
        self.export_model_btn.clicked.connect(self.export_model)
        self.export_model_btn.setMinimumWidth(80)  # 设置最小宽度
        
        # 添加按钮到布局
        training_buttons_layout.addWidget(self.start_training_btn)
        training_buttons_layout.addWidget(self.pause_resume_btn)
        training_buttons_layout.addWidget(self.stop_training_btn)
        training_buttons_layout.addWidget(self.export_model_btn)
        
        training_actions_layout.addWidget(self.training_progress_bar)
        training_actions_layout.addWidget(self.training_status_label)
        training_actions_layout.addWidget(self.training_log)
        training_actions_layout.addWidget(self.training_plot_widget)
        training_actions_layout.addLayout(training_buttons_layout)
        
        training_layout.addWidget(training_actions_group)
        
        # 设置滚动区域的内容并添加到主布局
        training_scroll_area.setWidget(training_scroll_content)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(training_scroll_area)
    
    def setup_settings_tab(self, tab):
        """设置设置选项卡的布局"""
        # 设置分组框标题样式
        group_box_style = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 15px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center; /* 居中显示 */
            color: white; /* 白色文字 */
            background-color: #FF66AA; /* 粉色背景 */
            padding: 2px 15px;
            border-radius: 3px;
        }
        """
        
        # 创建滚动区域，确保在窗口较小时能显示所有内容
        settings_scroll_area = QtWidgets.QScrollArea()
        settings_scroll_area.setWidgetResizable(True)
        settings_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域内容窗口
        settings_scroll_content = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_scroll_content)
        settings_layout.setSpacing(15)
        
        # 模型设置
        model_group = QtWidgets.QGroupBox("模型设置")
        model_group.setStyleSheet(group_box_style)
        model_layout = QtWidgets.QFormLayout(model_group)
        model_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["默认模型", "流行风格", "古典风格", "电子风格"])
        
        model_layout.addRow("生成模型:", self.model_combo)
        
        # 添加导出选项
        export_options_group = QtWidgets.QGroupBox("导出选项")
        export_options_group.setStyleSheet(group_box_style)
        export_options_layout = QtWidgets.QVBoxLayout(export_options_group)
        export_options_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        # 创建JSON格式选项
        json_format_layout = QtWidgets.QHBoxLayout()
        
        json_format_label = QtWidgets.QLabel("JSON格式:")
        self.json_pretty_rb = QtWidgets.QRadioButton("美化格式")
        self.json_pretty_rb.setChecked(True)
        self.json_compact_rb = QtWidgets.QRadioButton("紧凑格式")
        
        json_format_group = QtWidgets.QButtonGroup(self)
        json_format_group.addButton(self.json_pretty_rb, 1)
        json_format_group.addButton(self.json_compact_rb, 2)
        
        json_format_layout.addWidget(json_format_label)
        json_format_layout.addWidget(self.json_pretty_rb)
        json_format_layout.addWidget(self.json_compact_rb)
        json_format_layout.addStretch()
        
        # 创建导出内容选项
        export_content_layout = QtWidgets.QVBoxLayout()
        export_content_label = QtWidgets.QLabel("导出内容:")
        
        self.export_bpm_cb = QtWidgets.QCheckBox("BPM和节拍信息")
        self.export_bpm_cb.setChecked(True)
        
        self.export_spectrum_cb = QtWidgets.QCheckBox("频谱特征")
        self.export_spectrum_cb.setChecked(True)
        
        self.export_volume_cb = QtWidgets.QCheckBox("音量和能量变化")
        self.export_volume_cb.setChecked(True)
        
        self.export_sections_cb = QtWidgets.QCheckBox("段落和过渡点")
        self.export_sections_cb.setChecked(True)
        
        self.export_visualization_cb = QtWidgets.QCheckBox("可视化数据 (文件较大)")
        self.export_visualization_cb.setChecked(False)
        
        export_content_layout.addWidget(export_content_label)
        export_content_layout.addWidget(self.export_bpm_cb)
        export_content_layout.addWidget(self.export_spectrum_cb)
        export_content_layout.addWidget(self.export_volume_cb)
        export_content_layout.addWidget(self.export_sections_cb)
        export_content_layout.addWidget(self.export_visualization_cb)
        
        # 添加自动导出选项
        auto_export_layout = QtWidgets.QHBoxLayout()
        
        self.auto_export_cb = QtWidgets.QCheckBox("分析后自动导出结果")
        self.auto_export_cb.setChecked(False)
        
        auto_export_layout.addWidget(self.auto_export_cb)
        auto_export_layout.addStretch()
        
        # 添加所有布局到导出选项组
        export_options_layout.addLayout(json_format_layout)
        export_options_layout.addLayout(export_content_layout)
        export_options_layout.addLayout(auto_export_layout)
        
        # 添加设置组到滚动区域布局
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(export_options_group)
        
        # 添加可视化设置组
        visualization_group = QtWidgets.QGroupBox("可视化设置")
        visualization_group.setStyleSheet(group_box_style)
        visualization_layout = QtWidgets.QVBoxLayout(visualization_group)
        visualization_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        # 显示启用可视化的复选框
        self.enable_visualization_cb = QtWidgets.QCheckBox("启用音频可视化 (可能影响性能)")
        self.enable_visualization_cb.setChecked(False)
        self.enable_visualization_cb.stateChanged.connect(self.toggle_visualization)
        visualization_layout.addWidget(self.enable_visualization_cb)
        
        # 添加可视化质量选项
        viz_quality_layout = QtWidgets.QHBoxLayout()
        viz_quality_label = QtWidgets.QLabel("可视化质量:")
        self.viz_quality_combo = QtWidgets.QComboBox()
        self.viz_quality_combo.addItems(["低 (流畅)", "中 (平衡)", "高 (精细)"])
        self.viz_quality_combo.setCurrentIndex(1)  # 默认选择中等质量
        viz_quality_layout.addWidget(viz_quality_label)
        viz_quality_layout.addWidget(self.viz_quality_combo)
        viz_quality_layout.addStretch()
        visualization_layout.addLayout(viz_quality_layout)
        
        # 添加高级可视化选项
        self.enable_realtime_viz_cb = QtWidgets.QCheckBox("启用实时可视化 (需要更高性能)")
        self.enable_realtime_viz_cb.setChecked(False)
        visualization_layout.addWidget(self.enable_realtime_viz_cb)
        
        self.cache_visualizations_cb = QtWidgets.QCheckBox("缓存可视化结果")
        self.cache_visualizations_cb.setChecked(True)
        visualization_layout.addWidget(self.cache_visualizations_cb)
        
        # 添加音频源分离设置
        audio_separation_group = QtWidgets.QGroupBox("人声分离设置")
        audio_separation_group.setStyleSheet(group_box_style)
        audio_separation_layout = QtWidgets.QVBoxLayout(audio_separation_group)
        audio_separation_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        # 启用人声分离复选框
        self.enable_source_separation_cb = QtWidgets.QCheckBox("启用人声分离 (可能显著增加处理时间)")
        self.enable_source_separation_cb.setChecked(True)
        audio_separation_layout.addWidget(self.enable_source_separation_cb)
        
        # 添加模型选择布局
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("分离模型:")
        self.model_combo = QtWidgets.QComboBox()
        # 模型选项将在初始化时动态填充
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        audio_separation_layout.addLayout(model_layout)
        
        # 添加测试源映射的按钮
        test_mapping_btn = QtWidgets.QPushButton("测试音频源内容")
        test_mapping_btn.setToolTip("播放每个分离的音频源3秒钟，以确认标签和内容是否匹配")
        test_mapping_btn.clicked.connect(self.test_audio_sources)
        audio_separation_layout.addWidget(test_mapping_btn)
        
        # 添加优先级选择布局
        priority_group = QtWidgets.QGroupBox("音频源优先级")
        priority_group.setStyleSheet(group_box_style)
        priority_layout = QtWidgets.QVBoxLayout(priority_group)
        priority_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        # 添加说明标签
        priority_label = QtWidgets.QLabel("拖拽调整音频源优先级顺序（顶部最高优先级）:")
        priority_layout.addWidget(priority_label)
        
        # 创建列表控件用于拖拽排序
        self.priority_list = QtWidgets.QListWidget()
        self.priority_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.priority_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.priority_list.setMaximumHeight(120)
        self.priority_list.setMinimumHeight(80)  # 设置最小高度，确保所有项目可见
        self.priority_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)  # 水平扩展，垂直固定
        self.priority_list.setAlternatingRowColors(True)  # 交替行颜色，提高可读性
        self.priority_list.setStyleSheet("QListWidget::item { padding: 5px; border-bottom: 1px solid #444; }")
        # 添加工具提示
        self.priority_list.setToolTip("拖拽列表项以调整音频源的优先级顺序\n顶部的音频源将被优先用于分析")
        
        # 自定义图标指示优先级
        priority_icons = {
            "vocals": "gui/resources/vocal_icon.png",
            "drums": "gui/resources/drum_icon.png",
            "bass": "gui/resources/bass_icon.png",
            "other": "gui/resources/other_icon.png"
        }
        
        # 添加默认的音频源选项 - 显示名称与实际内容一致
        source_display_names = {
            "vocals": "人声 (Vocals)",
            "drums": "鼓声 (Drums)",
            "bass": "贝斯 (Bass)",
            "other": "其他乐器 (Other)"
        }
        
        for source_id, display_name in source_display_names.items():
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, source_id)
            # 尝试设置图标，如果图标文件不存在则忽略
            icon_path = priority_icons.get(source_id)
            if icon_path and os.path.exists(icon_path):
                item.setIcon(QtGui.QIcon(icon_path))
            self.priority_list.addItem(item)
        
        # 连接排序变化的信号
        self.priority_list.model().rowsMoved.connect(self.on_priority_changed)
        
        priority_layout.addWidget(self.priority_list)
        
        # 添加按钮用于重置排序
        reset_priority_btn = QtWidgets.QPushButton("重置默认顺序")
        reset_priority_btn.clicked.connect(self.reset_source_priority)
        priority_layout.addWidget(reset_priority_btn)
        
        audio_separation_layout.addWidget(priority_group)
        
        # 添加导出分离音频选项
        self.export_separated_audio_cb = QtWidgets.QCheckBox("分析后导出分离的音频")
        self.export_separated_audio_cb.setChecked(False)
        audio_separation_layout.addWidget(self.export_separated_audio_cb)
        
        # 添加音频降噪设置
        noise_reduction_group = QtWidgets.QGroupBox("音频降噪设置")
        noise_reduction_group.setStyleSheet(group_box_style)
        noise_reduction_layout = QtWidgets.QVBoxLayout(noise_reduction_group)
        noise_reduction_layout.setContentsMargins(10, 15, 10, 10)  # 调整内边距
        
        # 启用音频降噪复选框
        self.enable_noise_reduction_cb = QtWidgets.QCheckBox("启用音频降噪 (将在人声分离前进行)")
        self.enable_noise_reduction_cb.setChecked(False)
        self.enable_noise_reduction_cb.setToolTip("启用此选项将在人声分离前对音频进行降噪处理，可以改善人声分离效果")
        noise_reduction_layout.addWidget(self.enable_noise_reduction_cb)
        
        # 降噪阈值滑块
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_label = QtWidgets.QLabel("降噪阈值:")
        threshold_label.setToolTip("较低的值会保留更多的原始信号，较高的值会进行更激进的噪声消除")
        self.noise_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_threshold_slider.setRange(0, 100)
        self.noise_threshold_slider.setValue(5)  # 默认值对应0.05
        self.noise_threshold_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.noise_threshold_slider.setTickInterval(10)
        self.noise_threshold_value = QtWidgets.QLabel("0.05")
        self.noise_threshold_slider.valueChanged.connect(lambda v: self.noise_threshold_value.setText(f"{v/100:.2f}"))
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.noise_threshold_slider)
        threshold_layout.addWidget(self.noise_threshold_value)
        noise_reduction_layout.addLayout(threshold_layout)
        
        # 降噪强度滑块
        strength_layout = QtWidgets.QHBoxLayout()
        strength_label = QtWidgets.QLabel("降噪强度:")
        strength_label.setToolTip("控制降噪效果的强度，较高的值会移除更多噪声但可能影响音质")
        self.noise_strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_strength_slider.setRange(0, 100)
        self.noise_strength_slider.setValue(75)  # 默认值对应0.75
        self.noise_strength_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.noise_strength_slider.setTickInterval(10)
        self.noise_strength_value = QtWidgets.QLabel("0.75")
        self.noise_strength_slider.valueChanged.connect(lambda v: self.noise_strength_value.setText(f"{v/100:.2f}"))
        strength_layout.addWidget(strength_label)
        strength_layout.addWidget(self.noise_strength_slider)
        strength_layout.addWidget(self.noise_strength_value)
        noise_reduction_layout.addLayout(strength_layout)
        
        # 重置默认值按钮
        reset_noise_btn = QtWidgets.QPushButton("重置默认值")
        reset_noise_btn.clicked.connect(self.reset_noise_reduction_params)
        noise_reduction_layout.addWidget(reset_noise_btn)
        
        # 添加噪音减少组到布局
        settings_layout.addWidget(noise_reduction_group)
        
        # 添加设置组到滚动区域布局
        settings_layout.addWidget(audio_separation_group)
        settings_layout.addWidget(visualization_group)
        
        settings_layout.addStretch()
        
        # 设置滚动区域的内容并添加到主布局
        settings_scroll_area.setWidget(settings_scroll_content)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(settings_scroll_area)
    
    def setup_subtitle_tab(self, tab):
        """设置字幕处理选项卡的布局"""
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域内容窗口
        scroll_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(scroll_content)
        main_layout.setSpacing(15)
        
        # 分组框样式
        group_box_style = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 15px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center; /* 居中显示 */
            color: white; /* 白色文字 */
            background-color: #FF66AA; /* 粉色背景 */
            padding: 2px 15px;
            border-radius: 3px;
        }
        """
        
        # 创建顶部区域 - 字幕文件选择
        subtitle_group = QtWidgets.QGroupBox("字幕文件")
        subtitle_group.setStyleSheet(group_box_style)
        subtitle_layout = QtWidgets.QVBoxLayout(subtitle_group)
        subtitle_layout.setContentsMargins(15, 20, 15, 15)
        
        # 创建输入字幕文件选择区域
        input_subtitle_layout = QtWidgets.QHBoxLayout()
        input_subtitle_label = QtWidgets.QLabel("输入字幕文件:")
        self.input_subtitle_path = QtWidgets.QLineEdit()
        self.input_subtitle_path.setPlaceholderText("请选择输入字幕文件(.srt)...")
        browse_input_subtitle_btn = QtWidgets.QPushButton("浏览")
        browse_input_subtitle_btn.clicked.connect(self.browse_input_subtitle)
        
        input_subtitle_layout.addWidget(input_subtitle_label)
        input_subtitle_layout.addWidget(self.input_subtitle_path)
        input_subtitle_layout.addWidget(browse_input_subtitle_btn)
        
        # 创建输出字幕文件选择区域
        output_subtitle_layout = QtWidgets.QHBoxLayout()
        output_subtitle_label = QtWidgets.QLabel("输出字幕文件:")
        self.output_subtitle_path = QtWidgets.QLineEdit()
        self.output_subtitle_path.setPlaceholderText("请选择输出字幕文件(.srt)...")
        browse_output_subtitle_btn = QtWidgets.QPushButton("浏览")
        browse_output_subtitle_btn.clicked.connect(self.browse_output_subtitle)
        
        output_subtitle_layout.addWidget(output_subtitle_label)
        output_subtitle_layout.addWidget(self.output_subtitle_path)
        output_subtitle_layout.addWidget(browse_output_subtitle_btn)
        
        # 添加到字幕文件组中
        subtitle_layout.addLayout(input_subtitle_layout)
        subtitle_layout.addLayout(output_subtitle_layout)
        
        # 创建字幕处理选项区域
        options_group = QtWidgets.QGroupBox("处理选项")
        options_group.setStyleSheet(group_box_style)
        options_layout = QtWidgets.QVBoxLayout(options_group)
        options_layout.setContentsMargins(15, 20, 15, 15)
        
        # 创建中文提取选项
        self.extract_chinese_cb = QtWidgets.QCheckBox("提取中文字幕")
        self.extract_chinese_cb.setChecked(True)
        self.extract_chinese_cb.setToolTip("从双语字幕中提取中文部分")
        options_layout.addWidget(self.extract_chinese_cb)
        
        # 创建合并相近字幕选项
        self.merge_nearby_cb = QtWidgets.QCheckBox("合并相近字幕")
        self.merge_nearby_cb.setChecked(True)
        self.merge_nearby_cb.setToolTip("合并相近的重复字幕内容")
        options_layout.addWidget(self.merge_nearby_cb)
        
        # 添加合并时间阈值设置
        merge_threshold_layout = QtWidgets.QHBoxLayout()
        merge_threshold_label = QtWidgets.QLabel("合并时间阈值(秒):")
        self.merge_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.merge_threshold_spin.setRange(0.1, 5.0)
        self.merge_threshold_spin.setValue(0.5)
        self.merge_threshold_spin.setSingleStep(0.1)
        self.merge_threshold_spin.setDecimals(1)
        self.merge_threshold_spin.setEnabled(self.merge_nearby_cb.isChecked())
        self.merge_nearby_cb.toggled.connect(self.merge_threshold_spin.setEnabled)
        
        merge_threshold_layout.addWidget(merge_threshold_label)
        merge_threshold_layout.addWidget(self.merge_threshold_spin)
        merge_threshold_layout.addStretch()
        options_layout.addLayout(merge_threshold_layout)
        
        # 添加字幕预览区域
        preview_group = QtWidgets.QGroupBox("字幕预览")
        preview_group.setStyleSheet(group_box_style)
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(15, 20, 15, 15)
        
        self.subtitle_preview = QtWidgets.QTextEdit()
        self.subtitle_preview.setReadOnly(True)
        self.subtitle_preview.setPlaceholderText("处理后的字幕将显示在这里...")
        self.subtitle_preview.setMinimumHeight(200)
        preview_layout.addWidget(self.subtitle_preview)
        
        # 底部区域 - 操作按钮和状态
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch(1)
        
        self.process_subtitle_btn = QtWidgets.QPushButton("处理字幕")
        self.process_subtitle_btn.setMinimumWidth(150)
        self.process_subtitle_btn.setMinimumHeight(30)
        self.process_subtitle_btn.setStyleSheet("QPushButton { background-color: #FF66AA; color: white; font-weight: bold; }")
        self.process_subtitle_btn.clicked.connect(self.process_subtitle)
        buttons_layout.addWidget(self.process_subtitle_btn)
        
        self.preview_subtitle_btn = QtWidgets.QPushButton("预览字幕")
        self.preview_subtitle_btn.setMinimumWidth(150)
        self.preview_subtitle_btn.setMinimumHeight(30)
        self.preview_subtitle_btn.clicked.connect(self.preview_subtitle)
        self.preview_subtitle_btn.setEnabled(False)
        buttons_layout.addWidget(self.preview_subtitle_btn)
        
        buttons_layout.addStretch(1)
        
        # 状态显示
        status_layout = QtWidgets.QHBoxLayout()
        self.subtitle_status_label = QtWidgets.QLabel("就绪")
        self.subtitle_progress_bar = QtWidgets.QProgressBar()
        self.subtitle_progress_bar.setTextVisible(True)
        self.subtitle_progress_bar.setValue(0)
        
        status_layout.addWidget(self.subtitle_status_label, 1)
        status_layout.addWidget(self.subtitle_progress_bar, 2)
        
        # 将所有组件添加到主布局
        main_layout.addWidget(subtitle_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(preview_group)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(status_layout)
        
        # 设置滚动区域的内容并添加到主布局
        scroll_area.setWidget(scroll_content)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(scroll_area)
    
    def setup_beatmap_generate_tab(self, tab):
        """设置谱面生成选项卡的布局"""
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域内容窗口
        scroll_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(scroll_content)
        main_layout.setSpacing(15)
        
        # 创建顶部水平布局，用于文件选择和轨道选择
        top_layout = QtWidgets.QHBoxLayout()
        
        # 左侧区域 - 文件选择
        file_selection_widget = QtWidgets.QWidget()
        file_layout = QtWidgets.QVBoxLayout(file_selection_widget)
        file_layout.setContentsMargins(0, 0, 10, 0)
        
        # 音频分析文件选择组
        audio_group = QtWidgets.QGroupBox("音频分析文件")
        audio_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        audio_layout = QtWidgets.QVBoxLayout(audio_group)
        audio_layout.setContentsMargins(15, 20, 15, 15)  # 增加内边距
        
        audio_file_row = QtWidgets.QHBoxLayout()
        self.beatmap_gen_audio_path = QtWidgets.QLineEdit()
        self.beatmap_gen_audio_path.setPlaceholderText("请选择音频分析文件夹...")
        
        browse_audio_btn = QtWidgets.QPushButton("浏览")
        browse_audio_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
                min-width: 60px; 
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
        """)
        browse_audio_btn.clicked.connect(self.browse_beatmap_gen_audio)
        
        audio_file_row.addWidget(self.beatmap_gen_audio_path, 3)
        audio_file_row.addWidget(browse_audio_btn, 1)
        audio_layout.addLayout(audio_file_row)
        
        # 输出目录选择
        output_row = QtWidgets.QHBoxLayout()
        output_label = QtWidgets.QLabel("谱面保存目录:")
        self.beatmap_gen_output_path = QtWidgets.QLineEdit()
        self.beatmap_gen_output_path.setPlaceholderText("谱面保存目录...")
        
        browse_output_btn = QtWidgets.QPushButton("浏览")
        browse_output_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
                min-width: 60px; 
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
        """)
        browse_output_btn.clicked.connect(self.browse_beatmap_gen_output)
        
        output_row.addWidget(output_label)
        output_row.addWidget(self.beatmap_gen_output_path, 3)
        output_row.addWidget(browse_output_btn, 1)
        audio_layout.addLayout(output_row)
        
        file_layout.addWidget(audio_group)
        
        # 谱面元数据
        metadata_group = QtWidgets.QGroupBox("谱面元数据")
        metadata_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        metadata_layout = QtWidgets.QGridLayout(metadata_group)
        metadata_layout.setContentsMargins(15, 20, 15, 15)  # 增加内边距
        
        metadata_layout.addWidget(QtWidgets.QLabel("曲目标题:"), 0, 0)
        self.beatmap_title = QtWidgets.QLineEdit()
        metadata_layout.addWidget(self.beatmap_title, 0, 1)
        
        metadata_layout.addWidget(QtWidgets.QLabel("艺术家:"), 1, 0)
        self.beatmap_artist = QtWidgets.QLineEdit()
        metadata_layout.addWidget(self.beatmap_artist, 1, 1)
        
        metadata_layout.addWidget(QtWidgets.QLabel("谱面作者:"), 2, 0)
        self.beatmap_creator = QtWidgets.QLineEdit("AI谱面生成器")
        metadata_layout.addWidget(self.beatmap_creator, 2, 1)
        
        file_layout.addWidget(metadata_group)
        
        # 添加到左侧布局
        top_layout.addWidget(file_selection_widget, 1)
        
        # 右侧区域 - 音频轨道选择
        track_selection_widget = QtWidgets.QWidget()
        track_layout = QtWidgets.QVBoxLayout(track_selection_widget)
        track_layout.setContentsMargins(10, 0, 0, 0)
        
        # 音频轨道选择
        sources_group = QtWidgets.QGroupBox("音频轨道选择")
        sources_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        sources_layout = QtWidgets.QVBoxLayout(sources_group)
        sources_layout.setContentsMargins(15, 20, 15, 15)  # 增加内边距
        
        # 说明标签
        sources_layout.addWidget(QtWidgets.QLabel("选择要用于生成谱面的音频轨道及其优先级:"))
        
        # 音频轨道表格
        self.source_table = QtWidgets.QTableWidget()
        self.source_table.setColumnCount(3)
        self.source_table.setHorizontalHeaderLabels(["选择", "音频轨道", "优先级"])
        self.source_table.setStyleSheet("""
            QTableWidget { 
                border: 1px solid #ddd; 
                gridline-color: #eee;
            }
            QTableWidget::item { 
                padding: 4px; 
                border-bottom: 1px solid #eee; 
            }
            QTableWidget::item:selected { 
                background-color: #ffebf3; 
                color: black;
            }
        """)
        
        # 设置表头样式 - 扁平化设计，不像按钮
        self.source_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section { 
                background-color: #FF66AA; 
                color: white;
                font-weight: bold;
                padding: 6px 8px;
                border: none;
            }
        """)
        
        self.source_table.verticalHeader().setVisible(False)  # 隐藏垂直表头
        
        # 设置表头自动拉伸模式
        self.source_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)  # 选择列固定宽度
        self.source_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)  # 音频轨道列自动拉伸
        self.source_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)  # 优先级列固定宽度
        
        # 设置列宽
        self.source_table.setColumnWidth(0, 60)  # 增加选择列宽度，确保不被遮挡
        self.source_table.setColumnWidth(2, 70)  # 优先级列宽度
        
        # 确保表格有足够的宽度
        self.source_table.setMinimumWidth(300)
        self.source_table.setMinimumHeight(150)
        self.source_table.setMaximumHeight(200)
        self.source_table.setAlternatingRowColors(True)  # 交替行颜色
        sources_layout.addWidget(self.source_table)
        
        # 选择数量和优先级
        options_row = QtWidgets.QHBoxLayout()
        
        # 最大轨道数量
        options_row.addWidget(QtWidgets.QLabel("最大轨道数:"))
        self.max_sources_spin = QtWidgets.QSpinBox()
        self.max_sources_spin.setRange(1, 5)
        self.max_sources_spin.setValue(3)
        self.max_sources_spin.setToolTip("设置要使用的最大轨道数量")
        options_row.addWidget(self.max_sources_spin)
        
        # 刷新按钮
        refresh_btn = QtWidgets.QPushButton("刷新轨道列表")
        refresh_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
                min-width: 100px; 
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
        """)
        refresh_btn.clicked.connect(self.refresh_audio_sources)
        options_row.addWidget(refresh_btn)
        
        sources_layout.addLayout(options_row)
        track_layout.addWidget(sources_group)
        
        # 添加到右侧布局
        top_layout.addWidget(track_selection_widget, 1)
        
        # 添加顶部布局到主布局
        main_layout.addLayout(top_layout)
        
        # 创建中间区域，用于难度设置和生成选项
        middle_layout = QtWidgets.QHBoxLayout()
        
        # 左侧 - 难度设置
        difficulty_group = QtWidgets.QGroupBox("难度设置")
        difficulty_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        difficulty_layout = QtWidgets.QVBoxLayout(difficulty_group)
        difficulty_layout.setContentsMargins(15, 20, 15, 15)  # 增加内边距
        
        # 难度选择
        difficulty_selector_row = QtWidgets.QHBoxLayout()
        difficulty_selector_row.addWidget(QtWidgets.QLabel("难度:"))
        self.difficulty_selector = QtWidgets.QComboBox()
        self.difficulty_selector.addItems(["Easy", "Normal", "Hard", "Expert"])
        self.difficulty_selector.setCurrentIndex(0)  # 默认选择Easy
        difficulty_selector_row.addWidget(self.difficulty_selector)
        difficulty_layout.addLayout(difficulty_selector_row)
        
        # 难度参数
        difficulty_params_grid = QtWidgets.QGridLayout()
        
        difficulty_params_grid.addWidget(QtWidgets.QLabel("接近速度(AR):"), 0, 0)
        self.ar_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ar_slider.setRange(0, 100)
        self.ar_slider.setValue(50)
        self.ar_value = QtWidgets.QLabel("5.0")
        self.ar_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        difficulty_params_grid.addWidget(self.ar_slider, 0, 1)
        difficulty_params_grid.addWidget(self.ar_value, 0, 2)
        
        difficulty_params_grid.addWidget(QtWidgets.QLabel("总体难度(OD):"), 1, 0)
        self.od_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.od_slider.setRange(0, 100)
        self.od_slider.setValue(50)
        self.od_value = QtWidgets.QLabel("5.0")
        self.od_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        difficulty_params_grid.addWidget(self.od_slider, 1, 1)
        difficulty_params_grid.addWidget(self.od_value, 1, 2)
        
        difficulty_params_grid.addWidget(QtWidgets.QLabel("血量消耗(HP):"), 2, 0)
        self.hp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hp_slider.setRange(0, 100)
        self.hp_slider.setValue(50)
        self.hp_value = QtWidgets.QLabel("5.0")
        self.hp_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        difficulty_params_grid.addWidget(self.hp_slider, 2, 1)
        difficulty_params_grid.addWidget(self.hp_value, 2, 2)
        
        difficulty_params_grid.addWidget(QtWidgets.QLabel("圆圈大小(CS):"), 3, 0)
        self.cs_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cs_slider.setRange(0, 100)
        self.cs_slider.setValue(40)
        self.cs_value = QtWidgets.QLabel("4.0")
        self.cs_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        difficulty_params_grid.addWidget(self.cs_slider, 3, 1)
        difficulty_params_grid.addWidget(self.cs_value, 3, 2)
        
        # 连接信号
        self.ar_slider.valueChanged.connect(lambda v: self.ar_value.setText(f"{v/10:.1f}"))
        self.od_slider.valueChanged.connect(lambda v: self.od_value.setText(f"{v/10:.1f}"))
        self.hp_slider.valueChanged.connect(lambda v: self.hp_value.setText(f"{v/10:.1f}"))
        self.cs_slider.valueChanged.connect(lambda v: self.cs_value.setText(f"{v/10:.1f}"))
        
        # 设置初始值
        self.ar_slider.valueChanged.emit(self.ar_slider.value())
        self.od_slider.valueChanged.emit(self.od_slider.value())
        self.hp_slider.valueChanged.emit(self.hp_slider.value())
        self.cs_slider.valueChanged.emit(self.cs_slider.value())
        
        difficulty_layout.addLayout(difficulty_params_grid)
        
        # 添加到中间左侧
        middle_layout.addWidget(difficulty_group)
        
        # 右侧 - 生成选项
        options_group = QtWidgets.QGroupBox("生成选项")
        options_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        options_layout = QtWidgets.QVBoxLayout(options_group)
        options_layout.setContentsMargins(15, 20, 15, 15)  # 增加内边距
        
        # 模型选项
        model_row = QtWidgets.QHBoxLayout()
        self.use_model_cb = QtWidgets.QCheckBox("使用AI优化摆放")
        self.use_model_cb.setToolTip("使用训练好的AI模型优化物件摆放位置")
        self.use_model_cb.setEnabled(False)  # 暂时禁用
        model_row.addWidget(self.use_model_cb)
        options_layout.addLayout(model_row)
        
        # 导出选项
        export_row = QtWidgets.QHBoxLayout()
        self.auto_open_cb = QtWidgets.QCheckBox("生成后自动打开")
        self.auto_open_cb.setChecked(True)
        export_row.addWidget(self.auto_open_cb)
        options_layout.addLayout(export_row)
        
        # 密度设置
        density_row = QtWidgets.QHBoxLayout()
        density_row.addWidget(QtWidgets.QLabel("谱面密度:"))
        self.density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.density_slider.setRange(1, 10)
        self.density_slider.setValue(10)
        self.density_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.density_slider.setTickInterval(1)
        self.density_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        self.density_value = QtWidgets.QLabel("10")
        density_row.addWidget(self.density_slider)
        density_row.addWidget(self.density_value)
        options_layout.addLayout(density_row)
        
        # 事件选择概率设置
        options_layout.addWidget(QtWidgets.QLabel("事件选择概率设置:"))
        
        # 节拍点选择概率
        beat_prob_row = QtWidgets.QHBoxLayout()
        beat_prob_row.addWidget(QtWidgets.QLabel("节拍点选择概率:"))
        self.beat_prob_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.beat_prob_slider.setRange(0, 100)
        self.beat_prob_slider.setValue(100)  # 默认1.0 (100%)
        self.beat_prob_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.beat_prob_slider.setTickInterval(10)
        self.beat_prob_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        self.beat_prob_value = QtWidgets.QLabel("1.00")
        beat_prob_row.addWidget(self.beat_prob_slider)
        beat_prob_row.addWidget(self.beat_prob_value)
        options_layout.addLayout(beat_prob_row)
        
        # 起始点选择概率
        onset_prob_row = QtWidgets.QHBoxLayout()
        onset_prob_row.addWidget(QtWidgets.QLabel("起始点选择概率:"))
        self.onset_prob_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.onset_prob_slider.setRange(0, 100)
        self.onset_prob_slider.setValue(100)  # 默认1.0 (100%)
        self.onset_prob_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.onset_prob_slider.setTickInterval(10)
        self.onset_prob_slider.setStyleSheet("QSlider::groove:horizontal { background: #ddd; } QSlider::handle:horizontal { background: #FF66AA; }")
        self.onset_prob_value = QtWidgets.QLabel("1.00")
        onset_prob_row.addWidget(self.onset_prob_slider)
        onset_prob_row.addWidget(self.onset_prob_value)
        options_layout.addLayout(onset_prob_row)
        
        # 强拍弱拍判定设置
        strong_beat_row = QtWidgets.QHBoxLayout()
        strong_beat_row.addWidget(QtWidgets.QLabel("强拍阈值:"))
        self.strong_beat_input = QtWidgets.QDoubleSpinBox()
        self.strong_beat_input.setRange(0.001, 1.0)  # 范围0.001-1.0
        self.strong_beat_input.setSingleStep(0.001)  # 步进值
        self.strong_beat_input.setDecimals(3)  # 显示3位小数
        self.strong_beat_input.setValue(0.075)  # 默认值0.075
        self.strong_beat_input.setFixedWidth(80)
        self.strong_beat_input.setToolTip("设置强拍判定阈值，当音量高于此值时被判定为强拍")
        strong_beat_row.addWidget(self.strong_beat_input)
        strong_beat_row.addStretch(1)
        options_layout.addLayout(strong_beat_row)
        
        # 次强拍阈值
        medium_beat_row = QtWidgets.QHBoxLayout()
        medium_beat_row.addWidget(QtWidgets.QLabel("次强拍阈值:"))
        self.medium_beat_input = QtWidgets.QDoubleSpinBox()
        self.medium_beat_input.setRange(0.001, 1.0)  # 范围0.001-1.0
        self.medium_beat_input.setSingleStep(0.001)  # 步进值
        self.medium_beat_input.setDecimals(3)  # 显示3位小数
        self.medium_beat_input.setValue(0.025)  # 默认值0.025
        self.medium_beat_input.setFixedWidth(80)
        self.medium_beat_input.setToolTip("设置次强拍判定阈值，当音量高于此值但低于强拍阈值时被判定为次强拍")
        medium_beat_row.addWidget(self.medium_beat_input)
        medium_beat_row.addStretch(1)
        options_layout.addLayout(medium_beat_row)
        
        # 确保阈值一致性 - 添加验证
        self.strong_beat_input.valueChanged.connect(self.ensure_threshold_consistency)
        self.medium_beat_input.valueChanged.connect(self.ensure_threshold_consistency)
        
        # 连接信号
        self.density_slider.valueChanged.connect(lambda v: self.density_value.setText(str(v)))
        self.beat_prob_slider.valueChanged.connect(lambda v: self.beat_prob_value.setText(f"{v/100:.2f}"))
        self.onset_prob_slider.valueChanged.connect(lambda v: self.onset_prob_value.setText(f"{v/100:.2f}"))
        
        # 添加填充空间，使选项面板与难度面板高度相同
        options_layout.addStretch(1)
        
        # 添加到中间右侧
        middle_layout.addWidget(options_group)
        
        # 添加中间区域到主布局
        main_layout.addLayout(middle_layout)
        
        # 底部区域 - 操作按钮和状态
        bottom_layout = QtWidgets.QVBoxLayout()
        
        # 操作按钮行
        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addStretch(1)  # 左侧弹性空间
        
        self.generate_btn = QtWidgets.QPushButton("生成谱面")
        self.generate_btn.setMinimumWidth(150)
        self.generate_btn.setMinimumHeight(30)
        self.generate_btn.setStyleSheet("QPushButton { background-color: #FF66AA; color: white; font-weight: bold; }")
        self.generate_btn.clicked.connect(self.generate_beatmap_from_tab)
        self.generate_btn.setEnabled(True)
        buttons_row.addWidget(self.generate_btn)
        
        buttons_row.addSpacing(20)  # 按钮之间的间距
        
        self.preview_btn = QtWidgets.QPushButton("预览谱面")
        self.preview_btn.setMinimumWidth(150)
        self.preview_btn.setMinimumHeight(30)
        self.preview_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.preview_btn.clicked.connect(self.preview_beatmap_from_tab)
        self.preview_btn.setEnabled(False)
        buttons_row.addWidget(self.preview_btn)
        
        buttons_row.addSpacing(20)  # 按钮之间的间距
        
        # 添加浏览谱面按钮
        self.browse_beatmap_btn = QtWidgets.QPushButton("浏览谱面")
        self.browse_beatmap_btn.setMinimumWidth(150)
        self.browse_beatmap_btn.setMinimumHeight(30)
        self.browse_beatmap_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.browse_beatmap_btn.clicked.connect(self.browse_beatmap)
        buttons_row.addWidget(self.browse_beatmap_btn)
        
        buttons_row.addStretch(1)  # 右侧弹性空间
        bottom_layout.addLayout(buttons_row)
        
        # 状态显示
        status_layout = QtWidgets.QHBoxLayout()
        status_label = QtWidgets.QLabel("状态:")
        status_label.setMinimumWidth(50)
        status_layout.addWidget(status_label)
        
        self.beatmap_gen_status = QtWidgets.QLabel("就绪")
        self.beatmap_gen_status.setStyleSheet("QLabel { font-weight: bold; }")
        status_layout.addWidget(self.beatmap_gen_status)
        
        bottom_layout.addLayout(status_layout)
        
        # 进度条
        self.beatmap_gen_progress = QtWidgets.QProgressBar()
        self.beatmap_gen_progress.setRange(0, 100)
        self.beatmap_gen_progress.setValue(0)
        self.beatmap_gen_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF66AA;
            }
        """)
        bottom_layout.addWidget(self.beatmap_gen_progress)
        
        # 添加底部区域到主布局
        main_layout.addLayout(bottom_layout)
        
        # 设置滚动区域
        scroll_area.setWidget(scroll_content)
        
        # 主布局
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(scroll_area)
    
    def setup_video_generate_tab(self, tab):
        """设置视频生成选项卡的布局"""
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域内容窗口
        scroll_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(scroll_content)
        main_layout.setSpacing(15)
        
        # 创建顶部区域 - 文件选择
        file_group = QtWidgets.QGroupBox("谱面文件")
        file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 居中显示 */
                color: white; /* 白色文字 */
                background-color: #FF66AA; /* 粉色背景 */
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        file_layout = QtWidgets.QVBoxLayout(file_group)
        file_layout.setContentsMargins(15, 20, 15, 15)
        
        # 谱面文件选择
        osu_file_row = QtWidgets.QHBoxLayout()
        osu_file_label = QtWidgets.QLabel("谱面文件:")
        self.osu_file_path = QtWidgets.QLineEdit()
        self.osu_file_path.setPlaceholderText("请选择.osu谱面文件...")
        
        browse_osu_btn = QtWidgets.QPushButton("浏览")
        browse_osu_btn.clicked.connect(self.browse_osu_file)
        
        osu_file_row.addWidget(osu_file_label)
        osu_file_row.addWidget(self.osu_file_path, 3)
        osu_file_row.addWidget(browse_osu_btn)
        
        file_layout.addLayout(osu_file_row)
        
        # 输出设置
        output_row = QtWidgets.QHBoxLayout()
        output_label = QtWidgets.QLabel("输出目录:")
        self.video_output_path = QtWidgets.QLineEdit()
        self.video_output_path.setPlaceholderText("视频输出目录...")
        
        browse_output_btn = QtWidgets.QPushButton("浏览")
        browse_output_btn.clicked.connect(self.browse_video_output)
        
        output_row.addWidget(output_label)
        output_row.addWidget(self.video_output_path, 3)
        output_row.addWidget(browse_output_btn)
        
        file_layout.addLayout(output_row)
        
        main_layout.addWidget(file_group)
        
        # 视频设置组
        video_settings_group = QtWidgets.QGroupBox("视频设置")
        video_settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                color: white;
                background-color: #FF66AA;
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        video_settings_layout = QtWidgets.QVBoxLayout(video_settings_group)
        video_settings_layout.setContentsMargins(15, 20, 15, 15)
        
        # FPS设置
        fps_row = QtWidgets.QHBoxLayout()
        fps_label = QtWidgets.QLabel("帧率(FPS):")
        self.fps_spinbox = QtWidgets.QSpinBox()
        self.fps_spinbox.setRange(30, 120)
        self.fps_spinbox.setValue(60)
        self.fps_spinbox.setSingleStep(10)
        
        fps_row.addWidget(fps_label)
        fps_row.addWidget(self.fps_spinbox)
        fps_row.addStretch(1)
        
        video_settings_layout.addLayout(fps_row)
        
        # 滚动速度设置
        speed_row = QtWidgets.QHBoxLayout()
        speed_label = QtWidgets.QLabel("滚动速度:")
        self.scroll_speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scroll_speed_slider.setRange(500, 2000)
        self.scroll_speed_slider.setValue(1000)
        self.scroll_speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.scroll_speed_slider.setTickInterval(100)
        self.scroll_speed_value = QtWidgets.QLabel("1000")
        
        speed_row.addWidget(speed_label)
        speed_row.addWidget(self.scroll_speed_slider, 3)
        speed_row.addWidget(self.scroll_speed_value)
        
        # 连接滑动条的信号
        self.scroll_speed_slider.valueChanged.connect(lambda v: self.scroll_speed_value.setText(str(v)))
        
        video_settings_layout.addLayout(speed_row)
        
        # 轨道宽度设置
        lane_width_row = QtWidgets.QHBoxLayout()
        lane_width_label = QtWidgets.QLabel("轨道宽度:")
        self.lane_width_spinbox = QtWidgets.QSpinBox()
        self.lane_width_spinbox.setRange(100, 500)
        self.lane_width_spinbox.setValue(200)
        self.lane_width_spinbox.setSingleStep(10)
        self.lane_width_spinbox.setSuffix(" px")
        
        lane_width_row.addWidget(lane_width_label)
        lane_width_row.addWidget(self.lane_width_spinbox)
        lane_width_row.addStretch(1)
        
        video_settings_layout.addLayout(lane_width_row)
        
        main_layout.addWidget(video_settings_group)
        
        # 预览区域
        preview_group = QtWidgets.QGroupBox("预览")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                color: white;
                background-color: #FF66AA;
                padding: 2px 15px;
                border-radius: 3px;
            }
        """)
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(15, 20, 15, 15)
        
        # 预览标签
        self.video_preview_label = QtWidgets.QLabel("选择谱面文件后可预览")
        self.video_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_preview_label.setStyleSheet("""
            font-size: 16px;
            color: #666666;
            background-color: #EEEEEE;
            border: 1px dashed #CCCCCC;
            min-height: 200px;
        """)
        
        preview_layout.addWidget(self.video_preview_label)
        
        main_layout.addWidget(preview_group)
        
        # 底部按钮区域
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch(1)
        
        # 预览按钮
        self.video_preview_btn = QtWidgets.QPushButton("谱面预览")
        self.video_preview_btn.setMinimumWidth(150)
        self.video_preview_btn.setMinimumHeight(30)
        self.video_preview_btn.clicked.connect(self.preview_video)
        self.video_preview_btn.setEnabled(False)
        
        # 生成按钮
        self.generate_video_btn = QtWidgets.QPushButton("生成视频")
        self.generate_video_btn.setMinimumWidth(150)
        self.generate_video_btn.setMinimumHeight(30)
        self.generate_video_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF66AA; 
                color: white; 
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #FF77BB; 
            }
            QPushButton:pressed { 
                background-color: #DD4488; 
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.generate_video_btn.clicked.connect(self.generate_video)
        self.generate_video_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.video_preview_btn)
        buttons_layout.addWidget(self.generate_video_btn)
        buttons_layout.addStretch(1)
        
        main_layout.addLayout(buttons_layout)
        
        # 状态区域
        status_layout = QtWidgets.QHBoxLayout()
        
        self.video_gen_status = QtWidgets.QLabel("就绪")
        self.video_gen_status.setStyleSheet("color: #666666;")
        
        self.video_gen_progress = QtWidgets.QProgressBar()
        self.video_gen_progress.setValue(0)
        
        status_layout.addWidget(self.video_gen_status, 1)
        status_layout.addWidget(self.video_gen_progress, 2)
        
        main_layout.addLayout(status_layout)
        
        # 设置滚动区域的内容并添加到主布局
        scroll_area.setWidget(scroll_content)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(scroll_area)
        
        # 检查视频生成模块是否可用
        if not VIDEO_GEN_AVAILABLE:
            error_label = QtWidgets.QLabel("视频生成模块不可用，请检查依赖项安装。")
            error_label.setStyleSheet("""
                color: red;
                font-weight: bold;
                padding: 10px;
                background-color: #FFEEEE;
                border: 1px solid red;
                border-radius: 5px;
            """)
            main_layout.insertWidget(0, error_label)
    
    def browse_audio(self):
        """浏览并选择音频文件"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.mp3 *.wav *.ogg *.flac)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self.status_label.setText(f"已选择文件: {os.path.basename(file_path)}")
    
    def browse_output_directory(self):
        """浏览并选择输出目录"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择输出目录", ""
        )
        if directory:
            self.output_path.setText(directory)
    
    def analyze_audio(self):
        """分析音频文件"""
        # 检查音频文件路径
        audio_path = self.file_path.text()
        if not audio_path or not os.path.exists(audio_path):
            QtWidgets.QMessageBox.warning(self, "路径错误", "请选择有效的音频文件。")
            return
        
        # 禁用分析按钮，防止重复操作
        self.analyze_btn.setEnabled(False)
        
        # 准备音频分析器
        self.audio_analyzer = AudioAnalyzer(use_gpu=self.use_gpu_cb.isChecked())
        
        # 连接信号
        self.audio_analyzer.analysis_progress.connect(self.update_analysis_progress)
        self.audio_analyzer.analysis_complete.connect(self.handle_analysis_complete)
        self.audio_analyzer.analysis_error.connect(self.handle_analysis_error)
        
        # 配置降噪设置
        if hasattr(self, 'enable_noise_reduction_cb'):
            self.audio_analyzer.set_use_noise_reduction(self.enable_noise_reduction_cb.isChecked())
            
            # 如果启用了降噪，设置降噪参数
            if self.enable_noise_reduction_cb.isChecked():
                threshold = self.noise_threshold_slider.value() / 100.0
                strength = self.noise_strength_slider.value() / 100.0
                self.audio_analyzer.set_noise_reduction_params(threshold, strength)
        
        # 配置人声分离设置
        if hasattr(self, 'enable_source_separation_cb'):
            self.audio_analyzer.set_use_source_separation(self.enable_source_separation_cb.isChecked())
            
            # 设置分离模型（如果选择了）
            if hasattr(self, 'model_combo') and self.enable_source_separation_cb.isChecked():
                model_key = self.model_combo.currentData()
                if model_key:
                    try:
                        self.audio_analyzer.set_separation_model(model_key)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "模型加载错误", f"无法加载所选模型: {str(e)}")
            
            # 设置音频源优先级 - 使用用户自定义的顺序
            if self.enable_source_separation_cb.isChecked() and hasattr(self, 'priority_list'):
                # 从列表中获取用户排序的优先级
                priority_sources = []
                for i in range(self.priority_list.count()):
                    item = self.priority_list.item(i)
                    source_id = item.data(QtCore.Qt.UserRole)
                    priority_sources.append(source_id)
                
                # 设置自定义优先级
                if priority_sources:
                    self.audio_analyzer.set_source_priority(priority_sources)
        
        # 加载音频文件
        success = self.audio_analyzer.load_audio(audio_path)
        if not success:
            QtWidgets.QMessageBox.warning(self, "加载失败", "无法加载音频文件，请检查文件格式。")
            self.analyze_btn.setEnabled(True)
            return
        
        # 检查是否使用手动BPM
        if self.manual_bpm_rb.isChecked():
            bpm = self.bpm_input.value()
            self.audio_analyzer.set_manual_bpm(bpm)
            
        # 创建QThread以避免UI阻塞
        self.analysis_thread = QtCore.QThread()
        self.audio_analyzer.moveToThread(self.analysis_thread)
        
        # 线程开始和结束时的处理
        self.analysis_thread.started.connect(self.audio_analyzer.analyze)
        self.audio_analyzer.analysis_complete.connect(self.analysis_thread.quit)
        self.audio_analyzer.analysis_error.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(lambda: self.analyze_btn.setEnabled(True))
        
        # 开始分析
        self.status_label.setText("正在分析音频...")
        self.progress_bar.setValue(0)
        self.analysis_thread.start()
    
    def update_analysis_progress(self, progress):
        """更新分析进度"""
        self.progress_bar.setValue(progress)
        
        # 根据进度更新状态文本
        if progress < 5:
            self.status_label.setText("正在处理音频...")
            if hasattr(self, 'enable_noise_reduction_cb') and self.enable_noise_reduction_cb.isChecked():
                self.status_label.setText("正在进行音频降噪...")
        elif progress < 20:
            self.status_label.setText("正在检测BPM和节拍...")
        elif progress < 40:
            self.status_label.setText("正在分析节拍强度...")
        elif progress < 60:
            self.status_label.setText("正在提取频谱特征...")
        elif progress < 80:
            self.status_label.setText("正在检测音频段落...")
        else:
            self.status_label.setText("正在完成分析...")
            
        # 如果启用了音频源分离，显示正在分析所有音频源
        if hasattr(self, 'enable_source_separation_cb') and self.enable_source_separation_cb.isChecked():
            self.status_label.setText(f"{self.status_label.text()} (正在分析所有音频源: {progress}%)")
    
    def handle_analysis_complete(self, features):
        """处理分析完成"""
        # 结束分析线程
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        
        # 更新UI状态
        self.progress_bar.setValue(100)
        
        # 保存features到audio_analyzer对象中，以便export_analysis使用
        self.audio_analyzer.features = features
        
        # 获取基本音频信息
        bpm = features.get("bpm", 0)
        bpm_source = features.get("beat_source", "default")
        beat_count = len(features.get("beat_times", []) or [])
        
        # 更新BPM输入框
        self.bpm_input.setValue(bpm)
        
        # 更新状态文本
        bpm_source_text = {"default": "自动检测", "manual": "手动设置", "beatmap": "从谱面导入"}
        source = bpm_source_text.get(bpm_source, "未知")
        
        # 显示活跃音频源信息
        active_source = features.get("active_source", "original")
        source_display = {
            "original": "原始音频",
            "vocals": "人声",
            "drums": "鼓声",
            "bass": "贝斯",
            "other": "其他乐器"
        }
        source_text = source_display.get(active_source, active_source)
        
        base_status = f"音频分析完成! BPM: {bpm} ({source}), 节拍数: {beat_count}, 活跃源: {source_text}"
        
        # 检查是否分析了所有源
        if 'all_sources_analysis' in features:
            analyzed_sources = list(features['all_sources_analysis'].keys())
            if len(analyzed_sources) > 1:
                source_names = [source_display.get(s, s) for s in analyzed_sources]
                self.status_label.setText(f"{base_status} | 已分析所有音频源: {', '.join(source_names)}")
            else:
                self.status_label.setText(base_status)
        else:
            self.status_label.setText(base_status)
        
        # 如果开启了分离音频导出
        if hasattr(self, 'export_separated_audio_cb') and self.export_separated_audio_cb.isChecked() and hasattr(self.audio_analyzer, 'separated_sources') and self.audio_analyzer.separated_sources:
            # 获取输出目录
            output_dir = self.output_path.text()
            if not output_dir:
                # 使用音频文件所在目录
                output_dir = os.path.dirname(self.file_path.text())
                
            # 创建分离音频子目录
            output_dir = os.path.join(output_dir, "separated_audio")
            
            # 导出分离的音频和分析数据
            exported_files = self.audio_analyzer.export_separated_audio(output_dir)
            
            if exported_files:
                # 源类型的人类可读描述
                source_descriptions = {
                    "vocals": "人声 (Vocals)",
                    "drums": "鼓声 (Drums)",
                    "bass": "贝斯 (Bass)",
                    "other": "其他乐器 (Other)"
                }
                
                # 显示导出成功信息，使用更友好的描述
                export_paths = []
                for k, v in exported_files.items():
                    # 获取音频和分析数据的路径
                    audio_path = v.get("audio", "")
                    analysis_path = v.get("analysis", "")
                    
                    # 使用更友好的描述而不是代码中的键名
                    source_desc = source_descriptions.get(k, k)
                    audio_filename = os.path.basename(audio_path) if audio_path else "未导出"
                    analysis_filename = os.path.basename(analysis_path) if analysis_path else "未导出"
                    
                    export_paths.append(f"{source_desc}:\n   音频: {audio_filename}\n   分析: {analysis_filename}")
                
                # 拼接成文本
                export_info = "\n".join(export_paths)
                
                QtWidgets.QMessageBox.information(
                    self, "导出成功", 
                    f"分离的音频和分析数据已导出到:\n{output_dir}\n\n{export_info}"
                )
        
        # 仅当可视化功能启用时才更新可视化器
        if self.enable_visualization_cb.isChecked():
            # 设置分离的音频源（如果有）
            if hasattr(self.audio_analyzer, 'separated_sources'):
                self.audio_visualizer.separated_sources = self.audio_analyzer.separated_sources
                
            self.audio_visualizer.set_audio_data(self.audio_analyzer.y, self.audio_analyzer.sr)
            self.audio_visualizer.set_audio_features(features)
        
        # 根据分析结果预设谱面参数
        density_suggestions = self.audio_analyzer.get_density_suggestion()
        
        # 检查是否需要自动导出
        if self.auto_export_cb.isChecked():
            self.export_analysis()
    
    def handle_analysis_error(self, error_message):
        """处理分析错误"""
        self.status_label.setText("分析失败: " + error_message)
        self.progress_bar.setValue(0)
        QtWidgets.QMessageBox.critical(self, "错误", "音频分析失败：\n" + error_message)
    
    def generate_beatmap(self):
        """生成谱面 - 重定向到新方法"""
        # 重定向到新的谱面生成方法
        self.generate_beatmap_from_tab()
    
    def preview_beatmap(self):
        """预览谱面 - 重定向到新方法"""
        # 重定向到新的谱面预览方法
        self.preview_beatmap_from_tab()
    
    def export_analysis(self):
        """导出音频分析结果"""
        # 检查是否有分析结果
        if not hasattr(self, 'audio_analyzer') or not hasattr(self.audio_analyzer, 'features'):
            QtWidgets.QMessageBox.warning(self, "导出错误", "没有可用的分析结果。请先分析音频。")
            return
        
        # 获取输出目录
        output_dir = self.output_path.text()
        if not output_dir:
            # 使用原始音频文件所在目录
            output_dir = os.path.dirname(self.file_path.text()) if self.file_path.text() else "."
        
        # 获取基本文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(self.file_path.text()))[0] if self.file_path.text() else "analysis"
        
        # 显示保存对话框
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出分析结果", 
            os.path.join(output_dir, f"{base_name}_analysis.json"),
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        # 检查是否分析了所有源
        if hasattr(self.audio_analyzer, 'features') and 'all_sources_analysis' in self.audio_analyzer.features:
            # 使用新的导出方法导出所有源的分析结果
            output_path = self.audio_analyzer.export_analysis_to_json(file_path)
            
            if output_path:
                # 获取包含所有分析结果的文件夹路径
                analysis_folder = os.path.dirname(output_path)
                
                # 所有源的分析结果以及单独文件都已导出
                QtWidgets.QMessageBox.information(
                    self, "导出成功", 
                    f"所有音频源的分析结果已导出到文件夹:\n{analysis_folder}\n\n"
                    f"主索引文件: {os.path.basename(output_path)}"
                )
        else:
            # 使用原有方法导出单个活跃源的分析结果
            output_path = self.audio_analyzer.export_analysis_to_json(file_path)
            
            if output_path:
                QtWidgets.QMessageBox.information(
                    self, "导出成功", 
                    f"分析结果已导出到:\n{output_path}"
                )
    
    def toggle_visualization(self, state):
        """切换可视化功能的启用状态"""
        is_enabled = state == QtCore.Qt.Checked
        
        # 切换可视化器的可见性
        self.audio_visualizer.setVisible(is_enabled)
        self.visualization_disabled_label.setVisible(not is_enabled)
        
        # 切换相关设置的启用状态
        self.viz_quality_combo.setEnabled(is_enabled)
        self.enable_realtime_viz_cb.setEnabled(is_enabled)
        self.cache_visualizations_cb.setEnabled(is_enabled)
        
        # 如果禁用可视化，则不再将分析结果传递给可视化器以节省资源
        if not is_enabled and hasattr(self.audio_analyzer, 'features'):
            self.status_label.setText("可视化已禁用，但分析结果仍然可用")

    def browse_beatmap(self):
        """浏览选择谱面文件"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择谱面文件", "", "osu谱面文件 (*.osu)"
        )
        if file_path:
            self.beatmap_file_path.setText(file_path)
            self.beatmap_status_label.setText(f"已选择谱面文件: {os.path.basename(file_path)}")

    def analyze_beatmap(self):
        """分析谱面文件"""
        beatmap_file = self.beatmap_file_path.text()
        if not beatmap_file or not os.path.exists(beatmap_file):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的谱面文件")
            return
        
        # 清除之前的分析结果
        self.clear_beatmap_analysis_results()
        
        # 更新状态和进度条
        self.beatmap_status_label.setText("正在分析谱面...")
        self.beatmap_progress_bar.setValue(0)
        
        # 加载谱面文件
        if not self.beatmap_analyzer.load_beatmap(beatmap_file):
            self.beatmap_status_label.setText("谱面文件加载失败")
            return
        
        # 开始分析
        self.beatmap_analyzer.analyze()

    def update_beatmap_analysis_progress(self, progress):
        """更新谱面分析进度"""
        self.beatmap_progress_bar.setValue(progress)

    def handle_beatmap_analysis_complete(self, results):
        """处理谱面分析完成事件"""
        self.beatmap_status_label.setText("谱面分析完成")
        self.beatmap_progress_bar.setValue(100)
        
        # 显示分析结果
        self.display_beatmap_analysis_results(results)

    def handle_beatmap_analysis_error(self, error_message):
        """处理谱面分析错误事件"""
        self.beatmap_status_label.setText(f"谱面分析出错: {error_message}")
        QtWidgets.QMessageBox.critical(self, "分析错误", f"谱面分析过程中出错: {error_message}")

    def clear_beatmap_analysis_results(self):
        """清除谱面分析结果"""
        self.beatmap_summary_text.clear()
        self.difficulty_analysis_text.clear()
        self.pattern_analysis_text.clear()
        
        # 清除图表小部件
        if hasattr(self, 'distribution_plot'):
            self.distribution_plot.setParent(None)
        if hasattr(self, 'heatmap_plot'):
            self.heatmap_plot.setParent(None)

    def display_beatmap_analysis_results(self, results):
        """显示谱面分析结果"""
        # 更新谱面概要
        self.beatmap_summary_text.setText(self.beatmap_analyzer.get_difficulty_summary())
        
        # 更新难度分析
        if "difficulty" in results and "difficulty_rating" in results:
            difficulty = results["difficulty"]
            rating = results["difficulty_rating"]
            
            difficulty_text = f"""## 难度详细分析

### 参数值
- 接近速度(AR): {difficulty.get('AR', 0):.1f}
- 判定精度(OD): {difficulty.get('OD', 0):.1f}
- 圆圈大小(CS): {difficulty.get('CS', 0):.1f}
- 生命消耗(HP): {difficulty.get('HP', 0):.1f}
- 滑条速度: {difficulty.get('slider_multiplier', 0):.2f}x
- 滑条点击率: {difficulty.get('slider_tick_rate', 0):.1f}

### 难度评级
- 综合难度: {rating.get('overall_level', '未知')}
- 数值评分: {rating.get('numerical_rating', 0):.2f}/10
- AR评级: {rating.get('ar_rating', '未知')}
- OD评级: {rating.get('od_rating', '未知')}
- CS评级: {rating.get('cs_rating', '未知')}
- HP评级: {rating.get('hp_rating', '未知')}
"""
            self.difficulty_analysis_text.setText(difficulty_text)
        
        # 更新物件分布图
        if "distribution" in results:
            # 创建时间间隔分布图
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            distribution_fig = self.beatmap_analyzer.generate_timing_distribution()
            if distribution_fig:
                self.distribution_plot = FigureCanvas(distribution_fig)
                layout = self.distribution_widget.layout()
                if layout:
                    # 清除旧的小部件
                    while layout.count():
                        item = layout.takeAt(0)
                        widget = item.widget()
                        if widget:
                            widget.deleteLater()
                else:
                    layout = QtWidgets.QVBoxLayout(self.distribution_widget)
                
                layout.addWidget(self.distribution_plot)
        
        # 更新热图
        if "heatmap" in results:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            heatmap_fig = self.beatmap_analyzer.generate_heatmap()
            if heatmap_fig:
                self.heatmap_plot = FigureCanvas(heatmap_fig)
                layout = self.heatmap_widget.layout()
                if layout:
                    # 清除旧的小部件
                    while layout.count():
                        item = layout.takeAt(0)
                        widget = item.widget()
                        if widget:
                            widget.deleteLater()
                else:
                    layout = QtWidgets.QVBoxLayout(self.heatmap_widget)
                
                layout.addWidget(self.heatmap_plot)
        
        # 更新模式识别
        if "patterns" in results:
            patterns = results["patterns"]
            counts = patterns.get("counts", {})
            percentages = patterns.get("percentage", {})
            
            pattern_text = "## 谱面模式分析\n\n"
            pattern_text += f"主要模式: {patterns.get('primary_pattern', '未知')}\n\n"
            pattern_text += "### 模式统计\n"
            
            for pattern, count in counts.items():
                percentage = percentages.get(pattern, 0)
                pattern_text += f"- {pattern}: {count} 次 ({percentage:.1f}%)\n"
            
            if "rhythm" in results:
                rhythm = results["rhythm"]
                pattern_text += f"\n### 节奏分析\n"
                pattern_text += f"- BPM: {rhythm.get('bpm', 0):.1f}\n"
                pattern_text += f"- 每秒物件数: {rhythm.get('objects_per_second', 0):.2f}\n"
                pattern_text += f"- 每拍物件数: {rhythm.get('objects_per_beat', 0):.2f}\n"
                pattern_text += f"- 密度级别: {rhythm.get('density_level', '未知')}\n"
                pattern_text += f"- 连打段落数: {rhythm.get('stream_sections_count', 0)}\n"
            
            self.pattern_analysis_text.setText(pattern_text)

    def export_beatmap_analysis(self):
        """导出谱面分析结果"""
        if not hasattr(self.beatmap_analyzer, 'analysis_results') or not self.beatmap_analyzer.analysis_results:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可导出的谱面分析结果")
            return
        
        # 从分析结果中获取元数据用于生成文件名
        metadata = self.beatmap_analyzer.analysis_results.get("metadata", {})
        title = metadata.get("title", "unknown")
        artist = metadata.get("artist", "unknown")
        version = metadata.get("version", "unknown")
        
        # 构建文件名基础部分，替换非法字符
        def sanitize_filename(name):
            """替换文件名中的非法字符"""
            illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in illegal_chars:
                name = name.replace(char, '_')
            return name
        
        base_filename = f"{sanitize_filename(title)}_{sanitize_filename(artist)}_{sanitize_filename(version)}_分析"
        
        # 确定默认保存文件夹路径
        default_dir = ""
        if self.output_path.text():
            # 如果设置了输出目录，使用该目录
            default_dir = self.output_path.text()
        elif self.beatmap_analyzer.beatmap_path:
            # 否则使用谱面文件所在目录
            default_dir = os.path.dirname(self.beatmap_analyzer.beatmap_path)
        
        # 提供三种格式的默认文件名
        json_path = os.path.join(default_dir, f"{base_filename}.json")
        html_path = os.path.join(default_dir, f"{base_filename}.html")
        txt_path = os.path.join(default_dir, f"{base_filename}.txt")
        
        # 根据上次使用的格式选择默认文件路径
        default_path = json_path  # 默认使用JSON格式
        if "HTML" in self.last_export_format:
            default_path = html_path
        elif "文本" in self.last_export_format:
            default_path = txt_path
        
        # 定义文件类型过滤器
        file_filters = "JSON文件 (*.json);;HTML报告 (*.html);;文本文件 (*.txt)"
        
        # 选择保存路径，使用上次的格式作为默认选择
        save_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出谱面分析", default_path, file_filters, self.last_export_format
        )
        
        if not save_path:
            return
        
        # 记住当前选择的格式，用于下次导出
        self.last_export_format = selected_filter
        
        try:
            # 确保文件扩展名与选择的过滤器匹配
            _, ext = os.path.splitext(save_path)
            expected_ext = ""
            
            if "JSON" in selected_filter:
                expected_ext = ".json"
            elif "HTML" in selected_filter:
                expected_ext = ".html"
            elif "文本" in selected_filter:
                expected_ext = ".txt"
            
            # 如果用户删除了扩展名或者与选择的过滤器不匹配，则添加正确的扩展名
            if not ext or ext.lower() != expected_ext:
                save_path = save_path + expected_ext
            
            # 根据文件扩展名选择导出格式
            _, ext = os.path.splitext(save_path)
            
            if ext.lower() == '.json':
                # 导出为JSON
                import json
                # 判断是否使用美化格式
                indent = 4 if hasattr(self, 'json_pretty_rb') and self.json_pretty_rb.isChecked() else None
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.beatmap_analyzer.analysis_results, f, ensure_ascii=False, indent=indent)
                    
            elif ext.lower() == '.html':
                # 导出为HTML报告
                self.export_beatmap_analysis_as_html(save_path)
                    
            elif ext.lower() == '.txt':
                # 导出为文本文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(self.beatmap_analyzer.get_difficulty_summary())
                    
            self.beatmap_status_label.setText(f"分析结果已导出到: {os.path.basename(save_path)}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出错误", f"导出分析结果时出错: {str(e)}")

    def export_beatmap_analysis_as_html(self, save_path):
        """将谱面分析结果导出为HTML报告"""
        try:
            # 导出热图和分布图
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import base64
            from io import BytesIO
            
            results = self.beatmap_analyzer.analysis_results
            
            # 将图像转换为base64嵌入HTML
            def fig_to_base64(fig):
                buf = BytesIO()
                canvas = FigureCanvas(fig)
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                return img_base64
            
            heatmap_base64 = ""
            distribution_base64 = ""
            
            # 生成热图的base64
            heatmap_fig = self.beatmap_analyzer.generate_heatmap()
            if heatmap_fig:
                heatmap_base64 = fig_to_base64(heatmap_fig)
                
            # 生成分布图的base64
            distribution_fig = self.beatmap_analyzer.generate_timing_distribution()
            if distribution_fig:
                distribution_base64 = fig_to_base64(distribution_fig)
            
            # 元数据
            metadata = results.get("metadata", {})
            title = metadata.get("title", "未知谱面")
            artist = metadata.get("artist", "未知艺术家")
            creator = metadata.get("creator", "未知作者")
            version = metadata.get("version", "")
            
            # 难度
            difficulty = results.get("difficulty", {})
            rating = results.get("difficulty_rating", {})
            
            # 物件计数
            objects_count = results.get("objects_count", {})
            
            # 生成HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>谱面分析: {title} [{version}]</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #FF66AA; padding-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        h1, h2, h3 {{ color: #FF66AA; }}
        h1 {{ font-size: 24px; }}
        h2 {{ font-size: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .chart-container {{ display: flex; justify-content: center; margin: 20px 0; }}
        .chart {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>osu!谱面分析报告</h1>
            <h2>{title} [{version}]</h2>
            <p>艺术家: {artist} | 谱师: {creator}</p>
        </div>
        
        <div class="section">
            <h2>谱面概要</h2>
            <table>
                <tr><th>难度等级</th><td>{rating.get('overall_level', '未知')}</td></tr>
                <tr><th>难度评分</th><td>{rating.get('numerical_rating', 0):.2f}/10</td></tr>
                <tr><th>物件总数</th><td>{objects_count.get('total', 0)}</td></tr>
                <tr><th>圆圈数</th><td>{objects_count.get('circles', 0)} ({objects_count.get('circles', 0)/max(1, objects_count.get('total', 1))*100:.1f}%)</td></tr>
                <tr><th>滑条数</th><td>{objects_count.get('sliders', 0)} ({objects_count.get('sliders', 0)/max(1, objects_count.get('total', 1))*100:.1f}%)</td></tr>
                <tr><th>转盘数</th><td>{objects_count.get('spinners', 0)} ({objects_count.get('spinners', 0)/max(1, objects_count.get('total', 1))*100:.1f}%)</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>难度参数</h2>
            <table>
                <tr><th>参数</th><th>数值</th><th>评级</th></tr>
                <tr><td>接近速度(AR)</td><td>{difficulty.get('AR', 0):.1f}</td><td>{rating.get('ar_rating', '未知')}</td></tr>
                <tr><td>判定精度(OD)</td><td>{difficulty.get('OD', 0):.1f}</td><td>{rating.get('od_rating', '未知')}</td></tr>
                <tr><td>圆圈大小(CS)</td><td>{difficulty.get('CS', 0):.1f}</td><td>{rating.get('cs_rating', '未知')}</td></tr>
                <tr><td>生命消耗(HP)</td><td>{difficulty.get('HP', 0):.1f}</td><td>{rating.get('hp_rating', '未知')}</td></tr>
                <tr><td>滑条速度</td><td>{difficulty.get('slider_multiplier', 0):.2f}x</td><td>-</td></tr>
                <tr><td>滑条点击率</td><td>{difficulty.get('slider_tick_rate', 0):.1f}</td><td>-</td></tr>
            </table>
        </div>
"""
            
            # 添加热图和分布图（如果有）
            if heatmap_base64:
                html += f"""
        <div class="section">
            <h2>物件分布热图</h2>
            <div class="chart-container">
                <img class="chart" src="data:image/png;base64,{heatmap_base64}" alt="物件分布热图">
            </div>
        </div>
"""
            
            if distribution_base64:
                html += f"""
        <div class="section">
            <h2>时间间隔分布</h2>
            <div class="chart-container">
                <img class="chart" src="data:image/png;base64,{distribution_base64}" alt="时间间隔分布">
            </div>
        </div>
"""
            
            # 添加节奏分析部分
            if "rhythm" in results:
                rhythm = results.get("rhythm", {})
                html += f"""
        <div class="section">
            <h2>节奏分析</h2>
            <table>
                <tr><th>BPM</th><td>{rhythm.get('bpm', 0):.1f}</td></tr>
                <tr><th>密度等级</th><td>{rhythm.get('density_level', '未知')}</td></tr>
                <tr><th>每秒物件数</th><td>{rhythm.get('objects_per_second', 0):.2f}</td></tr>
                <tr><th>每拍物件数</th><td>{rhythm.get('objects_per_beat', 0):.2f}</td></tr>
                <tr><th>连打段落数</th><td>{rhythm.get('stream_sections_count', 0)}</td></tr>
            </table>
        </div>
"""
            
            # 添加模式分析部分
            if "patterns" in results:
                patterns = results.get("patterns", {})
                counts = patterns.get("counts", {})
                percentages = patterns.get("percentage", {})
                
                html += f"""
        <div class="section">
            <h2>模式分析</h2>
            <p>主要模式: <strong>{patterns.get('primary_pattern', '未知')}</strong></p>
            <table>
                <tr><th>模式</th><th>计数</th><th>占比</th></tr>
"""
                
                for pattern, count in counts.items():
                    percentage = percentages.get(pattern, 0)
                    html += f"                <tr><td>{pattern}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>\n"
                
                html += "            </table>\n        </div>\n"
            
            # 结束HTML
            html += """
    </div>
</body>
</html>
"""
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
        except Exception as e:
            raise Exception(f"生成HTML报告时出错: {str(e)}")

    def browse_dataset_folder(self):
        """浏览数据集文件夹"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择谱面文件夹", os.path.expanduser("~")
        )
        if folder_path:
            self.dataset_folder_path.setText(folder_path)
            self.dataset_status_label.setText(f"已选择文件夹: {folder_path}")
    
    def browse_dataset_output(self):
        """浏览数据集输出文件夹"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择数据集输出文件夹", os.path.expanduser("~")
        )
        if folder_path:
            self.dataset_output_path.setText(folder_path)
    
    def scan_dataset_folder(self):
        """扫描数据集文件夹中的谱面文件"""
        folder_path = self.dataset_folder_path.text().strip()
        if not folder_path or not os.path.isdir(folder_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的谱面文件夹!")
            return
        
        selected_mode = self.mode_combo.currentText()
        selected_difficulty = self.difficulty_combo.currentText()
        limit = self.files_limit_spin.value()
        
        self.dataset_files_list.clear()
        self.dataset_progress_bar.setValue(0)
        self.dataset_status_label.setText("正在扫描文件夹...")
        
        # 开始扫描
        found_files = []
        total_scanned = 0
        skipped_files = 0
        
        # 难度范围映射
        difficulty_ranges = {
            "所有难度": (0, 10),
            "Easy": (0, 2.5),
            "Normal": (2.5, 4),
            "Hard": (4, 5.5),
            "Insane": (5.5, 7),
            "Expert": (7, 8.5),
            "Expert+": (8.5, 10)
        }
        
        # 获取当前难度的范围
        min_diff, max_diff = difficulty_ranges.get(selected_difficulty, (0, 10))
        
        # 模式映射 (文件中的Mode值对应的游戏模式)
        mode_mapping = {
            "0": "std",   # osu!standard
            "1": "taiko",  # osu!taiko
            "2": "catch",  # osu!catch
            "3": "mania"   # osu!mania
        }
        
        # mania模式中的键数标记 (文件名中通常出现的标记)
        mania_key_patterns = [
            "[1K", "[2K", "[3K", "[4K", "[5K", "[6K", "[7K", "[8K", "[9K", "[10K",
            "1K]", "2K]", "3K]", "4K]", "5K]", "6K]", "7K]", "8K]", "9K]", "10K]"
        ]
        
        # 递归扫描文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".osu"):
                    total_scanned += 1
                    file_path = os.path.join(root, file)
                    
                    # 简单检查文件内容以确定它是否符合要求
                    try:
                        # 首先检查文件名中是否有模式标记 (快速过滤)
                        file_name = os.path.basename(file_path)
                        
                        # 如果选择的是std模式，但文件名中包含其他模式的标记，则跳过
                        if selected_mode == "std":
                            # 检查文件名中是否有mania的键数标记
                            if any(key_pattern in file_name for key_pattern in mania_key_patterns):
                                skipped_files += 1
                                continue
                            
                            # 检查文件名中是否有taiko或catch的标记
                            if "taiko" in file_name.lower() or "catch" in file_name.lower():
                                skipped_files += 1
                                continue
                        
                        # 如果选择的是mania模式，但文件名中没有键数标记，可能需要检查文件内容
                        if selected_mode == "mania" and not any(key_pattern in file_name for key_pattern in mania_key_patterns):
                            # 这里暂不跳过，继续检查文件内容
                            pass
                        
                        # 读取文件内容进行详细检查
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # 检查模式
                            mode_line = [line for line in content.split("\n") if "Mode:" in line]
                            if mode_line:
                                file_mode_num = mode_line[0].split(":")[-1].strip()
                                file_mode = mode_mapping.get(file_mode_num, "unknown")
                                
                                # 如果选择的模式与文件模式不匹配，则跳过
                                if selected_mode != file_mode:
                                    skipped_files += 1
                                    continue
                            
                            # 如果不是"所有难度"，则检查谱面难度是否在范围内
                            if selected_difficulty != "所有难度":
                                # 检查难度
                                difficulty_lines = [
                                    line for line in content.split("\n") 
                                    if any(keyword in line for keyword in ["OverallDifficulty:", "HPDrainRate:", "CircleSize:", "ApproachRate:"])
                                ]
                                
                                if difficulty_lines:
                                    # 获取难度值
                                    diff_values = []
                                    for line in difficulty_lines:
                                        try:
                                            diff_value = float(line.split(":")[-1].strip())
                                            diff_values.append(diff_value)
                                        except:
                                            pass
                                    
                                    # 如果有难度值，计算平均难度
                                    if diff_values:
                                        avg_difficulty = sum(diff_values) / len(diff_values)
                                        if avg_difficulty < min_diff or avg_difficulty > max_diff:
                                            continue
                                
                                # 另外尝试从谱面名称判断难度
                                version_lines = [line for line in content.split("\n") if "Version:" in line]
                                if version_lines:
                                    version = version_lines[0].split(":")[-1].strip().lower()
                                    
                                    # 根据谱面名称匹配难度
                                    difficulty_keywords = {
                                        "easy": "Easy",
                                        "normal": "Normal",
                                        "hard": "Hard",
                                        "insane": "Insane",
                                        "expert": "Expert",
                                        "extreme": "Expert+",
                                        "extra": "Expert"
                                    }
                                    
                                    # 检查谱面名称是否包含难度关键词
                                    matched_difficulty = None
                                    for keyword, diff in difficulty_keywords.items():
                                        if keyword in version:
                                            matched_difficulty = diff
                                            break
                                    
                                    # 如果谱面名称匹配到难度关键词，但与选择的难度不匹配，则跳过
                                    if matched_difficulty and matched_difficulty != selected_difficulty:
                                        continue
                            
                            # 添加到发现的文件列表
                            found_files.append(file_path)
                            self.dataset_files_list.addItem(file_path)
                            
                            # 限制文件数量
                            if len(found_files) >= limit:
                                break
                    except Exception as e:
                        print(f"无法处理文件 {file_path}: {str(e)}")
                
                # 更新进度条
                self.dataset_progress_bar.setValue(min(100, int(total_scanned / 1000 * 100)))
                QtCore.QCoreApplication.processEvents()
                
                # 达到限制就停止
                if len(found_files) >= limit:
                    break
            
            # 达到限制就停止
            if len(found_files) >= limit:
                break
        
        # 更新状态
        self.dataset_progress_bar.setValue(100)
        self.dataset_status_label.setText(f"找到 {len(found_files)} 个符合条件的谱面文件，跳过了 {skipped_files} 个不匹配谱面")
    
    def process_dataset(self):
        """处理数据集中的谱面文件"""
        if self.dataset_files_list.count() == 0:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可处理的文件，请先扫描文件夹!")
            return
        
        output_path = self.dataset_output_path.text().strip()
        if not output_path or not os.path.isdir(output_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请选择有效的输出文件夹!")
            return
        
        # 创建处理结果目录
        dataset_output_dir = os.path.join(output_path, "beatmap_dataset")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 创建临时批次目录
        batches_dir = os.path.join(dataset_output_dir, "batches")
        os.makedirs(batches_dir, exist_ok=True)
        
        # 获取所有文件路径
        beatmap_files = []
        for i in range(self.dataset_files_list.count()):
            beatmap_files.append(self.dataset_files_list.item(i).text())
        
        total_files = len(beatmap_files)
        processed_files = 0
        all_results = []
        
        # 添加错误文件夹跟踪
        error_folders = set()
        skipped_files = 0
        
        # 获取GPU加速设置
        use_gpu = self.dataset_use_gpu_checkbox.isChecked() and self.dataset_use_gpu_checkbox.isEnabled()
        
        # 如果使用GPU加速，设置GPU设备
        if use_gpu:
            gpu_device_idx = self.dataset_gpu_device_combo.currentIndex()
            if torch.cuda.is_available() and gpu_device_idx < torch.cuda.device_count():
                torch.cuda.set_device(gpu_device_idx)
                self.dataset_status_label.setText(f"已启用GPU加速：{torch.cuda.get_device_name(gpu_device_idx)}")
            else:
                use_gpu = False
                self.dataset_status_label.setText("GPU设备设置失败，改用CPU处理")
        
        # 设置音频分析器使用GPU
        self.audio_analyzer.set_use_gpu(use_gpu)
        
        # 设置每批处理的文件数
        batch_size = self.batch_size_spin.value()
        total_batches = (total_files + batch_size - 1) // batch_size  # 向上取整计算批次数
        
        # 更新状态
        if use_gpu:
            self.dataset_status_label.setText(f"已启用GPU加速：{torch.cuda.get_device_name(gpu_device_idx)}\n"
                                             f"准备处理 {total_files} 个文件，共 {total_batches} 批")
        else:
            self.dataset_status_label.setText(f"使用CPU处理 {total_files} 个文件，共 {total_batches} 批")
        QtCore.QCoreApplication.processEvents()
        
        for batch_index in range(total_batches):
            # 获取当前批次的文件
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = beatmap_files[start_idx:end_idx]
            
            batch_results = []
            
            # 更新状态
            self.dataset_status_label.setText(f"正在处理批次 {batch_index+1}/{total_batches}..."
                                             f"{' (GPU加速)' if use_gpu else ''}")
            self.dataset_progress_bar.setValue(int(batch_index / total_batches * 100))
            QtCore.QCoreApplication.processEvents()
            
            # 处理当前批次的文件
            for file_idx, file_path in enumerate(batch_files):
                try:
                    # 获取所在的文件夹路径
                    folder_path = os.path.dirname(file_path)
                    
                    # 如果文件所在文件夹已标记为错误，则跳过
                    if folder_path in error_folders:
                        skipped_files += 1
                        processed_files += 1
                        continue
                    
                    # 更新状态
                    file_name = os.path.basename(file_path)
                    self.dataset_status_label.setText(
                        f"批次 {batch_index+1}/{total_batches}, "
                        f"文件 {file_idx+1}/{len(batch_files)}: {file_name}"
                    )
                    QtCore.QCoreApplication.processEvents()
                    
                    # 加载谱面文件
                    if self.beatmap_analyzer.load_beatmap(file_path):
                        # 分析谱面
                        result = self.beatmap_analyzer.analyze()
                        
                        # 尝试找到关联的音频文件
                        audio_file = None
                        try:
                            # 从谱面内容中获取音频文件名
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                audio_lines = [line for line in content.split("\n") if "AudioFilename:" in line]
                                if audio_lines:
                                    audio_filename = audio_lines[0].split(":")[-1].strip()
                                    beatmap_dir = os.path.dirname(file_path)
                                    audio_file = os.path.join(beatmap_dir, audio_filename)
                                    
                                    # 检查文件是否存在
                                    if not os.path.exists(audio_file):
                                        audio_file = None
                        except:
                            audio_file = None
                        
                        # 如果找到音频文件，则进行音频分析
                        if audio_file and os.path.exists(audio_file):
                            try:
                                if self.audio_analyzer.load_audio(audio_file):
                                    audio_features = self.audio_analyzer.analyze()
                                    result["audio_features"] = audio_features
                            except Exception as e:
                                # 音频分析出错，记录错误并将整个文件夹标记为跳过
                                error_message = f"无法分析音频文件 {audio_file}: {str(e)}"
                                print(error_message)
                                self.dataset_status_label.setText(f"错误: {error_message}\n将跳过文件夹: {folder_path}")
                                QtCore.QCoreApplication.processEvents()
                                
                                # 将文件夹添加到错误文件夹集合
                                error_folders.add(folder_path)
                                # 不添加到结果中，直接跳过
                                continue
                        
                        # 将结果添加到批次结果和总结果
                        item_result = {
                            "beatmap_file": file_path,
                            "audio_file": audio_file,
                            "analysis": result
                        }
                        batch_results.append(item_result)
                        all_results.append(item_result)
                
                except Exception as e:
                    # 处理文件出错，记录错误并将整个文件夹标记为跳过
                    folder_path = os.path.dirname(file_path)
                    error_message = f"处理文件 {file_path} 时出错: {str(e)}"
                    print(error_message)
                    self.dataset_status_label.setText(f"错误: {error_message}\n将跳过文件夹: {folder_path}")
                    QtCore.QCoreApplication.processEvents()
                    
                    # 将文件夹添加到错误文件夹集合
                    error_folders.add(folder_path)
                
                processed_files += 1
                
                # 更新子进度
                sub_progress = int((file_idx + 1) / len(batch_files) * 100)
                self.dataset_progress_bar.setValue(
                    int((batch_index * 100 + sub_progress) / total_batches)
                )
                QtCore.QCoreApplication.processEvents()
            
            # 保存当前批次结果
            batch_file = os.path.join(batches_dir, f"batch_{batch_index+1}.json")
            try:
                with open(batch_file, "w", encoding="utf-8") as f:
                    json.dump(batch_results, f, indent=2)
                    
                self.dataset_status_label.setText(
                    f"批次 {batch_index+1}/{total_batches} 已完成并保存。"
                    f"已处理: {processed_files}/{total_files} 个文件，"
                    f"跳过了 {skipped_files} 个文件"
                )
                QtCore.QCoreApplication.processEvents()
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "警告", 
                    f"保存批次 {batch_index+1} 时出错: {str(e)}"
                )
        
        # 保存错误文件夹日志
        if error_folders:
            error_log_path = os.path.join(dataset_output_dir, "error_folders.txt")
            try:
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"处理过程中跳过了 {len(error_folders)} 个错误文件夹:\n\n")
                    for folder in sorted(error_folders):
                        f.write(f"{folder}\n")
            except Exception as e:
                print(f"保存错误文件夹日志出错: {str(e)}")
        
        # 保存完整数据集
        try:
            # 检查是否需要分割数据集
            is_split_enabled = self.enable_split_check.isChecked()
            
            if is_split_enabled:
                # 获取分割比例
                train_percent = self.train_percent_spin.value() / 100.0
                val_percent = self.val_percent_spin.value() / 100.0
                test_percent = self.test_percent_spin.value() / 100.0
                
                # 获取分割方式
                split_method = self.split_method_combo.currentText()
                
                # 创建子目录
                train_dir = os.path.join(dataset_output_dir, "train")
                val_dir = os.path.join(dataset_output_dir, "val")
                test_dir = os.path.join(dataset_output_dir, "test")
                
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)
                
                # 根据选择的分割方式进行不同处理
                if split_method == "按文件夹分割":
                    # 按文件夹分组谱面
                    folder_groups = {}
                    for item in all_results:
                        beatmap_file = item["beatmap_file"]
                        # 获取所在的文件夹路径
                        folder_path = os.path.dirname(beatmap_file)
                        
                        if folder_path not in folder_groups:
                            folder_groups[folder_path] = []
                        
                        folder_groups[folder_path].append(item)
                    
                    # 随机打乱文件夹列表（不打乱文件夹内的文件）
                    import random
                    folder_paths = list(folder_groups.keys())
                    
                    # 确保不包含错误文件夹
                    valid_folder_paths = [path for path in folder_paths if path not in error_folders]
                    if len(valid_folder_paths) < len(folder_paths):
                        self.dataset_status_label.setText(
                            f"数据集分割时排除了 {len(folder_paths) - len(valid_folder_paths)} 个错误文件夹"
                        )
                        QtCore.QCoreApplication.processEvents()
                        folder_paths = valid_folder_paths
                    
                    random.shuffle(folder_paths)
                    
                    # 计算每个集合应包含的文件夹数量
                    total_folders = len(folder_paths)
                    if total_folders == 0:
                        QtWidgets.QMessageBox.warning(
                            self, 
                            "警告", 
                            "所有文件夹都有错误，无法进行数据集分割!"
                        )
                        return
                    
                    train_folders_count = max(1, int(total_folders * train_percent))
                    val_folders_count = max(1, int(total_folders * val_percent))
                    
                    # 确保分割后的文件夹数量不超过总文件夹数
                    if train_folders_count + val_folders_count > total_folders:
                        # 按比例调整
                        total_ratio = train_percent + val_percent
                        train_folders_count = max(1, int(total_folders * (train_percent / total_ratio)))
                        val_folders_count = max(1, int(total_folders * (val_percent / total_ratio)))
                        
                        # 进一步确保不超过总数
                        if train_folders_count + val_folders_count > total_folders:
                            val_folders_count = max(1, total_folders - train_folders_count)
                    
                    # 分割文件夹到训练集、验证集和测试集
                    train_folders = folder_paths[:train_folders_count]
                    val_folders = folder_paths[train_folders_count:train_folders_count + val_folders_count]
                    test_folders = folder_paths[train_folders_count + val_folders_count:]
                    
                    # 根据文件夹分组整理数据
                    train_data = []
                    for folder in train_folders:
                        train_data.extend(folder_groups[folder])
                        
                    val_data = []
                    for folder in val_folders:
                        val_data.extend(folder_groups[folder])
                        
                    test_data = []
                    for folder in test_folders:
                        test_data.extend(folder_groups[folder])
                    
                    # 保存分割信息以便查看
                    split_info = {
                        "method": "按文件夹分割",
                        "train_folders": train_folders,
                        "val_folders": val_folders,
                        "test_folders": test_folders,
                        "train_percent": train_percent,
                        "val_percent": val_percent,
                        "test_percent": test_percent,
                        "total_folders": total_folders,
                        "total_files": total_files
                    }
                    
                    with open(os.path.join(dataset_output_dir, "split_info.json"), "w", encoding="utf-8") as f:
                        json.dump(split_info, f, indent=2)
                    
                    success_message = (
                        f"成功处理 {processed_files} 个文件!\n"
                        f"数据集已按文件夹分割并保存到以下目录:\n"
                        f"- 训练集 ({len(train_data)}个样本，{len(train_folders)}个文件夹): {train_dir}\n"
                        f"- 验证集 ({len(val_data)}个样本，{len(val_folders)}个文件夹): {val_dir}\n"
                        f"- 测试集 ({len(test_data)}个样本，{len(test_folders)}个文件夹): {test_dir}\n"
                        f"分割信息已保存至: {os.path.join(dataset_output_dir, 'split_info.json')}"
                    )
                    
                else:  # "随机分割"
                    # 随机打乱数据
                    import random
                    random.shuffle(all_results)
                    
                    # 计算每个集合的大小
                    total_samples = len(all_results)
                    train_size = int(total_samples * train_percent)
                    val_size = int(total_samples * val_percent)
                    
                    # 分割数据集
                    train_data = all_results[:train_size]
                    val_data = all_results[train_size:train_size + val_size]
                    test_data = all_results[train_size + val_size:]
                    
                    # 保存分割信息
                    split_info = {
                        "method": "随机分割",
                        "train_percent": train_percent,
                        "val_percent": val_percent,
                        "test_percent": test_percent,
                        "train_size": len(train_data),
                        "val_size": len(val_data),
                        "test_size": len(test_data),
                        "total_files": total_files
                    }
                    
                    with open(os.path.join(dataset_output_dir, "split_info.json"), "w", encoding="utf-8") as f:
                        json.dump(split_info, f, indent=2)
                    
                    success_message = (
                        f"成功处理 {processed_files} 个文件!\n"
                        f"数据集已随机分割并保存到以下目录:\n"
                        f"- 训练集 ({len(train_data)}个样本): {train_dir}\n"
                        f"- 验证集 ({len(val_data)}个样本): {val_dir}\n"
                        f"- 测试集 ({len(test_data)}个样本): {test_dir}\n"
                        f"分割信息已保存至: {os.path.join(dataset_output_dir, 'split_info.json')}"
                    )
                
                # 保存训练集
                train_file = os.path.join(train_dir, "dataset.json")
                with open(train_file, "w", encoding="utf-8") as f:
                    json.dump(train_data, f, indent=2)
                
                # 保存验证集
                val_file = os.path.join(val_dir, "dataset.json")
                with open(val_file, "w", encoding="utf-8") as f:
                    json.dump(val_data, f, indent=2)
                
                # 保存测试集
                test_file = os.path.join(test_dir, "dataset.json")
                with open(test_file, "w", encoding="utf-8") as f:
                    json.dump(test_data, f, indent=2)
                
                # 保存完整数据集（可选）
                full_dataset_file = os.path.join(dataset_output_dir, "dataset_full.json")
                with open(full_dataset_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                
                # 添加错误文件夹的统计信息
                error_folder_count = len(error_folders)
                success_message = (
                    f"数据集处理完成，已分割为训练集({len(train_data)}个样本)、"
                    f"验证集({len(val_data)}个样本)和测试集({len(test_data)}个样本)\n"
                    f"处理过程中跳过了 {skipped_files} 个文件，{error_folder_count} 个错误文件夹"
                )
                
                self.dataset_status_label.setText(success_message)
            else:
                # 不分割，直接保存完整数据集
                dataset_file = os.path.join(dataset_output_dir, "dataset.json")
                with open(dataset_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                
                # 添加错误文件夹的统计信息
                error_folder_count = len(error_folders)
                success_message = (
                    f"数据集处理完成，已保存到: {dataset_file}\n"
                    f"成功处理 {processed_files - skipped_files} 个文件，"
                    f"跳过了 {skipped_files} 个文件，{error_folder_count} 个错误文件夹"
                )
                
                self.dataset_status_label.setText(success_message)
            
            self.dataset_progress_bar.setValue(100)
            
            QtWidgets.QMessageBox.information(
                self, "处理完成", success_message
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "错误", 
                f"保存数据集时出错: {str(e)}"
            )
    
    def export_dataset(self):
        """导出处理后的数据集"""
        output_path = self.dataset_output_path.text().strip()
        if not output_path or not os.path.isdir(output_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的输出文件夹!")
            return
        
        dataset_dir = os.path.join(output_path, "beatmap_dataset")
        if not os.path.exists(dataset_dir):
            QtWidgets.QMessageBox.warning(self, "警告", "未找到处理好的数据集目录，请先处理数据集!")
            return
        
        # 检查是否有分割后的数据集或批次数据
        is_split = os.path.exists(os.path.join(dataset_dir, "train")) and \
                  os.path.exists(os.path.join(dataset_dir, "val")) and \
                  os.path.exists(os.path.join(dataset_dir, "test"))
        
        batches_dir = os.path.join(dataset_dir, "batches")
        has_batches = os.path.exists(batches_dir) and len(os.listdir(batches_dir)) > 0
        
        # 选择导出位置
        export_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出数据集", os.path.expanduser("~") + "/dataset_export.zip", "压缩文件 (*.zip)"
        )
        
        if not export_path:
            return
        
        try:
            import shutil
            
            # 创建一个临时目录用于整理要导出的文件
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            if is_split:
                # 复制分割后的数据集文件
                train_dir = os.path.join(temp_dir, "train")
                val_dir = os.path.join(temp_dir, "val")
                test_dir = os.path.join(temp_dir, "test")
                
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)
                
                # 复制训练集
                train_src = os.path.join(dataset_dir, "train", "dataset.json")
                if os.path.exists(train_src):
                    shutil.copy2(train_src, os.path.join(train_dir, "dataset.json"))
                
                # 复制验证集
                val_src = os.path.join(dataset_dir, "val", "dataset.json")
                if os.path.exists(val_src):
                    shutil.copy2(val_src, os.path.join(val_dir, "dataset.json"))
                
                # 复制测试集
                test_src = os.path.join(dataset_dir, "test", "dataset.json")
                if os.path.exists(test_src):
                    shutil.copy2(test_src, os.path.join(test_dir, "dataset.json"))
                
                # 创建README文件，说明数据集结构
                with open(os.path.join(temp_dir, "README.txt"), "w", encoding="utf-8") as f:
                    f.write("OSU谱面数据集\n")
                    f.write("===========\n\n")
                    f.write("本数据集包含以下部分：\n")
                    f.write("- train/: 训练集\n")
                    f.write("- val/: 验证集\n")
                    f.write("- test/: 测试集\n\n")
                    f.write("每个子集都包含一个dataset.json文件，其中包含谱面分析和音频特征提取的结果。\n")
            else:
                # 复制单一数据集文件或批次文件
                dataset_file = os.path.join(dataset_dir, "dataset.json")
                
                if has_batches:
                    # 如果有批次数据，导出批次数据
                    batches_export_dir = os.path.join(temp_dir, "batches")
                    os.makedirs(batches_export_dir, exist_ok=True)
                    
                    # 复制所有批次文件
                    batch_files = [f for f in os.listdir(batches_dir) if f.endswith('.json')]
                    for batch_file in batch_files:
                        src_path = os.path.join(batches_dir, batch_file)
                        dst_path = os.path.join(batches_export_dir, batch_file)
                        shutil.copy2(src_path, dst_path)
                    
                    # 如果存在完整数据集，也导出它
                    full_dataset = os.path.join(dataset_dir, "dataset.json")
                    if os.path.exists(full_dataset):
                        shutil.copy2(full_dataset, os.path.join(temp_dir, "dataset_full.json"))
                    
                    with open(os.path.join(temp_dir, "README.txt"), "w", encoding="utf-8") as f:
                        f.write("OSU谱面数据集 (批次处理)\n")
                        f.write("=================\n\n")
                        f.write("本数据集包含以下内容：\n")
                        f.write("- batches/: 包含分批处理的数据集文件\n")
                        if os.path.exists(full_dataset):
                            f.write("- dataset_full.json: 完整的合并数据集\n\n")
                        f.write("每个批次文件包含一部分谱面分析和音频特征提取的结果。\n")
                else:
                    # 只有单一数据集文件
                    if os.path.exists(dataset_file):
                        shutil.copy2(dataset_file, os.path.join(temp_dir, "dataset.json"))
                
                    # 创建README文件
                    with open(os.path.join(temp_dir, "README.txt"), "w", encoding="utf-8") as f:
                        f.write("OSU谱面数据集\n")
                        f.write("===========\n\n")
                        f.write("本数据集包含一个dataset.json文件，其中包含谱面分析和音频特征提取的结果。\n")
            
            # 创建压缩文件
            shutil.make_archive(export_path.replace(".zip", ""), 'zip', temp_dir)
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            QtWidgets.QMessageBox.information(
                self, "导出成功", 
                f"数据集已成功导出到: {export_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "错误", 
                f"导出数据集时出错: {str(e)}"
            )
    
    def adjust_split_percentages(self):
        """调整数据集分割的百分比，确保总和为100%"""
        # 获取当前百分比
        train_percent = self.train_percent_spin.value()
        val_percent = self.val_percent_spin.value()
        test_percent = self.test_percent_spin.value()
        
        # 计算总和
        total = train_percent + val_percent + test_percent
        
        # 如果总和不是100%，调整百分比
        if total != 100:
            # 确定是哪个触发了变更
            sender = self.sender()
            
            if sender == self.train_percent_spin:
                # 训练集被更改，平衡验证集和测试集
                remaining = 100 - train_percent
                ratio = val_percent / (val_percent + test_percent) if (val_percent + test_percent) > 0 else 0.5
                new_val = int(remaining * ratio)
                new_test = remaining - new_val
                
                self.val_percent_spin.blockSignals(True)
                self.test_percent_spin.blockSignals(True)
                self.val_percent_spin.setValue(new_val)
                self.test_percent_spin.setValue(new_test)
                self.val_percent_spin.blockSignals(False)
                self.test_percent_spin.blockSignals(False)
                
            elif sender == self.val_percent_spin:
                # 验证集被更改，调整测试集
                new_test = 100 - train_percent - val_percent
                
                self.test_percent_spin.blockSignals(True)
                self.test_percent_spin.setValue(new_test)
                self.test_percent_spin.blockSignals(False)
                
            elif sender == self.test_percent_spin:
                # 测试集被更改，调整验证集
                new_val = 100 - train_percent - test_percent
                
                self.val_percent_spin.blockSignals(True)
                self.val_percent_spin.setValue(new_val)
                self.val_percent_spin.blockSignals(False)

    # 模型训练部分的方法开始
    def start_training(self):
        """开始训练模型"""
        # 检查是否安装了PyTorch
        if not TORCH_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, 
                "PyTorch未安装", 
                "训练功能需要PyTorch库。请使用命令安装：\npip install torch torchvision torchaudio\n\n更多信息请访问: https://pytorch.org/get-started/locally/"
            )
            return
            
        # 检查输入
        dataset_root_path = self.dataset_root_path.text().strip()
        model_output_path = self.model_output_path.text().strip()
        
        if not dataset_root_path or not os.path.isdir(dataset_root_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请选择有效的数据集根目录!")
            return
            
        # 检查是否已检测到必要的子文件夹
        if not hasattr(self, 'train_data_path') or not self.train_data_path or not os.path.isdir(self.train_data_path):
            QtWidgets.QMessageBox.warning(self, "警告", "未检测到训练集文件夹(train)，请选择包含训练数据的正确目录!")
            return
            
        if self.use_early_stopping.isChecked() and (not hasattr(self, 'val_data_path') or not self.val_data_path or not os.path.isdir(self.val_data_path)):
            QtWidgets.QMessageBox.warning(self, "警告", "使用早停功能需要验证集文件夹(val)，请选择包含验证数据的正确目录!")
            return
            
        if not model_output_path or not os.path.isdir(model_output_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请选择有效的模型保存路径!")
            return
        
        # 获取训练参数
        model_architecture = self.model_architecture_combo.currentText()
        batch_size = self.batch_size_spin.value()
        learning_rate = float(self.learning_rate_combo.currentText())
        epochs = self.epochs_spin.value()
        use_early_stopping = self.use_early_stopping.isChecked()
        use_checkpoint = self.use_checkpoint.isChecked()
        use_mixed_precision = self.use_mixed_precision.isChecked()
        
        # 获取GPU相关设置
        use_gpu = self.use_gpu_checkbox.isChecked() and self.use_gpu_checkbox.isEnabled()
        gpu_device = 0  # 默认使用第一个GPU
        
        if use_gpu and self.gpu_device_combo.isEnabled() and self.gpu_device_combo.currentText() != "无可用设备":
            # 从字符串 "GPU x: xxxxx" 中提取设备ID
            try:
                device_text = self.gpu_device_combo.currentText()
                gpu_device = int(device_text.split(':')[0].replace('GPU', '').strip())
            except:
                gpu_device = 0
        
        # 显示训练配置确认对话框
        config_msg = f"""
训练配置:
- 数据集根目录: {dataset_root_path}
- 训练集: {os.path.basename(self.train_data_path)}
- 验证集: {os.path.basename(self.val_data_path) if hasattr(self, 'val_data_path') and self.val_data_path else "无"}
- 测试集: {os.path.basename(self.test_data_path) if hasattr(self, 'test_data_path') and self.test_data_path else "无"}
- 模型架构: {model_architecture}
- 批次大小: {batch_size}
- 学习率: {learning_rate}
- 训练周期: {epochs}
- 使用GPU: {'是' if use_gpu else '否'}
- {'GPU设备: ' + self.gpu_device_combo.currentText() if use_gpu else ''}
- 混合精度: {'是' if use_mixed_precision else '否'}
- 早停: {'是' if use_early_stopping else '否'}
- 检查点: {'是' if use_checkpoint else '否'}

确认开始训练?
        """
        
        reply = QtWidgets.QMessageBox.question(
            self, "训练确认", config_msg, 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.No:
            return
            
        # 清理之前的训练曲线图
        for i in reversed(range(self.training_plot_layout.count())): 
            widget = self.training_plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # 创建训练曲线图表
        self.train_canvas = self.create_training_figure()
        self.training_plot_layout.addWidget(self.train_canvas)
        
        # 清空训练日志
        self.training_log.clear()
        self.add_training_log("准备开始训练...")
        
        if use_gpu:
            self.add_training_log("GPU加速已启用")
            if use_mixed_precision:
                self.add_training_log("混合精度训练已启用")
        
        # 更新UI状态
        self.training_status_label.setText(f"训练正在进行 - {model_architecture} 模型")
        self.training_progress_bar.setValue(0)
        self.start_training_btn.setEnabled(False)
        self.pause_resume_btn.setEnabled(True)
        self.pause_resume_btn.setText("暂停训练")
        self.stop_training_btn.setEnabled(True)
        
        # 创建训练参数字典
        training_params = {
            'model_architecture': model_architecture,
            'training_data_path': self.train_data_path,
            'validation_data_path': self.val_data_path if hasattr(self, 'val_data_path') else "",
            'test_data_path': self.test_data_path if hasattr(self, 'test_data_path') else "",
            'model_output_path': model_output_path,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'use_early_stopping': use_early_stopping,
            'use_checkpoint': use_checkpoint,
            'use_mixed_precision': use_mixed_precision,
            'use_gpu': use_gpu,
            'gpu_device': gpu_device
        }
        
        # 创建并启动训练线程
        self.training_thread = TrainingThread(
            dataset_root=dataset_root_path,
            model_save_path=model_output_path,
            model_type=model_architecture,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            use_early_stopping=use_early_stopping,
            use_checkpoint=use_checkpoint,
            use_mixed_precision=use_mixed_precision
        )
        
        # 连接信号
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.epoch_completed.connect(self.update_training_plot)
        self.training_thread.training_finished.connect(self.handle_training_finished)
        self.training_thread.log_message.connect(self.add_training_log)
        self.training_thread.status_updated.connect(self.update_training_status)
        self.training_thread.plot_updated.connect(self.update_plot_from_pixmap)
        
        # 启动线程
        self.training_thread.start()

    def add_training_log(self, message):
        """添加训练日志"""
        self.training_log.append(message)
        # 滚动到底部
        self.training_log.verticalScrollBar().setValue(
            self.training_log.verticalScrollBar().maximum()
        )
        
    def update_training_progress(self, progress):
        """更新训练进度条"""
        self.training_progress_bar.setValue(progress)
        
    def update_training_plot(self, epoch, train_loss, val_loss=None):
        """更新训练图表"""
        try:
            # 获取图和轴
            fig = self.train_canvas.figure
            ax = fig.axes[0]
            
            # 获取当前数据
            lines = ax.get_lines()
            train_line = lines[0]
            epochs = list(train_line.get_xdata())
            train_losses = list(train_line.get_ydata())
            
            # 添加新数据点
            epochs.append(epoch)
            train_losses.append(train_loss)
            
            # 更新训练损失线
            train_line.set_xdata(epochs)
            train_line.set_ydata(train_losses)
            
            # 更新验证损失线（如果有）
            if val_loss is not None and len(lines) > 1:
                val_line = lines[1]
                val_losses = list(val_line.get_ydata())
                val_losses.append(val_loss)
                val_line.set_xdata(epochs)
                val_line.set_ydata(val_losses)
            
            # 调整坐标轴
            ax.relim()
            ax.autoscale_view()
            
            # 刷新画布
            self.train_canvas.draw()
            
        except Exception as e:
            self.add_training_log(f"更新图表出错: {str(e)}")
            
    def toggle_training_pause(self):
        """切换训练暂停/继续状态"""
        if not hasattr(self, 'training_thread') or not self.training_thread.isRunning():
            return
            
        if self.training_thread.is_paused:
            # 恢复训练
            self.training_thread.resume_training()
            self.pause_resume_btn.setText("暂停训练")
            self.training_status_label.setText("训练正在进行")
            self.add_training_log("训练已恢复")
        else:
            # 暂停训练
            self.training_thread.pause_training()
            self.pause_resume_btn.setText("继续训练")
            self.training_status_label.setText("训练已暂停")
            self.add_training_log("训练已暂停")
            
    def stop_training(self):
        """停止训练过程"""
        if not hasattr(self, 'training_thread') or not self.training_thread.isRunning():
            return
            
        reply = QtWidgets.QMessageBox.question(
            self, "停止训练", "确定要停止当前训练进程吗? 这将丢失未保存的训练进度。", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # 停止训练线程
            self.training_thread.stop_training()
            self.add_training_log("正在停止训练...")
            
            # 等待线程结束
            if self.training_thread.isRunning():
                self.training_thread.wait(2000)  # 等待最多2秒
            
            # 更新UI状态
            self.pause_resume_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(False)
            self.start_training_btn.setEnabled(True)
            self.training_status_label.setText("训练已停止")
            self.add_training_log("训练已停止")
            
    def handle_training_finished(self, success, message):
        """处理训练完成事件"""
        if success:
            self.training_status_label.setText("训练已完成")
            self.add_training_log(f"训练成功: {message}")
        else:
            self.training_status_label.setText("训练失败")
            self.add_training_log(f"训练失败: {message}")
        
        # 更新UI状态
        self.pause_resume_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(False)
        self.start_training_btn.setEnabled(True)
        
        # 显示完成通知
        QtWidgets.QMessageBox.information(self, "训练状态", message)
    
    def create_training_figure(self):
        """创建训练曲线图表"""
        # 导入matplotlib用于绘图
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(fig)
        
        # 创建训练损失和验证损失的子图
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        
        # 初始化空数据
        epochs = []
        train_losses = []
        val_losses = []
        
        # 绘制初始曲线
        train_line, = ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        val_line, = ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        ax.legend()
        fig.tight_layout()
        
        return canvas
    
    def export_model(self):
        """导出训练好的模型"""
        # 检查是否安装了PyTorch
        if not TORCH_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, 
                "PyTorch未安装", 
                "模型导出功能需要PyTorch库。请使用命令安装：\npip install torch torchvision torchaudio\n\n更多信息请访问: https://pytorch.org/get-started/locally/"
            )
            return
            
        # 检查是否有训练好的模型
        # 在实际项目中，应检查模型是否已加载
        
        # 生成自动文件名 - 格式: 模型架构_日期时间_参数信息.pt
        current_time = time.strftime("%Y%m%d_%H%M%S")
        
        # 如果已经训练过模型，使用训练时的参数
        if hasattr(self, 'training_thread') and hasattr(self.training_thread, 'training_params'):
            model_arch = self.training_thread.training_params.get('model_architecture', 'Unknown')
            batch_size = self.training_thread.training_params.get('batch_size', 0)
            learning_rate = self.training_thread.training_params.get('learning_rate', 0)
            use_gpu = self.training_thread.training_params.get('use_gpu', False)
        else:
            # 如果没有训练过，使用当前界面的设置
            model_arch = self.model_architecture_combo.currentText()
            batch_size = self.batch_size_spin.value()
            learning_rate = float(self.learning_rate_combo.currentText())
            use_gpu = self.use_gpu_checkbox.isChecked() and self.use_gpu_checkbox.isEnabled()
        
        # 文件名形式：架构_时间_批次大小_学习率_设备.pt
        device_info = "gpu" if use_gpu else "cpu"
        auto_filename = f"{model_arch}_{current_time}_b{batch_size}_lr{learning_rate:.4f}_{device_info}.pt"
        
        # 如果设置了模型输出路径，使用该路径作为默认路径
        default_path = ""
        if hasattr(self, 'model_output_path') and self.model_output_path.text().strip():
            default_path = os.path.join(self.model_output_path.text().strip(), auto_filename)
        else:
            default_path = auto_filename
        
        # 打开文件对话框，但已预填充自动生成的文件名
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出模型", default_path, "PyTorch模型 (*.pt);;所有文件 (*)"
        )
        
        if save_path:
            if not save_path.endswith('.pt'):
                save_path += '.pt'
            
            # 在实际项目中，应将模型保存到指定路径
            try:
                # 模拟模型保存
                with open(save_path, 'w') as f:
                    f.write("# 模型占位符")
                    # 在实际项目中，这里应该是torch.save(model.state_dict(), save_path)
                
                self.add_training_log(f"模型已导出至: {save_path}")
                QtWidgets.QMessageBox.information(self, "导出成功", f"模型已导出至: {save_path}")
            except Exception as e:
                error_msg = f"模型导出失败: {str(e)}"
                self.add_training_log(error_msg)
                QtWidgets.QMessageBox.critical(self, "导出失败", error_msg)
    # 模型训练部分的方法结束

    def toggle_gpu_options(self, state):
        """切换GPU选项的启用状态"""
        if not TORCH_AVAILABLE:
            return
            
        self.gpu_device_combo.setEnabled(state)
        self.use_mixed_precision.setEnabled(state)
        # 注意：不要禁用use_gpu_checkbox本身，否则无法重新启用

    def browse_dataset_root(self):
        """浏览数据集根目录并自动检测子文件夹"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择数据集根目录", os.path.expanduser("~")
        )
        if folder_path:
            self.dataset_root_path.setText(folder_path)
            
            # 自动检测数据集子文件夹
            self.detect_dataset_subfolders(folder_path)
    
    def browse_model_output(self):
        """浏览模型保存路径"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择模型保存路径", os.path.expanduser("~")
        )
        if folder_path:
            self.model_output_path.setText(folder_path)
    
    def detect_dataset_subfolders(self, root_path):
        """检测数据集根目录中的train/val/test子文件夹"""
        # 定义要检测的子文件夹名称
        subfolders = {
            "train": {"label": self.train_folder_label, "detected": False, "path": ""},
            "val": {"label": self.val_folder_label, "detected": False, "path": ""},
            "test": {"label": self.test_folder_label, "detected": False, "path": ""}
        }
        
        # 设置状态标签样式
        status_style_detected = "color: green; font-weight: bold;"
        status_style_missing = "color: red;"
        
        # 检查子文件夹是否存在
        for subfolder_name, info in subfolders.items():
            subfolder_path = os.path.join(root_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                info["detected"] = True
                info["path"] = subfolder_path
                info["label"].setText(f"{subfolder_name.capitalize()}: 已检测 ({os.path.basename(root_path)}/{subfolder_name})")
                info["label"].setStyleSheet(status_style_detected)
            else:
                info["label"].setText(f"{subfolder_name.capitalize()}: 未检测")
                info["label"].setStyleSheet(status_style_missing)
        
        # 将检测到的路径存储为属性，供训练时使用
        self.train_data_path = subfolders["train"]["path"]
        self.val_data_path = subfolders["val"]["path"]
        self.test_data_path = subfolders["test"]["path"]
        
        # 更新训练界面状态
        if subfolders["train"]["detected"] and subfolders["val"]["detected"]:
            self.add_training_log("数据集子文件夹检测成功")
            # 允许启动训练
            self.start_training_btn.setEnabled(True)
        else:
            if not subfolders["train"]["detected"]:
                self.add_training_log("警告: 未检测到训练集文件夹 (train)")
            if not subfolders["val"]["detected"]:
                self.add_training_log("警告: 未检测到验证集文件夹 (val)")
            # 如果缺少必要的子文件夹，禁用训练按钮
            self.start_training_btn.setEnabled(False)

    # 在MainWindow类中添加init_training_signals方法定义
    def init_training_signals(self):
        """
        初始化训练信号连接
        """
        # 目前没有训练线程，初始化为None
        self.training_thread = None
        
        # 简单初始化，不调用不存在的方法
        print("训练信号初始化完成")

    def update_training_status(self, status):
        """更新训练状态标签"""
        self.training_status_label.setText(status)
        
    def update_plot_from_pixmap(self, pixmap):
        """从QPixmap更新训练图表"""
        try:
            # 清除当前布局中的所有小部件
            for i in reversed(range(self.training_plot_layout.count())): 
                widget = self.training_plot_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            
            # 创建一个QLabel来显示pixmap
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            
            # 将标签添加到布局中
            self.training_plot_layout.addWidget(label)
        except Exception as e:
            self.add_training_log(f"更新图表失败: {str(e)}")

    def toggle_bpm_mode(self, checked):
        """切换BPM设置模式"""
        # 只有当切换到选中状态时才执行
        if not checked:
            return
            
        # 根据选中的按钮更新UI状态
        self.bpm_input.setEnabled(self.manual_bpm_rb.isChecked())
        self.import_bpm_btn.setEnabled(self.manual_bpm_rb.isChecked())
        
    def import_bpm_from_beatmap(self):
        """从谱面文件导入BPM"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择OSU谱面文件",
            "",
            "OSU文件 (*.osu)"
        )
        
        if file_path:
            success = self.audio_analyzer.import_beatmap_bpm(file_path)
            if success:
                self.status_label.setText(f"已从谱面导入BPM: {self.audio_analyzer.bpm:.2f}")
                self.bpm_value.setText(f"{self.audio_analyzer.bpm:.2f}")
            else:
                self.status_label.setText("从谱面导入BPM失败")
    
    def browse_beatmap_gen_audio(self):
        """浏览谱面生成音频分析文件夹"""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择音频分析文件夹",
            ""
        )
        
        if folder_path:
            self.beatmap_gen_audio_path.setText(folder_path)
            # 尝试从文件夹名称自动填充元数据
            try:
                import os
                folder_name = os.path.basename(folder_path)
                
                # 如果文件夹名称包含分析结果标识，说明可能是导出的音频分析结果
                if "_analysis" in folder_name:
                    # 从文件夹名称中提取基本信息
                    base_name = folder_name.replace("_analysis", "")
                    self.beatmap_title.setText(base_name)
            except:
                # 如果自动填充失败，不做处理
                pass
                
            # 自动刷新音频轨道列表
            self.refresh_audio_sources()
    
    def browse_beatmap_gen_output(self):
        """浏览谱面生成输出目录"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择谱面输出目录",
            ""
        )
        
        if dir_path:
            self.beatmap_gen_output_path.setText(dir_path)
    
    def generate_beatmap_from_tab(self):
        """从谱面生成选项卡生成谱面"""
        import os  # 确保在函数内部可以访问os模块
        
        # 检查音频分析文件夹
        analysis_folder = self.beatmap_gen_audio_path.text()
        if not analysis_folder or not os.path.isdir(analysis_folder):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的音频分析文件夹")
            return
            
        # 检查分析文件夹中是否有分析文件
        analysis_files = [f for f in os.listdir(analysis_folder) if f.endswith('.json')]
        if not analysis_files:
            QtWidgets.QMessageBox.warning(self, "警告", "所选文件夹中没有找到分析文件（.json）")
            return
            
        # 检查是否选择了音频轨道
        selected_sources = []
        for i in range(self.source_table.rowCount()):
            # 获取复选框状态
            checkbox_widget = self.source_table.cellWidget(i, 0)
            checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
            
            if checkbox.isChecked():
                # 获取轨道ID
                track_item = self.source_table.item(i, 1)
                source_id = track_item.data(QtCore.Qt.UserRole)
                
                # 获取优先级
                priority_widget = self.source_table.cellWidget(i, 2)
                priority_spin = priority_widget.findChild(QtWidgets.QSpinBox)
                priority = priority_spin.value()
                
                selected_sources.append({
                    "id": source_id,
                    "name": track_item.text(),
                    "priority": priority
                })
        
        if not selected_sources:
            QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个音频轨道")
            return
            
        # 获取最大轨道数量
        max_sources = self.max_sources_spin.value()
        
        # 按优先级排序
        selected_sources.sort(key=lambda x: x["priority"], reverse=True)
        
        # 如果超过最大数量，截取优先级最高的几个
        if len(selected_sources) > max_sources:
            selected_sources = selected_sources[:max_sources]
            QtWidgets.QMessageBox.information(
                self, 
                "轨道数量已调整", 
                f"已选择的音频轨道数量超过了最大值，将只使用优先级最高的 {max_sources} 个轨道。"
            )
            
        # 检查输出目录
        output_dir = self.beatmap_gen_output_path.text()
        if not output_dir:
            output_dir = os.path.dirname(analysis_folder)
            self.beatmap_gen_output_path.setText(output_dir)
            
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                QtWidgets.QMessageBox.critical(self, "错误", "无法创建输出目录")
                return
            
        # 获取元数据
        title = self.beatmap_title.text() or "未命名"
        artist = self.beatmap_artist.text() or "未知艺术家"
        creator = self.beatmap_creator.text() or "AI谱面生成器"
        difficulty = self.difficulty_selector.currentText()
        
        # 获取难度参数
        ar = self.ar_slider.value()
        od = self.od_slider.value()
        hp = self.hp_slider.value()
        cs = self.cs_slider.value()
        
        # 获取生成参数
        use_model = self.use_model_cb.isChecked()
        density = self.density_slider.value()
        
        # 设置状态
        self.beatmap_gen_status.setText("正在加载分析数据...")
        self.beatmap_gen_progress.setValue(10)
        
        # 导入BeatmapGenerator类
        try:
            # 尝试直接导入
            import importlib.util
            import sys
            import os
            import json
            import traceback
            from pumianzhizuo.beatmap.beatmap_generator import BeatmapGenerator
        except ImportError:
            # 如果失败，尝试添加项目根目录到Python路径
            try:
                import os
                import sys
                
                # 获取项目根目录路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                
                # 添加到Python路径
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # 再次尝试导入
                from pumianzhizuo.beatmap.beatmap_generator import BeatmapGenerator
            except ImportError as e:
                # 导入失败，尝试直接从相对路径导入
                try:
                    import importlib.util
                    import os
                    
                    # 构建beatmap_generator.py的绝对路径
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    module_path = os.path.join(os.path.dirname(current_dir), 'beatmap', 'beatmap_generator.py')
                    
                    # 动态导入模块
                    spec = importlib.util.spec_from_file_location("beatmap_generator", module_path)
                    beatmap_generator_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(beatmap_generator_module)
                    
                    # 从模块中获取BeatmapGenerator类
                    BeatmapGenerator = beatmap_generator_module.BeatmapGenerator
                except Exception as inner_e:
                    # 所有导入方法均失败
                    QtWidgets.QMessageBox.critical(
                        self, 
                        "错误", 
                        f"无法导入谱面生成器，请检查安装\n\n详细错误:\n{str(e)}\n{str(inner_e)}"
                    )
                    return
        
        # 加载分析数据
        try:
            # 创建谱面生成器实例
            beatmap_generator = BeatmapGenerator()
            
            # 设置谱面元数据
            beatmap_generator.set_metadata(title, artist, creator, difficulty)
            
            # 设置难度参数
            beatmap_generator.set_difficulty(ar, od, hp, cs)
            
            # 设置生成参数
            model_path = None
            if use_model:
                # TODO: 实现模型选择逻辑
                model_path = None
                
            beatmap_generator.set_generation_params(density, use_model, model_path)
            
            # 设置事件选择概率
            beat_prob = self.beat_prob_slider.value() / 100.0
            onset_prob = self.onset_prob_slider.value() / 100.0
            beatmap_generator.set_event_selection_probabilities(beat_prob, onset_prob)
            
            # 设置强拍弱拍阈值 - 使用数值输入框的值
            if hasattr(self, 'strong_beat_input') and hasattr(self, 'medium_beat_input'):
                strong_threshold = self.strong_beat_input.value()
                medium_threshold = self.medium_beat_input.value()
            # 向后兼容 - 使用滑动条的值
            else:
                strong_threshold = self.strong_beat_slider.value() / 100.0
                medium_threshold = self.medium_beat_slider.value() / 100.0
                
            beatmap_generator.set_beat_strength_thresholds(strong_threshold, medium_threshold)
            
            # 分析数据字典，用于存储每个轨道的分析数据
            analysis_data_map = {}
            
            # 设置状态
            self.beatmap_gen_status.setText("正在加载音频轨道分析数据...")
            self.beatmap_gen_progress.setValue(20)
            
            # 检查是否有索引文件
            index_files = [f for f in os.listdir(analysis_folder) if f.endswith('.json') and not any(
                suffix in f for suffix in ["_vocals", "_drums", "_bass", "_other"])]
            
            import json
            loaded_sources = 0
            total_sources = len(selected_sources)
            
            if index_files:
                # 从索引文件加载轨道信息
                try:
                    with open(os.path.join(analysis_folder, index_files[0]), 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    
                    # 获取轨道文件路径
                    if "source_files" in index_data:
                        sources = index_data["source_files"]
                        
                        # 按照优先级顺序载入分析数据
                        for source in selected_sources:
                            source_id = source["id"]
                            if source_id in sources:
                                source_info = sources[source_id]
                                file_path = os.path.join(os.path.dirname(analysis_folder), source_info["file_path"])
                                
                                # 载入分析数据
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        source_data = json.load(f)
                                    analysis_data_map[source_id] = {
                                        "data": source_data,
                                        "priority": source["priority"]
                                    }
                                    loaded_sources += 1
                                    
                                    # 更新进度
                                    progress = 20 + (loaded_sources / total_sources) * 20
                                    self.beatmap_gen_progress.setValue(int(progress))
                                    self.beatmap_gen_status.setText(f"已加载 {loaded_sources}/{total_sources} 个音频轨道")
                                    QtCore.QCoreApplication.processEvents()
                                    
                                except Exception as e:
                                    print(f"载入 {source_id} 分析数据失败: {str(e)}")
                except Exception as e:
                    print(f"从索引文件加载音频源信息失败: {str(e)}")
            
            # 如果未能通过索引文件加载所有需要的数据，从文件名推断
            if len(analysis_data_map) < len(selected_sources):
                for source in selected_sources:
                    source_id = source["id"]
                    if source_id not in analysis_data_map:
                        # 查找对应轨道的文件
                        for file in analysis_files:
                            if f"_{source_id}" in file or (source_id == "original" and not any(
                                    suffix in file for suffix in ["_vocals", "_drums", "_bass", "_other"])):
                                try:
                                    with open(os.path.join(analysis_folder, file), 'r', encoding='utf-8') as f:
                                        source_data = json.load(f)
                                    analysis_data_map[source_id] = {
                                        "data": source_data,
                                        "priority": source["priority"]
                                    }
                                    loaded_sources += 1
                                    
                                    # 更新进度
                                    progress = 20 + (loaded_sources / total_sources) * 20
                                    self.beatmap_gen_progress.setValue(int(progress))
                                    self.beatmap_gen_status.setText(f"已加载 {loaded_sources}/{total_sources} 个音频轨道")
                                    QtCore.QCoreApplication.processEvents()
                                    
                                    break
                                except Exception as e:
                                    print(f"载入 {source_id} 分析数据失败: {str(e)}")
            
            # 检查是否成功加载了所有需要的轨道数据
            if not analysis_data_map:
                raise Exception("未能加载任何音频轨道的分析数据")
            
            # 加载分析数据到谱面生成器
            beatmap_generator.load_analysis_data(analysis_data_map)
            
            # 显示已加载的轨道及其优先级
            loaded_sources_info = []
            for source in selected_sources:
                if source["id"] in analysis_data_map:
                    loaded_sources_info.append(f"{source['name']}(优先级:{source['priority']})")
            
            # 更新进度
            self.beatmap_gen_progress.setValue(40)
            self.beatmap_gen_status.setText("正在生成谱面...")
            QtCore.QCoreApplication.processEvents()
            
            # 生成谱面物件
            beatmap_generator.generate_beatmap()
            
            # 更新进度
            self.beatmap_gen_progress.setValue(70)
            self.beatmap_gen_status.setText("正在生成谱面文件...")
            QtCore.QCoreApplication.processEvents()
            
            # 构建输出文件路径
            difficulty_text = difficulty.replace(' ', '')
            sanitized_title = ''.join(c for c in title if c.isalnum() or c in ' -_[](){}')
            sanitized_artist = ''.join(c for c in artist if c.isalnum() or c in ' -_[](){}')
            output_filename = f"{sanitized_artist} - {sanitized_title} [{difficulty_text}].osu"
            output_path = os.path.join(output_dir, output_filename)
            
            # 生成osu文件
            beatmap_file_path = beatmap_generator.generate_osu_file(output_path)
            
            # 更新状态
            self.beatmap_gen_status.setText("谱面生成完成!")
            self.beatmap_gen_progress.setValue(100)
            self.preview_btn.setEnabled(True)
            
            # 显示成功消息和使用的轨道信息
            used_tracks_info = "\n".join([f"- {source}" for source in loaded_sources_info])
            QtWidgets.QMessageBox.information(
                self, 
                "生成完成", 
                f"谱面已生成: {output_filename}\n\n使用的音频轨道:\n{used_tracks_info}"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.beatmap_gen_status.setText("生成失败: " + str(e))
            self.beatmap_gen_progress.setValue(0)
            QtWidgets.QMessageBox.critical(self, "错误", f"谱面生成失败: {str(e)}")
            return
    
    def preview_beatmap_from_tab(self):
        """预览生成的谱面"""
        import os
        import glob
        import subprocess

        # 检查输出目录
        output_dir = self.beatmap_gen_output_path.text()
        if not output_dir or not os.path.exists(output_dir):
            QtWidgets.QMessageBox.warning(self, "错误", "谱面输出目录不存在，请先生成谱面")
            return
            
        # 构建标题和艺术家名以查找谱面文件
        title = self.beatmap_title.text() or "未命名"
        artist = self.beatmap_artist.text() or "未知艺术家"
        difficulty = self.difficulty_selector.currentText()
        
        # 查找生成的谱面文件
        difficulty_text = difficulty.replace(' ', '')
        sanitized_title = ''.join(c for c in title if c.isalnum() or c in ' -_[](){}')
        sanitized_artist = ''.join(c for c in artist if c.isalnum() or c in ' -_[](){}')
        filename_pattern = f"{sanitized_artist} - {sanitized_title} [{difficulty_text}].osu"
        
        beatmap_file = os.path.join(output_dir, filename_pattern)
        
        if not os.path.exists(beatmap_file):
            # 尝试使用通配符查找类似的文件
            import glob
            similar_files = glob.glob(os.path.join(output_dir, f"*{sanitized_title}*[{difficulty_text}]*.osu"))
            
            if similar_files:
                beatmap_file = similar_files[0]
            else:
                QtWidgets.QMessageBox.warning(self, "文件未找到", "找不到生成的谱面文件，请先生成谱面")
                return
        
        # 尝试找到osu安装目录
        osu_path = None
        
        # 尝试从Windows注册表查找osu安装路径
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"osu\shell\open\command")
            command = winreg.QueryValue(key, "")
            # 从注册表命令中提取路径
            # 通常格式: "C:\Path\to\osu.exe" "%1"
            if command and '"' in command:
                osu_path = command.split('"')[1]
        except:
            pass
        
        # 如果注册表查找失败，尝试常见的安装路径
        if not osu_path or not os.path.exists(osu_path):
            common_paths = [
                os.path.join(os.environ.get('LOCALAPPDATA', ''), "osu!", "osu!.exe"),
                os.path.join(os.environ.get('PROGRAMFILES', ''), "osu!", "osu!.exe"),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), "osu!", "osu!.exe")
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    osu_path = path
                    break
        
        if not osu_path or not os.path.exists(osu_path):
            # 如果自动查找失败，让用户选择osu可执行文件
            osu_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择osu!.exe", "", "osu! (osu!.exe)"
            )
            
            if not osu_path:
                QtWidgets.QMessageBox.warning(self, "操作取消", "用户取消了选择osu!.exe")
                return
        
        # 启动osu并打开谱面
        try:
            import subprocess
            subprocess.Popen([osu_path, beatmap_file])
            self.beatmap_gen_status.setText(f"已启动osu!预览谱面: {os.path.basename(beatmap_file)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "启动失败", f"无法启动osu!预览谱面: {str(e)}")

    def showEvent(self, event):
        """窗口显示时的处理"""
        super().showEvent(event)
        
        # 填充模型选择下拉菜单
        if hasattr(self, 'model_combo') and hasattr(self, 'audio_analyzer'):
            # 清空现有选项
            self.model_combo.clear()
            
            # 获取可用模型
            try:
                available_models = self.audio_analyzer.get_available_models()
                for model_id, model_name in available_models.items():
                    self.model_combo.addItem(model_name, model_id)
                
                # 默认选择htdemucs模型
                default_index = self.model_combo.findData("htdemucs")
                if default_index >= 0:
                    self.model_combo.setCurrentIndex(default_index)
            except:
                # 如果还没初始化完成，添加默认选项
                self.model_combo.addItem("Demucs HT（混合变体）", "htdemucs")

    def reset_source_priority(self):
        """重置音频源优先级到默认顺序"""
        # 清空现有列表
        self.priority_list.clear()
        
        # 重新添加默认顺序的音频源
        source_display_names = {
            "vocals": "人声 (Vocals)",
            "drums": "鼓声 (Drums)",
            "bass": "贝斯 (Bass)",
            "other": "其他乐器 (Other)"
        }
        
        for source_id in self.audio_analyzer.DEFAULT_PRIORITY:
            display_name = source_display_names.get(source_id, source_id)
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, source_id)
            # 获取图标路径
            icon_path = self.get_source_icon_path(source_id)
            if icon_path:
                item.setIcon(QtGui.QIcon(icon_path))
            self.priority_list.addItem(item)
    
    def reset_noise_reduction_params(self):
        """重置降噪参数到默认值"""
        # 重置阈值滑块
        self.noise_threshold_slider.setValue(5)  # 默认值0.05
        self.noise_threshold_value.setText("0.05")
        
        # 重置强度滑块
        self.noise_strength_slider.setValue(75)  # 默认值0.75
        self.noise_strength_value.setText("0.75")
    
    def get_source_icon_path(self, source_id):
        """获取音频源对应的图标路径"""
        icons = {
            "vocals": "gui/resources/vocal_icon.png",
            "drums": "gui/resources/drum_icon.png",
            "bass": "gui/resources/bass_icon.png",
            "other": "gui/resources/other_icon.png"
        }
        return icons.get(source_id)

    def on_priority_changed(self):
        """当优先级顺序改变时触发"""
        # 记录当前优先级顺序
        current_priority = []
        for i in range(self.priority_list.count()):
            item = self.priority_list.item(i)
            source_id = item.data(QtCore.Qt.UserRole)
            current_priority.append(source_id)
        
        # 如果分析器已存在，立即更新优先级
        if hasattr(self, 'audio_analyzer') and self.audio_analyzer:
            self.audio_analyzer.set_source_priority(current_priority)
            
            # 如果已经完成分离，重新选择活跃源
            if hasattr(self.audio_analyzer, 'separated_sources') and self.audio_analyzer.separated_sources:
                self.audio_analyzer._select_active_source()
                
                # 如果可视化器存在，更新显示
                if hasattr(self, 'audio_visualizer') and self.audio_visualizer:
                    self.audio_visualizer.set_active_source(self.audio_analyzer.active_source)
                    self.audio_visualizer.update_visualization()
        
        # 更新状态栏以提示用户
        self.status_bar.showMessage(f"已更新音频源优先级: {' > '.join(current_priority)}", 3000)

    def test_audio_sources(self):
        """测试分离的音频源标签与内容是否匹配"""
        # 检查是否已分离音频
        if not hasattr(self, 'audio_analyzer') or not hasattr(self.audio_analyzer, 'separated_sources') or not self.audio_analyzer.separated_sources:
            QtWidgets.QMessageBox.warning(self, "未分离音频", "请先启用人声分离并分析音频，然后再测试音频源")
            return
        
        # 设置播放工具
        try:
            import sounddevice as sd
            import time
        except ImportError:
            QtWidgets.QMessageBox.warning(self, "缺少依赖", "需要安装sounddevice库才能测试音频。请运行 'pip install sounddevice'")
            return
        
        # 获取分离的音频源
        sources = self.audio_analyzer.separated_sources
        sr = self.audio_analyzer.sr
        play_duration = 3  # 秒
        
        # 音频源显示名称
        source_display_names = {
            "vocals": "人声(vocals)",
            "drums": "鼓声(drums)",
            "bass": "贝斯(bass)",
            "other": "其他乐器(other)"
        }
        
        # 创建播放队列
        play_queue = []
        for source_name, audio_data in sources.items():
            display_name = source_display_names.get(source_name, source_name)
            play_queue.append((source_name, display_name, audio_data))
        
        # 创建进度对话框
        progress = QtWidgets.QProgressDialog("测试音频源中...", "取消", 0, len(play_queue), self)
        progress.setWindowTitle("音频源测试")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        # 播放每个源的一部分
        for i, (source_name, display_name, audio_data) in enumerate(play_queue):
            # 更新进度对话框
            progress.setValue(i)
            progress.setLabelText(f"正在播放: {display_name}\n请聆听此音频是否与标签相符")
            
            # 处理取消
            if progress.wasCanceled():
                break
            
            # 计算要播放的样本数
            samples = min(int(play_duration * sr), len(audio_data))
            
            # 找到最大音量段落
            if len(audio_data) > sr * 10:  # 如果音频长度超过10秒
                # 将音频分成10段，找到能量最大的段落
                segment_length = len(audio_data) // 10
                max_energy = 0
                start_idx = 0
                
                for j in range(10):
                    segment = audio_data[j * segment_length:(j+1) * segment_length]
                    energy = np.sum(segment**2)
                    if energy > max_energy:
                        max_energy = energy
                        start_idx = j * segment_length
                
                # 确保不会越界
                if start_idx + samples > len(audio_data):
                    start_idx = len(audio_data) - samples
                
                # 提取最响亮的片段
                audio_segment = audio_data[start_idx:start_idx + samples]
            else:
                # 音频较短，直接从头播放
                audio_segment = audio_data[:samples]
            
            # 检查音频是否是静音
            if np.max(np.abs(audio_segment)) < 0.01:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "音频太安静", 
                    f"警告: {display_name} 源的音频音量极小，可能是静音或分离有问题"
                )
                continue
            
            # 标准化音频音量
            audio_segment = audio_segment / np.max(np.abs(audio_segment)) * 0.8
            
            # 播放音频
            sd.play(audio_segment, sr)
            
            # 等待播放完成
            time.sleep(play_duration)
            sd.stop()
            
            # 短暂停顿，分隔不同源的播放
            time.sleep(0.5)
        
        # 完成播放
        progress.setValue(len(play_queue))
        QtWidgets.QMessageBox.information(
            self, 
            "测试完成", 
            "所有音频源已播放完毕。如果发现标签与内容不匹配，请在GitHub报告此问题。"
        )

    def refresh_audio_sources(self):
        """扫描分析文件夹并更新音频轨道表格"""
        analysis_folder = self.beatmap_gen_audio_path.text()
        if not analysis_folder or not os.path.isdir(analysis_folder):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的音频分析文件夹")
            return
        
        # 清空当前表格
        self.source_table.setRowCount(0)
        
        # 获取音频轨道列表
        source_list = []
        
        # 检查是否有索引文件
        index_files = [f for f in os.listdir(analysis_folder) if f.endswith('.json') and not any(
            suffix in f for suffix in ["_vocals", "_drums", "_bass", "_other"])]
        
        if index_files:
            # 尝试从索引文件加载
            try:
                import json
                with open(os.path.join(analysis_folder, index_files[0]), 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # 检查是否包含音频源信息
                if "source_files" in index_data:
                    # 添加音频源到列表
                    sources = index_data["source_files"]
                    for source_id, source_info in sources.items():
                        display_name = source_info.get("display_name", source_id)
                        source_list.append({
                            "id": source_id,
                            "name": display_name
                        })
            except Exception as e:
                print(f"从索引文件加载音频源信息失败: {str(e)}")
        
        # 如果没有索引文件或加载失败，从文件名推断
        if not source_list:
            analysis_files = [f for f in os.listdir(analysis_folder) if f.endswith('.json')]
            
            # 定义音频源类型映射
            source_types = {
                "vocals": "人声(vocals)",
                "drums": "鼓声(drums)",
                "bass": "贝斯(bass)",
                "other": "其他乐器(other)",
                "original": "原始音频(original)"
            }
            
            # 添加默认的"原始音频"选项
            has_original = False
            
            for file in analysis_files:
                source_id = None
                # 检查文件名以识别音频源类型
                for source_type in source_types:
                    if f"_{source_type}" in file:
                        source_id = source_type
                        break
                
                if source_id == "original":
                    has_original = True
                    
                if source_id:
                    display_name = source_types.get(source_id, source_id)
                    source_list.append({
                        "id": source_id,
                        "name": display_name
                    })
            
            # 如果没有找到原始音频，但有至少一个分析文件，添加原始音频选项
            if not has_original and analysis_files:
                source_list.append({
                    "id": "original",
                    "name": source_types["original"]
                })
        
        # 填充表格
        self.source_table.setRowCount(len(source_list))
        
        # 默认优先级，从高到低
        default_priorities = [5, 4, 3, 2, 1]
        
        for i, source in enumerate(source_list):
            # 创建复选框
            checkbox = QtWidgets.QCheckBox()
            checkbox_widget = QtWidgets.QWidget()
            checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(QtCore.Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_widget.setStyleSheet("background-color: transparent;")
            
            # 创建音频轨道显示项
            track_item = QtWidgets.QTableWidgetItem(source["name"])
            track_item.setData(QtCore.Qt.UserRole, source["id"])
            # 设置图标（如果有）
            icon_path = self.get_source_icon_path(source["id"])
            if icon_path and os.path.exists(icon_path):
                track_item.setIcon(QtGui.QIcon(icon_path))
            
            # 创建优先级微调框
            priority_spin = QtWidgets.QSpinBox()
            priority_spin.setRange(1, 10)
            priority_spin.setFixedWidth(50)  # 设置固定宽度
            priority_spin.setAlignment(QtCore.Qt.AlignCenter)  # 设置数字居中
            priority_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)  # 使用上下箭头
            priority_spin.setStyleSheet("""
                QSpinBox {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    padding: 2px;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    width: 16px;
                }
            """)
            
            # 设置默认优先级
            priority = default_priorities[i] if i < len(default_priorities) else 1
            priority_spin.setValue(priority)
            
            priority_widget = QtWidgets.QWidget()
            priority_layout = QtWidgets.QHBoxLayout(priority_widget)
            priority_layout.addWidget(priority_spin)
            priority_layout.setAlignment(QtCore.Qt.AlignCenter)
            priority_layout.setContentsMargins(0, 0, 0, 0)
            priority_widget.setStyleSheet("background-color: transparent;")
            
            # 添加到表格，并设置行高
            self.source_table.setCellWidget(i, 0, checkbox_widget)
            self.source_table.setItem(i, 1, track_item)
            self.source_table.setCellWidget(i, 2, priority_widget)
            self.source_table.setRowHeight(i, 30)  # 设置统一的行高
            
            # 默认选择前N个
            if i < self.max_sources_spin.value():
                checkbox.setChecked(True)
        
        # 调整表格行高
        self.source_table.resizeRowsToContents()

    def ensure_threshold_consistency(self):
        """确保强拍阈值始终大于次强拍阈值"""
        # 适配新的数值输入框
        if hasattr(self, 'strong_beat_input') and hasattr(self, 'medium_beat_input'):
            strong_value = self.strong_beat_input.value()
            medium_value = self.medium_beat_input.value()
            
            # 如果次强拍阈值大于等于强拍阈值，自动调整
            if medium_value >= strong_value:
                # 将次强拍阈值设为强拍阈值的60%
                new_medium_value = strong_value * 0.6
                self.medium_beat_input.blockSignals(True)  # 阻止信号触发递归
                self.medium_beat_input.setValue(new_medium_value)
                self.medium_beat_input.blockSignals(False)
        # 保留对旧版滑动条的支持（为向后兼容）
        elif hasattr(self, 'strong_beat_slider') and hasattr(self, 'medium_beat_slider'):
            strong_value = self.strong_beat_slider.value()
            medium_value = self.medium_beat_slider.value()
            
            # 如果次强拍阈值大于等于强拍阈值，自动调整
            if medium_value >= strong_value:
                # 将次强拍阈值设为强拍阈值的60%
                new_medium_value = int(strong_value * 0.6)
                self.medium_beat_slider.setValue(new_medium_value)

    def browse_input_subtitle(self):
        """浏览选择输入字幕文件"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择输入字幕文件", "", "字幕文件 (*.srt)"
        )
        if file_path:
            self.input_subtitle_path.setText(file_path)
            # 自动设置输出文件路径
            if not self.output_subtitle_path.text():
                input_path = file_path
                file_name = os.path.basename(input_path)
                file_dir = os.path.dirname(input_path)
                base_name, ext = os.path.splitext(file_name)
                output_path = os.path.join(file_dir, f"{base_name}_中文{ext}")
                self.output_subtitle_path.setText(output_path)

    def browse_output_subtitle(self):
        """浏览选择输出字幕文件"""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择输出字幕文件", "", "字幕文件 (*.srt)"
        )
        if file_path:
            self.output_subtitle_path.setText(file_path)

    def process_subtitle(self):
        """处理字幕"""
        # 获取输入和输出文件路径
        input_path = self.input_subtitle_path.text()
        output_path = self.output_subtitle_path.text()
        
        # 检查输入文件
        if not input_path or not os.path.exists(input_path):
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的输入字幕文件")
            return
        
        # 检查输出文件路径
        if not output_path:
            QtWidgets.QMessageBox.warning(self, "错误", "请指定输出字幕文件路径")
            return
        
        # 获取处理选项
        extract_chinese = self.extract_chinese_cb.isChecked()
        merge_nearby = self.merge_nearby_cb.isChecked()
        merge_threshold = self.merge_threshold_spin.value()
        
        # 设置进度条初始状态
        self.subtitle_progress_bar.setValue(10)
        self.subtitle_status_label.setText("正在处理字幕...")
        
        try:
            # 导入必要的模块
            import re
            
            # 定义函数来提取中文文本
            def extract_chinese_text(line):
                # 匹配<b>标签中的中文文本
                pattern = r'<b>\s*​?\s*​?(.*?)​?\s*​?\s*</b>'
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
                return None
            
            # 定义解析时间的函数
            def parse_time(time_str):
                # 解析时间格式为秒数，便于比较
                hours, minutes, rest = time_str.split(':', 2)
                seconds, milliseconds = rest.split(',')
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
            
            # 定义时间格式化函数
            def format_time(seconds):
                # 将秒数转换回SRT时间格式
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds_part = seconds % 60
                seconds_int = int(seconds_part)
                milliseconds = int((seconds_part - seconds_int) * 1000)
                return f"{hours:01d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"
            
            # 打开并读取字幕文件
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 更新进度条
            self.subtitle_progress_bar.setValue(30)
            self.subtitle_status_label.setText("分析字幕内容...")
            
            subtitle_dict = {}  # 用于存储唯一文本的字幕
            current_index = None
            current_timestamp = None
            current_start_time = None
            current_end_time = None
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # 字幕索引
                if line.isdigit():
                    current_index = int(line)
                    i += 1
                    continue
                
                # 时间戳
                if '-->' in line:
                    current_timestamp = line
                    time_parts = line.split(' --> ')
                    current_start_time = parse_time(time_parts[0])
                    current_end_time = parse_time(time_parts[1])
                    i += 1
                    continue
                
                # 字幕文本
                if extract_chinese and '<font' in line and current_start_time is not None:
                    chinese_text = extract_chinese_text(line)
                    if chinese_text:
                        # 使用文本作为键，确保相同文本只处理一次
                        if chinese_text not in subtitle_dict:
                            subtitle_dict[chinese_text] = {
                                'start_times': [current_start_time],
                                'end_times': [current_end_time]
                            }
                        elif merge_nearby:
                            # 检查是否与上一个时间段连续或重叠
                            last_end = subtitle_dict[chinese_text]['end_times'][-1]
                            if current_start_time <= last_end or abs(current_start_time - last_end) < merge_threshold:
                                # 更新结束时间为较大的值
                                subtitle_dict[chinese_text]['end_times'][-1] = max(current_end_time, last_end)
                            else:
                                # 添加新的时间段
                                subtitle_dict[chinese_text]['start_times'].append(current_start_time)
                                subtitle_dict[chinese_text]['end_times'].append(current_end_time)
                # 不提取中文，直接处理
                elif not extract_chinese and line and line != "<font face=\"Sans Serif\" size=\"18\">" and line != "</font>" and current_start_time is not None:
                    if line not in subtitle_dict:
                        subtitle_dict[line] = {
                            'start_times': [current_start_time],
                            'end_times': [current_end_time]
                        }
                    elif merge_nearby:
                        # 检查是否与上一个时间段连续或重叠
                        last_end = subtitle_dict[line]['end_times'][-1]
                        if current_start_time <= last_end or abs(current_start_time - last_end) < merge_threshold:
                            # 更新结束时间为较大的值
                            subtitle_dict[line]['end_times'][-1] = max(current_end_time, last_end)
                        else:
                            # 添加新的时间段
                            subtitle_dict[line]['start_times'].append(current_start_time)
                            subtitle_dict[line]['end_times'].append(current_end_time)
                
                i += 1
            
            # 更新进度条
            self.subtitle_progress_bar.setValue(60)
            self.subtitle_status_label.setText("生成处理后的字幕...")
            
            # 将处理后的字幕写入输出文件
            preview_text = ""
            with open(output_path, 'w', encoding='utf-8') as f:
                index = 1
                for text, times in subtitle_dict.items():
                    for i in range(len(times['start_times'])):
                        start_formatted = format_time(times['start_times'][i])
                        end_formatted = format_time(times['end_times'][i])
                        
                        f.write(f"{index}\n")
                        f.write(f"{start_formatted} --> {end_formatted}\n")
                        f.write(f"{text}\n\n")
                        
                        # 为预览添加内容（最多显示前10个字幕）
                        if index <= 10:
                            preview_text += f"{index}\n{start_formatted} --> {end_formatted}\n{text}\n\n"
                        
                        index += 1
            
            # 更新进度条和状态
            self.subtitle_progress_bar.setValue(100)
            self.subtitle_status_label.setText(f"处理完成，共生成了{index-1}个字幕")
            
            # 更新预览文本
            if preview_text:
                self.subtitle_preview.setText(preview_text)
                if index > 10:
                    self.subtitle_preview.append("...(更多字幕已省略)...")
            else:
                self.subtitle_preview.setText("(没有找到可处理的字幕)")
            
            # 启用预览按钮
            self.preview_subtitle_btn.setEnabled(True)
            
            # 显示成功消息
            QtWidgets.QMessageBox.information(
                self, 
                "处理完成", 
                f"字幕处理完成！\n输出文件：{output_path}\n共处理了{index-1}个字幕"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.subtitle_status_label.setText(f"处理失败: {str(e)}")
            self.subtitle_progress_bar.setValue(0)
            QtWidgets.QMessageBox.critical(self, "错误", f"字幕处理失败: {str(e)}")

    def preview_subtitle(self):
        """预览处理后的字幕文件"""
        input_file = self.input_subtitle_path.text()
        output_file = self.output_subtitle_path.text()
        
        if not output_file or not os.path.exists(output_file):
            QtWidgets.QMessageBox.warning(self, "错误", "输出字幕文件不存在，请先处理字幕")
            return
            
        # 使用系统默认程序打开字幕文件
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == 'Windows':
                os.startfile(output_file)
            elif system == 'Darwin':  # macOS
                subprocess.call(('open', output_file))
            else:  # Linux
                subprocess.call(('xdg-open', output_file))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "预览失败", f"无法打开字幕文件: {str(e)}")

    def browse_osu_file(self):
        """浏览并选择osu谱面文件"""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择谱面文件", "", "osu谱面文件 (*.osu)"
        )
        
        if file_path:
            self.osu_file_path.setText(file_path)
            
            # 启用相关按钮
            self.video_preview_btn.setEnabled(True)
            self.generate_video_btn.setEnabled(True)
            
            # 更新状态
            self.video_gen_status.setText("已选择谱面文件，可以生成视频")

    def browse_video_output(self):
        """浏览并选择视频输出目录"""
        file_dialog = QtWidgets.QFileDialog()
        directory = file_dialog.getExistingDirectory(
            self, "选择输出目录", ""
        )
        
        if directory:
            self.video_output_path.setText(directory)

    def preview_video(self):
        """预览谱面文件内容"""
        osu_path = self.osu_file_path.text()
        
        if not osu_path or not os.path.exists(osu_path):
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的osu谱面文件")
            return
            
        try:
            # 解析谱面文件
            parser = OsuParser(osu_path)
            hits = parser.get_hits_for_1k()
            
            # 获取基本信息
            info_text = f"谱面信息:\n"
            info_text += f"音频文件: {parser.audio_filename}\n"
            info_text += f"总音符数: {len(hits)}\n"
            
            # 分析音符类型
            tap_count = sum(1 for h in hits if h["type"] == "tap")
            hold_count = sum(1 for h in hits if h["type"] == "hold")
            info_text += f"Tap音符: {tap_count}\n"
            info_text += f"Hold音符: {hold_count}\n"
            
            # 计算持续时间
            if hits:
                first_note_time = min(h["time"] for h in hits)
                last_note_time = max(h["end_time"] if "end_time" in h else h["time"] for h in hits)
                duration_ms = last_note_time - first_note_time
                duration_sec = duration_ms / 1000
                info_text += f"谱面长度: {int(duration_sec//60)}:{int(duration_sec%60):02d}\n"
            
            # 显示预览信息
            self.video_preview_label.setText(info_text)
            self.video_preview_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.video_preview_label.setStyleSheet("""
                font-size: 14px;
                color: #333333;
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 10px;
                text-align: left;
            """)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "预览失败", f"无法解析谱面文件: {str(e)}")
            self.video_preview_label.setText(f"预览失败: {str(e)}")

    def generate_video(self):
        """生成音游视频"""
        osu_path = self.osu_file_path.text()
        output_dir = self.video_output_path.text()
        
        if not osu_path or not os.path.exists(osu_path):
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的osu谱面文件")
            return
            
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "错误", "请指定输出视频目录")
            return
            
        if not os.path.isdir(output_dir):
            QtWidgets.QMessageBox.warning(self, "错误", "指定的输出路径不是一个有效的目录")
            return
            
        # 构建输出文件路径
        osu_filename = os.path.basename(osu_path)
        osu_basename = os.path.splitext(osu_filename)[0]
        output_path = os.path.join(output_dir, f"{osu_basename}_1k.mp4")
        
        # 获取设置
        fps = self.fps_spinbox.value()
        scroll_speed = int(self.scroll_speed_slider.value())
        
        # 更新状态
        self.video_gen_status.setText("正在生成视频...")
        self.video_gen_progress.setValue(10)
        
        # 禁用按钮
        self.video_preview_btn.setEnabled(False)
        self.generate_video_btn.setEnabled(False)
        
        try:
            # 在单独的线程中创建视频
            import threading
            
            def video_generation_task():
                try:
                    # 调用视频生成函数
                    lane_width = self.lane_width_spinbox.value()
                    create_vsrg_video(osu_path, output_path, fps=fps, scroll_speed=scroll_speed, lane_width=lane_width)
                    
                    # 完成后在主线程中更新UI
                    QtCore.QMetaObject.invokeMethod(
                        self, 
                        "handle_video_generation_complete", 
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, output_path)
                    )
                except Exception as e:
                    # 出错时在主线程中显示错误
                    QtCore.QMetaObject.invokeMethod(
                        self, 
                        "handle_video_generation_error", 
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, str(e))
                    )
            
            # 启动线程
            thread = threading.Thread(target=video_generation_task)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.video_gen_status.setText(f"生成失败: {str(e)}")
            self.video_gen_progress.setValue(0)
            self.video_preview_btn.setEnabled(True)
            self.generate_video_btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "错误", f"视频生成失败: {str(e)}")

    @QtCore.pyqtSlot(str)
    def handle_video_generation_complete(self, output_path):
        """处理视频生成完成事件"""
        self.video_gen_status.setText("视频生成完成!")
        self.video_gen_progress.setValue(100)
        
        # 重新启用按钮
        self.video_preview_btn.setEnabled(True)
        self.generate_video_btn.setEnabled(True)
        
        # 弹出成功消息
        QtWidgets.QMessageBox.information(
            self, 
            "生成完成", 
            f"视频已生成: {output_path}"
        )
        
        # 询问是否打开视频
        reply = QtWidgets.QMessageBox.question(
            self,
            "打开视频",
            "是否立即打开生成的视频?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                import subprocess
                import platform
                
                system = platform.system()
                if system == 'Windows':
                    os.startfile(output_path)
                elif system == 'Darwin':  # macOS
                    subprocess.call(('open', output_path))
                else:  # Linux
                    subprocess.call(('xdg-open', output_path))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "打开失败", f"无法打开视频文件: {str(e)}")

    @QtCore.pyqtSlot(str)
    def handle_video_generation_error(self, error_message):
        """处理视频生成错误事件"""
        self.video_gen_status.setText(f"生成失败: {error_message}")
        self.video_gen_progress.setValue(0)
        
        # 重新启用按钮
        self.video_preview_btn.setEnabled(True)
        self.generate_video_btn.setEnabled(True)
        
        # 显示错误消息
        QtWidgets.QMessageBox.critical(self, "错误", f"视频生成失败: {error_message}")

def main():
    """程序入口函数"""
    app = QtWidgets.QApplication(sys.argv)
    window = OsuStyleMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
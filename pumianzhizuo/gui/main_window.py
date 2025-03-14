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


class TrainingThread(QtCore.QThread):
    """模型训练线程类"""
    # 定义信号
    progress_updated = QtCore.pyqtSignal(int)  # 进度更新信号
    epoch_completed = QtCore.pyqtSignal(int, float, float)  # 周期完成信号(epoch, train_loss, val_loss)
    training_finished = QtCore.pyqtSignal(bool, str)  # 训练完成信号(成功/失败, 消息)
    training_log = QtCore.pyqtSignal(str)  # 训练日志信号
    
    def __init__(self, training_params):
        """初始化训练线程"""
        super().__init__()
        self.training_params = training_params
        self.is_running = False
        self.is_paused = False
    
    def run(self):
        """运行训练过程"""
        self.is_running = True
        self.is_paused = False
        
        try:
            # 解析训练参数
            model_architecture = self.training_params.get('model_architecture', 'Transformer')
            training_data_path = self.training_params.get('training_data_path', '')
            validation_data_path = self.training_params.get('validation_data_path', '')
            model_output_path = self.training_params.get('model_output_path', '')
            batch_size = self.training_params.get('batch_size', 16)
            learning_rate = self.training_params.get('learning_rate', 0.001)
            epochs = self.training_params.get('epochs', 50)
            use_early_stopping = self.training_params.get('use_early_stopping', True)
            use_checkpoint = self.training_params.get('use_checkpoint', True)
            use_mixed_precision = self.training_params.get('use_mixed_precision', True)
            use_gpu = self.training_params.get('use_gpu', False)
            gpu_device = self.training_params.get('gpu_device', 0)
            
            # 设置训练设备
            if use_gpu and torch.cuda.is_available():
                if gpu_device < torch.cuda.device_count():
                    device = torch.device(f'cuda:{gpu_device}')
                    self.training_log.emit(f"使用GPU训练: {torch.cuda.get_device_name(gpu_device)}")
                else:
                    device = torch.device('cuda:0')
                    self.training_log.emit(f"指定的GPU设备不存在，使用默认GPU设备: {torch.cuda.get_device_name(0)}")
                
                # 输出GPU信息
                gpu_properties = torch.cuda.get_device_properties(device)
                self.training_log.emit(f"GPU内存: {gpu_properties.total_memory / 1024 / 1024 / 1024:.2f} GB")
                self.training_log.emit(f"CUDA版本: {torch.version.cuda}")
                
                # 检查是否支持混合精度训练
                if use_mixed_precision and not torch.cuda.is_bf16_supported() and not torch.cuda.is_fp16_supported():
                    self.training_log.emit("警告: 当前GPU不支持混合精度训练，已禁用此功能")
                    use_mixed_precision = False
            else:
                if use_gpu and not torch.cuda.is_available():
                    self.training_log.emit("警告: 未检测到可用的GPU，将使用CPU训练")
                device = torch.device('cpu')
                self.training_log.emit("使用CPU训练")
                
                # 在CPU上禁用混合精度训练
                if use_mixed_precision:
                    self.training_log.emit("警告: CPU不支持混合精度训练，已禁用此功能")
                    use_mixed_precision = False
            
            # 模拟训练过程
            self.training_log.emit(f"开始加载训练数据: {training_data_path}")
            time.sleep(1)  # 模拟数据加载
            self.progress_updated.emit(5)
            
            self.training_log.emit(f"创建{model_architecture}模型")
            time.sleep(1)  # 模拟模型创建
            self.progress_updated.emit(10)
            
            # 设置混合精度训练
            if use_mixed_precision and device.type == 'cuda':
                self.training_log.emit("启用混合精度训练")
                # 在实际项目中，这里应该使用torch.cuda.amp.GradScaler()
                scaler = "GradScaler实例"  # 仅作为示例
            
            # 模拟训练循环
            for epoch in range(1, epochs + 1):
                if not self.is_running:
                    self.training_log.emit("训练被用户终止")
                    break
                
                # 等待如果暂停
                while self.is_paused and self.is_running:
                    time.sleep(0.1)
                
                # 模拟训练过程
                self.training_log.emit(f"Epoch {epoch}/{epochs}:")
                
                # 模拟批次训练
                n_batches = 10  # 模拟10个批次
                for batch in range(1, n_batches + 1):
                    if not self.is_running:
                        break
                    
                    # 等待如果暂停
                    while self.is_paused and self.is_running:
                        time.sleep(0.1)
                    
                    # 模拟批次训练
                    time.sleep(0.1)
                    self.training_log.emit(f"  Batch {batch}/{n_batches}")
                
                # 计算模拟损失 - GPU训练时收敛通常更快
                if device.type == 'cuda':
                    # GPU训练模拟更快的收敛
                    train_loss = 1.8 * np.exp(-0.15 * epoch) + 0.3 * np.random.random()
                else:
                    # CPU训练模拟较慢的收敛
                    train_loss = 2.0 * np.exp(-0.1 * epoch) + 0.5 * np.random.random()
                
                # 计算模拟验证损失
                if use_early_stopping and validation_data_path:
                    if device.type == 'cuda':
                        val_loss = 1.9 * np.exp(-0.12 * epoch) + 0.5 * np.random.random()
                    else:
                        val_loss = 2.2 * np.exp(-0.08 * epoch) + 0.7 * np.random.random()
                    self.training_log.emit(f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                else:
                    val_loss = None
                    self.training_log.emit(f"  训练损失: {train_loss:.4f}")
                
                # 发出周期完成信号
                self.epoch_completed.emit(epoch, train_loss, val_loss)
                
                # 更新进度条
                progress = int(10 + 85 * epoch / epochs)
                self.progress_updated.emit(progress)
                
                # 模拟检查点保存
                if use_checkpoint and epoch % 5 == 0:
                    self.training_log.emit(f"保存检查点: epoch_{epoch}.pt")
                
                # 模拟早停
                if use_early_stopping and val_loss is not None and val_loss < 0.3:
                    self.training_log.emit("触发早停: 验证损失已达到目标")
                    break
            
            # 模拟保存最终模型
            final_model_path = os.path.join(model_output_path, f"{model_architecture}_final.pt")
            self.training_log.emit(f"保存最终模型: {final_model_path}")
            time.sleep(1)  # 模拟保存时间
            
            self.progress_updated.emit(100)
            self.training_finished.emit(True, "训练完成")
            
        except Exception as e:
            self.training_log.emit(f"训练出错: {str(e)}")
            self.training_finished.emit(False, f"训练失败: {str(e)}")
        
        finally:
            self.is_running = False
    
    def pause(self):
        """暂停训练"""
        self.is_paused = True
    
    def resume(self):
        """恢复训练"""
        self.is_paused = False
    
    def stop(self):
        """停止训练"""
        self.is_running = False


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

    def setup_appearance(self):
        """设置外观样式"""
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
        
        # 创建"生成谱面"选项卡
        generate_tab = QtWidgets.QWidget()
        tab_widget.addTab(generate_tab, "生成谱面")
        
        # 创建"谱面分析"选项卡
        beatmap_analysis_tab = QtWidgets.QWidget()
        tab_widget.addTab(beatmap_analysis_tab, "谱面分析")
        
        # 创建"谱面预览"选项卡
        preview_tab = QtWidgets.QWidget()
        tab_widget.addTab(preview_tab, "谱面预览")
        
        # 创建"数据集处理"选项卡
        dataset_tab = QtWidgets.QWidget()
        tab_widget.addTab(dataset_tab, "数据集处理")
        
        # 创建"模型训练"选项卡
        model_training_tab = QtWidgets.QWidget()
        tab_widget.addTab(model_training_tab, "模型训练")
        
        # 创建"设置"选项卡
        settings_tab = QtWidgets.QWidget()
        tab_widget.addTab(settings_tab, "设置")
        
        # 设置"生成谱面"选项卡的布局
        generate_layout = QtWidgets.QVBoxLayout(generate_tab)
        generate_layout.setSpacing(15)
        
        # 文件选择部分
        file_group = QtWidgets.QGroupBox("音频文件")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        
        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setPlaceholderText("请选择.mp3或.wav文件...")
        
        browse_btn = QtWidgets.QPushButton("浏览")
        try:
            browse_btn.setIcon(QtGui.QIcon("gui/resources/folder_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        browse_btn.clicked.connect(self.browse_audio)
        
        file_layout.addWidget(self.file_path, 3)
        file_layout.addWidget(browse_btn, 1)
        
        generate_layout.addWidget(file_group)
        
        # 谱面参数部分
        params_container = QtWidgets.QHBoxLayout()
        
        # 基本参数组
        basic_params_group = QtWidgets.QGroupBox("基本参数")
        basic_params_layout = QtWidgets.QFormLayout(basic_params_group)
        
        self.difficulty_combo = QtWidgets.QComboBox()
        self.difficulty_combo.addItems(["Easy", "Normal", "Hard", "Expert", "Expert+"])
        
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(["Standard", "Stream", "Jump", "Technical"])
        
        self.ar_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ar_slider.setRange(0, 10)
        self.ar_slider.setValue(8)
        
        self.od_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.od_slider.setRange(0, 10)
        self.od_slider.setValue(7)
        
        # 添加表单项
        basic_params_layout.addRow("难度级别:", self.difficulty_combo)
        basic_params_layout.addRow("谱面风格:", self.style_combo)
        basic_params_layout.addRow("AR值 (8):", self.ar_slider)
        basic_params_layout.addRow("OD值 (7):", self.od_slider)
        
        # 连接滑块值变化的信号
        self.ar_slider.valueChanged.connect(
            lambda value: basic_params_layout.itemAt(4).widget().setText(f"AR值 ({value}):")
        )
        self.od_slider.valueChanged.connect(
            lambda value: basic_params_layout.itemAt(6).widget().setText(f"OD值 ({value}):")
        )
        
        # 高级参数组
        advanced_params_group = QtWidgets.QGroupBox("高级参数")
        advanced_params_layout = QtWidgets.QFormLayout(advanced_params_group)
        
        self.stream_density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stream_density_slider.setRange(0, 100)
        self.stream_density_slider.setValue(50)
        
        self.jump_intensity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.jump_intensity_slider.setRange(0, 100)
        self.jump_intensity_slider.setValue(50)
        
        # 添加表单项
        advanced_params_layout.addRow("流串密度 (50%):", self.stream_density_slider)
        advanced_params_layout.addRow("跳跃强度 (50%):", self.jump_intensity_slider)
        
        # 连接滑块值变化的信号
        self.stream_density_slider.valueChanged.connect(
            lambda value: advanced_params_layout.itemAt(0).widget().setText(f"流串密度 ({value}%):")
        )
        self.jump_intensity_slider.valueChanged.connect(
            lambda value: advanced_params_layout.itemAt(2).widget().setText(f"跳跃强度 ({value}%):")
        )
        
        # 添加参数组到容器
        params_container.addWidget(basic_params_group)
        params_container.addWidget(advanced_params_group)
        
        generate_layout.addLayout(params_container)
        
        # 操作区域
        actions_layout = QtWidgets.QHBoxLayout()
        
        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        
        # 操作按钮
        analyze_btn = QtWidgets.QPushButton("分析音频")
        try:
            analyze_btn.setIcon(QtGui.QIcon("gui/resources/analyze_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        analyze_btn.clicked.connect(self.analyze_audio)
        
        generate_btn = QtWidgets.QPushButton("生成谱面")
        try:
            generate_btn.setIcon(QtGui.QIcon("gui/resources/generate_icon.png"))
        except:
            pass  # 如果图标不存在，则不设置图标
        generate_btn.clicked.connect(self.generate_beatmap)
        
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
        actions_layout.addWidget(analyze_btn, 1)
        actions_layout.addWidget(generate_btn, 1)
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
        
        dataset_params_layout.addWidget(output_folder_label, 6, 0)
        dataset_params_layout.addLayout(output_folder_layout, 6, 1)
        
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
        
        # 设置模型训练选项卡内容
        training_layout = QtWidgets.QVBoxLayout(model_training_tab)
        
        # 创建滚动区域，确保在窗口较小时能显示所有内容
        training_scroll_area = QtWidgets.QScrollArea()
        training_scroll_area.setWidgetResizable(True)
        training_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # 创建滚动区域的内容容器
        training_scroll_content = QtWidgets.QWidget()
        training_scroll_layout = QtWidgets.QVBoxLayout(training_scroll_content)
        training_scroll_layout.setSpacing(15)
        
        # 训练数据设置
        training_data_group = QtWidgets.QGroupBox("训练数据设置")
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
        
        training_scroll_layout.addWidget(training_data_group)
        
        # 模型训练参数
        training_params_group = QtWidgets.QGroupBox("训练参数")
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
        
        training_scroll_layout.addWidget(training_params_group)
        
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
        
        training_scroll_layout.addWidget(training_actions_group)
        
        # 设置滚动区域的内容并添加到主布局
        training_scroll_area.setWidget(training_scroll_content)
        training_layout.addWidget(training_scroll_area)
        
        # 设置设置选项卡内容
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)
        
        # 模型设置
        model_group = QtWidgets.QGroupBox("模型设置")
        model_layout = QtWidgets.QFormLayout(model_group)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["默认模型", "流行风格", "古典风格", "电子风格"])
        
        model_layout.addRow("生成模型:", self.model_combo)
        
        # 输出设置
        output_group = QtWidgets.QGroupBox("输出设置")
        output_layout = QtWidgets.QFormLayout(output_group)
        
        self.output_path = QtWidgets.QLineEdit()
        self.output_path.setPlaceholderText("默认：与音频文件相同目录")
        
        browse_output_btn = QtWidgets.QPushButton("浏览")
        browse_output_btn.setFixedWidth(100)
        browse_output_btn.clicked.connect(self.browse_output_directory)
        
        output_path_layout = QtWidgets.QHBoxLayout()
        output_path_layout.addWidget(self.output_path)
        output_path_layout.addWidget(browse_output_btn)
        
        output_layout.addRow("输出目录:", output_path_layout)
        
        # 添加导出选项
        export_options_group = QtWidgets.QGroupBox("导出选项")
        export_options_layout = QtWidgets.QVBoxLayout(export_options_group)
        
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
        
        # 添加设置组到设置布局
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(output_group)
        settings_layout.addWidget(export_options_group)
        
        # 添加可视化设置组
        visualization_group = QtWidgets.QGroupBox("可视化设置")
        visualization_layout = QtWidgets.QVBoxLayout(visualization_group)
        
        # 添加启用可视化的复选框
        self.enable_visualization_cb = QtWidgets.QCheckBox("启用音频可视化 (可能影响性能)")
        self.enable_visualization_cb.setChecked(False)  # 默认关闭
        self.enable_visualization_cb.stateChanged.connect(self.toggle_visualization)
        
        # 添加可视化质量选项
        viz_quality_layout = QtWidgets.QHBoxLayout()
        
        viz_quality_label = QtWidgets.QLabel("可视化质量:")
        self.viz_quality_combo = QtWidgets.QComboBox()
        self.viz_quality_combo.addItems(["低 (流畅)", "中 (平衡)", "高 (精细)"])
        self.viz_quality_combo.setCurrentIndex(1)  # 默认选择中等质量
        
        viz_quality_layout.addWidget(viz_quality_label)
        viz_quality_layout.addWidget(self.viz_quality_combo)
        viz_quality_layout.addStretch()
        
        # 添加高级可视化选项
        self.enable_realtime_viz_cb = QtWidgets.QCheckBox("启用实时可视化 (需要更高性能)")
        self.enable_realtime_viz_cb.setChecked(False)
        
        self.cache_visualizations_cb = QtWidgets.QCheckBox("缓存可视化结果")
        self.cache_visualizations_cb.setChecked(True)
        
        # 添加到可视化布局
        visualization_layout.addWidget(self.enable_visualization_cb)
        visualization_layout.addLayout(viz_quality_layout)
        visualization_layout.addWidget(self.enable_realtime_viz_cb)
        visualization_layout.addWidget(self.cache_visualizations_cb)
        
        # 添加设置组到设置布局
        settings_layout.addWidget(visualization_group)
        settings_layout.addStretch()
        # 创建状态栏
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("准备就绪")
    
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
        file_path = self.file_path.text()
        if not file_path or not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的音频文件")
            return
        
        # 更新UI状态
        self.status_label.setText("正在加载音频...")
        self.progress_bar.setValue(0)
        
        # 加载音频文件
        if not self.audio_analyzer.load_audio(file_path):
            QtWidgets.QMessageBox.warning(self, "错误", "音频文件加载失败")
            self.status_label.setText("音频加载失败")
            return
        
        # 更新UI状态
        self.status_label.setText("正在分析音频...")
        
        # 在后台线程中执行分析，避免界面卡顿
        self.analysis_thread = QtCore.QThread()
        self.audio_analyzer.moveToThread(self.analysis_thread)
        
        # 连接信号
        self.analysis_thread.started.connect(self.audio_analyzer.analyze)
        
        # 启动线程
        self.analysis_thread.start()
    
    def update_analysis_progress(self, progress):
        """更新分析进度"""
        self.progress_bar.setValue(progress)
        
        # 根据进度更新状态文本
        if progress < 20:
            self.status_label.setText("正在检测BPM和节拍...")
        elif progress < 40:
            self.status_label.setText("正在分析节拍强度...")
        elif progress < 60:
            self.status_label.setText("正在提取频谱特征...")
        elif progress < 80:
            self.status_label.setText("正在检测音频段落...")
        else:
            self.status_label.setText("正在完成分析...")
    
    def handle_analysis_complete(self, features):
        """处理分析完成"""
        # 结束分析线程
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        
        # 更新UI状态
        self.progress_bar.setValue(100)
        
        # 获取基本音频信息
        bpm = features.get("bpm", 0)
        beat_count = len(features.get("beat_times", []))
        
        # 更新状态文本
        self.status_label.setText(f"音频分析完成! BPM: {bpm}, 节拍数: {beat_count}")
        
        # 注释掉提示框，避免打断后续操作
        # QtWidgets.QMessageBox.information(
        #     self, "分析完成", 
        #     f"音频分析完成！\n"
        #     f"检测到BPM：{bpm}\n"
        #     f"节拍数：{beat_count}\n"
        #     f"音频长度：{features.get('duration', 0):.1f}秒"
        # )
        
        # 仅当可视化功能启用时才更新可视化器
        if self.enable_visualization_cb.isChecked():
            self.audio_visualizer.set_audio_data(self.audio_analyzer.y, self.audio_analyzer.sr)
            self.audio_visualizer.set_audio_features(features)
        
        # 根据分析结果预设谱面参数
        density_suggestions = self.audio_analyzer.get_density_suggestion()
        
        # 更新流串密度滑块
        if "stream_density" in density_suggestions:
            density = int(density_suggestions["stream_density"] * 100)
            self.stream_density_slider.setValue(density)
        
        # 更新跳跃强度滑块
        if "jump_intensity" in density_suggestions:
            intensity = int(density_suggestions["jump_intensity"] * 100)
            self.jump_intensity_slider.setValue(intensity)
        
        # 检查是否需要自动导出
        if self.auto_export_cb.isChecked():
            self.export_analysis()
    
    def handle_analysis_error(self, error_message):
        """处理分析错误"""
        # 结束分析线程
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        
        # 更新UI状态
        self.progress_bar.setValue(0)
        self.status_label.setText("分析失败")
        
        # 显示错误消息
        QtWidgets.QMessageBox.critical(
            self, "分析错误", 
            f"音频分析过程中出错：\n{error_message}"
        )
    
    def generate_beatmap(self):
        """生成谱面"""
        file_path = self.file_path.text()
        if not file_path or not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择有效的音频文件")
            return
        
        # 这里实际上会调用谱面生成模块生成谱面
        # 为了演示，我们模拟一个进度条
        self.status_label.setText("生成谱面中...")
        self.progress_bar.setValue(0)
        
        for i in range(101):
            QtCore.QCoreApplication.processEvents()
            self.progress_bar.setValue(i)
            QtCore.QThread.msleep(30)
        
        self.status_label.setText("谱面生成完成！")
        # 注释掉提示框，避免打断批量处理
        # QtWidgets.QMessageBox.information(
        #     self, "成功", 
        #     "谱面生成完成！\n已保存至：" + os.path.splitext(file_path)[0] + ".osu"
        # )
    
    def preview_beatmap(self):
        """预览谱面"""
        # 检查是否已经加载了音频
        if not hasattr(self.audio_analyzer, 'y') or self.audio_analyzer.y is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先分析音频文件")
            return
        
        # 切换到预览选项卡
        tab_widget = self.centralWidget().findChild(QtWidgets.QTabWidget)
        tab_widget.setCurrentIndex(1)
        
        # 这里实际上会调用谱面预览模块显示谱面
        self.status_label.setText("谱面预览模式")
        
        # 仅当可视化功能启用时才更新可视化器
        if self.enable_visualization_cb.isChecked():
            # 确保可视化器已经设置了音频数据
            if self.audio_visualizer.audio_data is None:
                self.audio_visualizer.set_audio_data(self.audio_analyzer.y, self.audio_analyzer.sr)
            
            # 如果有分析结果，也设置给可视化器
            if self.audio_analyzer.features:
                self.audio_visualizer.set_audio_features(self.audio_analyzer.features)
            
            # 切换到波形图视图，并确保显示节拍
            self.audio_visualizer.waveform_btn.setChecked(True)
            self.audio_visualizer.show_beats_cb.setChecked(True)
            self.audio_visualizer.update_visualization()
        else:
            # 提示用户可视化功能已禁用
            self.status_label.setText("谱面预览模式 (可视化已禁用)")
    
    def export_analysis(self):
        """导出音频分析结果"""
        # 检查是否已经进行了分析
        if not hasattr(self.audio_analyzer, 'features') or not self.audio_analyzer.features:
            QtWidgets.QMessageBox.warning(self, "警告", "请先分析音频文件")
            return
        
        # 确定默认保存路径
        if self.output_path.text():
            # 如果设置了输出目录，使用该目录
            default_dir = self.output_path.text()
            audio_filepath = self.file_path.text() if self.file_path.text() else ""
            audio_filename = os.path.basename(audio_filepath)
            
            # 获取音频文件所在目录名称作为前缀，确保文件名唯一
            parent_dir_name = "unknown"
            if audio_filepath:
                parent_dir = os.path.dirname(audio_filepath)
                if parent_dir:
                    parent_dir_name = os.path.basename(parent_dir)
            
            # 添加时间戳确保唯一性
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 构建基础文件名
            base_filename = os.path.splitext(audio_filename)[0]
            unique_filename = f"{parent_dir_name}_{base_filename}_{timestamp}"
            
            default_path = os.path.join(default_dir, f"{unique_filename}.analysis.json")
        else:
            # 否则使用音频文件所在目录
            audio_filepath = self.file_path.text() if self.file_path.text() else ""
            if audio_filepath:
                # 获取音频文件所在目录名称
                parent_dir = os.path.dirname(audio_filepath)
                parent_dir_name = os.path.basename(parent_dir) if parent_dir else "unknown"
                
                # 添加时间戳确保唯一性
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 构建基础文件名
                base_filename = os.path.splitext(os.path.basename(audio_filepath))[0]
                unique_filename = f"{parent_dir_name}_{base_filename}_{timestamp}"
                
                default_path = os.path.join(os.path.dirname(audio_filepath), f"{unique_filename}.analysis.json")
            else:
                default_path = ""
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存分析结果", default_path, "JSON文件 (*.json)"
        )
        
        if not file_path:
            return  # 用户取消了保存
        
        # 显示保存进度
        self.status_label.setText("正在导出分析结果...")
        self.progress_bar.setValue(50)
        QtCore.QCoreApplication.processEvents()
        
        try:
            # 根据设置过滤要导出的数据
            export_data = self.audio_analyzer.features.copy()
            
            # 如果不导出可视化数据
            if not self.export_visualization_cb.isChecked() and "visualization" in export_data:
                del export_data["visualization"]
            
            # 过滤其他选项
            if not self.export_bpm_cb.isChecked():
                keys_to_remove = ["bpm", "beat_times", "beat_strengths", "beat_regularity", 
                                 "strong_beats", "beat_grid", "grid_mapped_beats", "osu"]
                for key in keys_to_remove:
                    if key in export_data:
                        del export_data[key]
            
            if not self.export_spectrum_cb.isChecked() and "spectral" in export_data:
                del export_data["spectral"]
            
            if not self.export_volume_cb.isChecked() and "volume" in export_data:
                del export_data["volume"]
                if "volume_changes" in export_data:
                    del export_data["volume_changes"]
            
            if not self.export_sections_cb.isChecked():
                if "sections" in export_data:
                    del export_data["sections"]
                if "transitions" in export_data:
                    del export_data["transitions"]
            
            # 确定JSON格式
            indent = 2 if self.json_pretty_rb.isChecked() else None
            
            # 自定义导出，而不是使用内置方法
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(export_data, f, indent=indent, ensure_ascii=False)
            
            output_path = file_path
            
            # 更新UI状态
            self.progress_bar.setValue(100)
            self.status_label.setText(f"分析结果已导出至: {os.path.basename(output_path)}")
            
            # 注释掉提示框，避免打断后续操作
            # QtWidgets.QMessageBox.information(
            #     self, "导出成功", 
            #     f"音频分析结果已成功导出至:\n{output_path}"
            # )
            
            # 注释掉询问是否打开所在文件夹的代码，避免打断后续操作
            # reply = QtWidgets.QMessageBox.question(
            #     self, "打开文件夹", 
            #     "是否要打开导出文件所在的文件夹?",
            #     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            #     QtWidgets.QMessageBox.No
            # )
            
            # if reply == QtWidgets.QMessageBox.Yes:
            #     # 打开文件所在文件夹
            #     folder_path = os.path.dirname(output_path)
            #     os.startfile(folder_path) if os.name == 'nt' else os.system(f'xdg-open "{folder_path}"')
                
        except Exception as e:
            # 显示错误消息
            self.progress_bar.setValue(0)
            self.status_label.setText("导出失败")
            
            QtWidgets.QMessageBox.critical(
                self, "导出错误", 
                f"导出分析结果时出错:\n{str(e)}"
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
        
        mode = self.mode_combo.currentText()
        selected_difficulty = self.difficulty_combo.currentText()
        limit = self.files_limit_spin.value()
        
        self.dataset_files_list.clear()
        self.dataset_progress_bar.setValue(0)
        self.dataset_status_label.setText("正在扫描文件夹...")
        
        # 开始扫描
        found_files = []
        total_scanned = 0
        
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
        
        # 递归扫描文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".osu"):
                    total_scanned += 1
                    file_path = os.path.join(root, file)
                    
                    # 简单检查文件内容以确定它是否符合要求
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # 检查模式
                            mode_line = [line for line in content.split("\n") if "Mode:" in line]
                            if mode_line:
                                file_mode = mode_line[0].split(":")[-1].strip()
                                if mode != "std" and mode != file_mode:
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
        self.dataset_status_label.setText(f"找到 {len(found_files)} 个符合条件的谱面文件")
    
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
        
        # 设置每批处理的文件数
        batch_size = self.batch_size_spin.value()
        total_batches = (total_files + batch_size - 1) // batch_size  # 向上取整计算批次数
        
        for batch_index in range(total_batches):
            # 获取当前批次的文件
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = beatmap_files[start_idx:end_idx]
            
            batch_results = []
            
            # 更新状态
            self.dataset_status_label.setText(f"正在处理批次 {batch_index+1}/{total_batches}...")
            self.dataset_progress_bar.setValue(int(batch_index / total_batches * 100))
            QtCore.QCoreApplication.processEvents()
            
            # 处理当前批次的文件
            for file_idx, file_path in enumerate(batch_files):
                try:
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
                                print(f"无法分析音频文件 {audio_file}: {str(e)}")
                        
                        # 将结果添加到批次结果和总结果
                        item_result = {
                            "beatmap_file": file_path,
                            "audio_file": audio_file,
                            "analysis": result
                        }
                        batch_results.append(item_result)
                        all_results.append(item_result)
                
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                
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
                    f"已处理: {processed_files}/{total_files} 个文件"
                )
                QtCore.QCoreApplication.processEvents()
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "警告", 
                    f"保存批次 {batch_index+1} 时出错: {str(e)}"
                )
        
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
                    random.shuffle(folder_paths)
                    
                    # 计算每个集合应包含的文件夹数量
                    total_folders = len(folder_paths)
                    train_folders_count = int(total_folders * train_percent)
                    val_folders_count = int(total_folders * val_percent)
                    
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
                
                self.dataset_status_label.setText(
                    f"数据集处理完成，已分割为训练集({len(train_data)}个样本)、"
                    f"验证集({len(val_data)}个样本)和测试集({len(test_data)}个样本)"
                )
            else:
                # 不分割，直接保存完整数据集
                dataset_file = os.path.join(dataset_output_dir, "dataset.json")
                with open(dataset_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                
                self.dataset_status_label.setText(f"数据集处理完成，已保存到: {dataset_file}")
                success_message = f"成功处理 {processed_files} 个文件!\n数据集已保存到: {dataset_file}"
            
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
        self.training_thread = TrainingThread(training_params)
        
        # 连接信号
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.epoch_completed.connect(self.update_training_plot)
        self.training_thread.training_finished.connect(self.handle_training_finished)
        self.training_thread.training_log.connect(self.add_training_log)
        
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
            self.training_thread.resume()
            self.pause_resume_btn.setText("暂停训练")
            self.training_status_label.setText("训练正在进行")
            self.add_training_log("训练已恢复")
        else:
            # 暂停训练
            self.training_thread.pause()
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
            self.training_thread.stop()
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

def main():
    """程序入口函数"""
    app = QtWidgets.QApplication(sys.argv)
    window = OsuStyleMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
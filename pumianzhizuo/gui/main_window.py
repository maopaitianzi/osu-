#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
osu!风格的谱面生成器主窗口
"""

import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

# 导入音频分析模块
from audio.analyzer import AudioAnalyzer
from audio.visualizer import AudioVisualizer


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
        
        # 连接信号
        self.audio_analyzer.analysis_progress.connect(self.update_analysis_progress)
        self.audio_analyzer.analysis_complete.connect(self.handle_analysis_complete)
        self.audio_analyzer.analysis_error.connect(self.handle_analysis_error)
        
        # 初始化UI
        self.init_ui()

    def setup_appearance(self):
        """设置外观样式"""
        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A1A;
                color: #FFFFFF;
            }
            QGroupBox {
                background-color: #282828;
                border: 2px solid #FF66AA;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #FFFFFF;
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
                color: #FFFFFF;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #FFFFFF;
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
                background-color: #282828;
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
                border: 1px solid #999999;
                height: 8px;
                background: #333333;
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
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #333333;
                color: white;
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
        
        # 创建"谱面预览"选项卡
        preview_tab = QtWidgets.QWidget()
        tab_widget.addTab(preview_tab, "谱面预览")
        
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
        self.status_label.setStyleSheet("color: #AAAAAA;")
        
        status_layout.addWidget(self.status_label)
        
        generate_layout.addLayout(status_layout)
        
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
            color: #AAAAAA;
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
        
        # 显示成功消息
        QtWidgets.QMessageBox.information(
            self, "分析完成", 
            f"音频分析完成！\n"
            f"检测到BPM：{bpm}\n"
            f"节拍数：{beat_count}\n"
            f"音频长度：{features.get('duration', 0):.1f}秒"
        )
        
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
        QtWidgets.QMessageBox.information(
            self, "成功", 
            "谱面生成完成！\n已保存至：" + os.path.splitext(file_path)[0] + ".osu"
        )
    
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
            audio_filename = os.path.basename(self.file_path.text()) if self.file_path.text() else "audio"
            base_filename = os.path.splitext(audio_filename)[0]
            default_path = os.path.join(default_dir, f"{base_filename}.analysis.json")
        else:
            # 否则使用音频文件所在目录
            default_path = os.path.splitext(self.file_path.text())[0] + ".analysis.json" if self.file_path.text() else ""
        
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
            
            # 显示成功消息
            QtWidgets.QMessageBox.information(
                self, "导出成功", 
                f"音频分析结果已成功导出至:\n{output_path}"
            )
            
            # 询问是否打开所在文件夹
            reply = QtWidgets.QMessageBox.question(
                self, "打开文件夹", 
                "是否要打开导出文件所在的文件夹?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                # 打开文件所在文件夹
                folder_path = os.path.dirname(output_path)
                os.startfile(folder_path) if os.name == 'nt' else os.system(f'xdg-open "{folder_path}"')
                
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


def main():
    """程序入口函数"""
    app = QtWidgets.QApplication(sys.argv)
    window = OsuStyleMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
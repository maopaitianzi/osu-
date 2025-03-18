#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
osu!风格的启动动画屏幕
"""

import os
import sys
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QFontDatabase


class OsuSplashScreen(QtWidgets.QSplashScreen):
    """osu!风格的启动动画屏幕类"""
    
    # 添加动画完成信号
    animation_finished = pyqtSignal()
    
    def __init__(self, parent=None):
        # 创建一个透明的pixmap作为启动画面的背景
        pixmap = QtGui.QPixmap(600, 400)
        pixmap.fill(Qt.transparent)
        super().__init__(pixmap)
        
        # 设置窗口无边框和透明背景
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 加载logo
        self.logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OSUMAP.png")
        self.logo = QtGui.QPixmap(self.logo_path) if os.path.exists(self.logo_path) else None
        if self.logo:
            self.logo = self.logo.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 动画相关变量
        self.opacity = 0.0
        self.progress = 0
        self.loading_dots = ""
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.update_dots)
        self.dot_timer.start(500)  # 每500毫秒更新一次点点
        
        # 设置渐变动画
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(1000)  # 1秒淡入
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_in_animation.start()
        
        # 设置logo动画 - 使用_logo_scale作为内部变量
        self._logo_scale = 0.7
        self.logo_animation = QPropertyAnimation(self, b"logo_scale")
        self.logo_animation.setDuration(800)
        self.logo_animation.setStartValue(0.7)
        self.logo_animation.setEndValue(1.0)
        self.logo_animation.setEasingCurve(QEasingCurve.OutElastic)
        self.logo_animation.start()
        
        # 连接动画完成信号
        self.logo_animation.finished.connect(self.start_bounce_animation)
        
        # 设置进度条动画
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(30)  # 每30毫秒更新一次进度
        
        # 标记动画是否完成
        self.animation_done = False
        self.main_window = None
    
    def start_bounce_animation(self):
        """启动弹跳动画"""
        self.bounce_animation = QPropertyAnimation(self, b"logo_scale")
        self.bounce_animation.setDuration(1500)
        self.bounce_animation.setStartValue(1.0)
        self.bounce_animation.setKeyValueAt(0.2, 0.95)
        self.bounce_animation.setKeyValueAt(0.4, 1.05)
        self.bounce_animation.setKeyValueAt(0.6, 0.97)
        self.bounce_animation.setKeyValueAt(0.8, 1.03)
        self.bounce_animation.setEndValue(1.0)
        self.bounce_animation.setEasingCurve(QEasingCurve.OutElastic)
        self.bounce_animation.start()
    
    def get_logo_scale(self):
        """获取logo缩放值"""
        return self._logo_scale
    
    def set_logo_scale(self, scale):
        """设置logo缩放值并重绘"""
        self._logo_scale = scale
        self.update()
    
    # 定义属性以便动画可以使用，使用不同的名称避免无限递归
    logo_scale = QtCore.pyqtProperty(float, get_logo_scale, set_logo_scale)
    
    def update_dots(self):
        """更新加载点点动画"""
        self.loading_dots = "." * ((len(self.loading_dots) + 1) % 4)
        self.update()
    
    def update_progress(self):
        """更新进度条"""
        if self.progress < 100:
            self.progress += 1
            self.update()
        else:
            self.progress_timer.stop()
            # 进度条到100%后，等待一小段时间再发出完成信号
            QTimer.singleShot(500, self.on_animation_complete)
    
    def on_animation_complete(self):
        """动画完成后的处理"""
        self.animation_done = True
        # 发出动画完成信号
        self.animation_finished.emit()
        # 如果已经设置了主窗口，则自动完成启动动画
        if self.main_window:
            self.finish(self.main_window)
    
    def drawContents(self, painter):
        """绘制启动画面内容"""
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 绘制半透明背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 20, 20)
        
        # 绘制logo
        if self.logo:
            scaled_logo = self.logo.scaled(
                int(self.logo.width() * self._logo_scale),
                int(self.logo.height() * self._logo_scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            logo_x = (self.width() - scaled_logo.width()) // 2
            logo_y = (self.height() - scaled_logo.height()) // 3
            painter.drawPixmap(logo_x, logo_y, scaled_logo)
        
        # 绘制加载文本
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 12)
        font.setBold(True)
        painter.setFont(font)
        loading_text = f"正在加载{self.loading_dots}"
        painter.drawText(
            QRect(0, int(self.height() * 0.65), self.width(), 30),
            Qt.AlignCenter,
            loading_text
        )
        
        # 绘制进度条背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(80, 80, 80, 200))
        progress_rect = QRect(
            int(self.width() * 0.2),
            int(self.height() * 0.75),
            int(self.width() * 0.6),
            10
        )
        painter.drawRoundedRect(progress_rect, 5, 5)
        
        # 绘制进度条
        painter.setBrush(QColor(255, 102, 170))  # osu!粉色 #FF66AA
        progress_width = int(progress_rect.width() * self.progress / 100)
        progress_fill_rect = QRect(
            progress_rect.x(),
            progress_rect.y(),
            progress_width,
            progress_rect.height()
        )
        painter.drawRoundedRect(progress_fill_rect, 5, 5)
        
        # 绘制进度文本
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 10)
        painter.setFont(font)
        progress_text = f"{self.progress}%"
        painter.drawText(
            QRect(0, int(self.height() * 0.8), self.width(), 20),
            Qt.AlignCenter,
            progress_text
        )
        
        # 绘制版权信息
        painter.setPen(QColor(180, 180, 180))
        font = QFont("Arial", 8)
        painter.setFont(font)
        copyright_text = "osu!谱面生成器 © 2024"
        painter.drawText(
            QRect(0, int(self.height() * 0.9), self.width(), 20),
            Qt.AlignCenter,
            copyright_text
        )
    
    def finish(self, main_window):
        """完成启动动画并显示主窗口"""
        # 保存主窗口引用
        self.main_window = main_window
        
        # 如果动画尚未完成，则不执行finish操作
        if not self.animation_done:
            return
            
        # 创建淡出动画
        self.fade_out_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out_animation.setDuration(800)  # 0.8秒淡出
        self.fade_out_animation.setStartValue(1.0)
        self.fade_out_animation.setEndValue(0.0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # 连接动画完成信号，显示主窗口并关闭启动画面
        self.fade_out_animation.finished.connect(lambda: self._finish_animation(main_window))
        self.fade_out_animation.start()
    
    def _finish_animation(self, main_window):
        """动画完成后的处理函数"""
        # 显示主窗口
        main_window.show()
        # 关闭启动画面
        super().finish(main_window)
        
        # 停止所有计时器
        if self.dot_timer.isActive():
            self.dot_timer.stop()
        if self.progress_timer.isActive():
            self.progress_timer.stop() 
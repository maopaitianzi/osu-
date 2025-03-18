#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
osu!风格的谱面生成器主程序入口
"""

import sys
import os
import time

# 将项目根目录添加到Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 获取根目录
sys.path.insert(0, root_dir)  # 将根目录添加到路径中

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui.main_window import OsuStyleMainWindow  # 恢复原来的导入
from gui.splash_screen import OsuSplashScreen  # 导入启动动画类


def main():
    """程序入口函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序图标
    icon_path = os.path.join(current_dir, "OSUMAP.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 创建主窗口，但不立即显示
    window = OsuStyleMainWindow()
    
    # 创建并显示启动动画
    splash = OsuSplashScreen()
    splash.show()
    
    # 确保启动动画显示
    app.processEvents()
    
    # 动画完成后再显示主窗口（不再使用固定延时）
    splash.finish(window)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
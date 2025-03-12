#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
osu!风格的谱面生成器主程序入口
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import OsuStyleMainWindow


def main():
    """程序入口函数"""
    app = QApplication(sys.argv)
    window = OsuStyleMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
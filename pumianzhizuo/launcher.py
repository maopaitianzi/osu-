#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OSU谱面生成器启动脚本
用于确保模块导入路径正确
"""

import os
import sys

def main():
    """
    主启动函数，设置环境并启动应用
    """
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保当前目录在路径中
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 确保父目录在路径中（确保能找到pumianzhizuo包）
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 打印路径调试信息
    print("Python路径:")
    for p in sys.path:
        print(f"  - {p}")
    
    # 尝试导入所需模块
    try:
        import main
        print("成功导入main模块")
    except ImportError as e:
        print(f"导入main模块失败: {e}")
        sys.exit(1)
    
    # 启动应用
    main.main()

if __name__ == "__main__":
    # 将工作目录设置为脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        main()
    except Exception as e:
        import traceback
        print(f"启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        
        # 保持控制台窗口打开
        input("\n按Enter键退出...")
        sys.exit(1) 
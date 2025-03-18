import sys
import os
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 打印当前路径和正在寻找的模块路径，帮助调试
print(f"当前目录: {current_dir}")
print(f"项目根目录: {project_root}")

# 尝试直接运行main.py中的代码
main_path = os.path.join(project_root, "pumianzhizuo", "main.py")
print(f"尝试加载模块: {main_path}")

# 假设GUI类和启动动画类在gui目录下
gui_path = os.path.join(project_root, "pumianzhizuo", "gui")
print(f"GUI目录: {gui_path}")

if os.path.exists(main_path):
    # 使用 importlib 动态导入带有特殊字符的模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", main_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    if __name__ == "__main__":
        main_module.main()
else:
    print(f"错误: 找不到主模块文件 {main_path}")
    print("尝试通过PyQt5直接启动GUI...")
    # 直接启动GUI作为备选方案
    import sys
    from PyQt5.QtWidgets import QApplication
    
    if os.path.exists(gui_path):
        sys.path.insert(0, gui_path)
        try:
            from main_window import OsuStyleMainWindow
            from splash_screen import OsuSplashScreen
            
            app = QApplication(sys.argv)
            
            # 设置应用程序图标
            icon_path = os.path.join(project_root, "pumianzhizuo", "OSUMAP.png")
            if os.path.exists(icon_path):
                from PyQt5.QtGui import QIcon
                app.setWindowIcon(QIcon(icon_path))
            
            # 创建主窗口但不立即显示
            window = OsuStyleMainWindow()
            
            # 显示启动动画
            splash = OsuSplashScreen()
            splash.show()
            app.processEvents()
            
            # 将主窗口传递给启动动画，动画会在完成后自动显示主窗口
            splash.finish(window)
            
            sys.exit(app.exec_())
        except ImportError as e:
            print(f"无法导入GUI模块: {e}")
    else:
        print(f"错误: 找不到GUI目录 {gui_path}") 

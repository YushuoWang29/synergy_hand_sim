"""
启动折纸手 CAD 工具
用法: python scripts/run_cad.py [--timeout SECONDS]

--timeout: 自动关闭时间（秒，0=禁用）
"""

import sys, argparse
sys.path.insert(0, '.')
from PyQt5 import QtWidgets, QtCore
from src.origami_cad.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="启动 Origami Hand CAD 工具")
    parser.add_argument("--timeout", type=float, default=0,
                        help="自动退出时间（秒，0=禁用）")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    
    if args.timeout > 0:
        print(f"CAD 将在 {args.timeout} 秒后自动关闭...")
        QtCore.QTimer.singleShot(int(args.timeout * 1000), app.quit)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


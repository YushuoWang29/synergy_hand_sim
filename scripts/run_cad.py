"""
启动折纸手 CAD 工具
用法: python scripts/run_cad.py
"""

import sys
sys.path.insert(0, '.')
from PyQt5 import QtWidgets
from src.origami_cad.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
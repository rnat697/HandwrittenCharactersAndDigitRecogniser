from scripts.MainMenu import mainGUI
import sys
from PyQt5.QtWidgets import *

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = mainGUI()
   ex.display();
   sys.exit(app.exec_())
# GUIs are hard http://stackoverflow.com/a/28291210

import numpy
import cv2
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import blender


class BlenderGUI(QDialog):
    def __init__(self, parent=None):
        super(BlenderGUI, self).__init__(parent)
        self.a = cv2.imread("picture.jpg")
        self.b = cv2.imread("replace.jpg")
        self.factor = 1
        self.result = blender.generate_midframe(self.a, self.b, self.factor);
        height, width, depth = self.result.shape
        depth = depth * width
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)

        self.mQImage = QImage(self.result, width, height, depth, QImage.Format_RGB888)

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()

    def keyPressEvent(self, QKeyEvent):
        super(BlenderGUI, self).keyPressEvent(QKeyEvent)
        if 'w' == QKeyEvent.text():
            self.factor += 0.2
            self.factor = numpy.clip(self.factor, 0, 1)
            self.result = cv2.cvtColor(blender.generate_midframe(self.a, self.b, self.factor), cv2.COLOR_BGR2RGB)
            height, width, depth = self.result.shape
            self.mQImage = QImage(self.result, width, height, depth, QImage.Format_RGB888)
        elif 'e' == QKeyEvent.text():
            self.factor -= 0.2
            self.factor = numpy.clip(self.factor, 0, 1)
            self.result = cv2.cvtColor(blender.generate_midframe(self.a, self.b, self.factor), cv2.COLOR_BGR2RGB)
            height, width, depth = self.result.shape
            self.mQImage = QImage(self.result, width, height, depth, QImage.Format_RGB888)
        elif 'q' == QKeyEvent.text():
            app.exit(1)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = BlenderGUI()
    w.resize(600, 400)
    w.show()
    app.exec_()

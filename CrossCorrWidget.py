import sys, re, math
import os.path
import numpy as np
import Constants as const
import CudaImageSupport as cimsup
import CudaCrossCorr as ccc
from functools import partial
from PyQt4 import QtGui, QtCore

# --------------------------------------------------------

class CheckButton(QtGui.QPushButton):
    def __init__(self, text, width, height):
        super(CheckButton, self).__init__(text)
        self.defaultStyle = 'background-color:transparent; color:transparent; width:{0}; height:{1}; border:1px solid rgb(0, 0, 0); padding:-1px;'.format(width, height)
        self.clickedStyle = 'background-color:rgba(255, 255, 255, 100); color:transparent; width:{0}; height:{1}; border:1px solid rgb(0, 0, 0); padding:-1px;'.format(width, height)
        self.wasClicked = False
        self.initUI()

    def initUI(self):
        self.setStyleSheet(self.defaultStyle)
        self.clicked.connect(self.handleButton)

    def handleButton(self):
        self.wasClicked = not self.wasClicked
        if self.wasClicked:
            self.setStyleSheet(self.clickedStyle)
        else:
            self.setStyleSheet(self.defaultStyle)

# --------------------------------------------------------

class ButtonGridOnLabel(QtGui.QLabel):
    def __init__(self, image, gridDim, parent):
        super(ButtonGridOnLabel, self).__init__(parent)
        self.grid = QtGui.QGridLayout()
        self.gridDim = gridDim
        self.image = image               # scaling and other changes will be executed on image buffer (not on image itself)
        self.initUI()

    def initUI(self):
        self.image.ReIm2AmPh()
        self.image.UpdateBuffer()
        self.createPixmap()
        self.grid.setMargin(0)
        self.grid.setSpacing(0)
        self.setLayout(self.grid)
        self.createGrid()

    def createPixmap(self):
        qImg = QtGui.QImage(cimsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
                           self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        # pixmap.convertFromImage(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)

    def changePixmap(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            self.createPixmap()

    def createGrid(self):
        if self.grid.count() > 0:
            rowCount = int(math.sqrt(self.grid.count()))
            colCount = rowCount
            old_positions = [(i, j) for i in range(rowCount) for j in range(colCount)]
            for pos in old_positions:
                button = self.grid.itemAtPosition(pos[0], pos[1]).widget()
                button.deleteLater()
                # self.grid.removeWidget(button)
                # button.setParent(None)

        positions = [(i, j) for i in range(self.gridDim) for j in range(self.gridDim)]
        btnWidth = math.ceil(const.ccWidgetDim / self.gridDim)

        for pos in positions:
            button = CheckButton('{0}'.format(pos), btnWidth, btnWidth)
            self.grid.addWidget(button, *pos)
        # print(self.grid.rowCount(), self.grid.columnCount())  # rowCount increases, but does not decrease

    def changeGrid(self, more=True):
        newGridDim = self.gridDim + 1 if more else self.gridDim - 1
        if 0 < newGridDim < 10:
            self.gridDim = newGridDim
            self.createGrid()

    def applyChangesToImage(self, image):
        image.UpdateImageFromBuffer()
        cropCoords = cimsup.DetermineCropCoords(image.width, image.height, image.shift)
        self.parent().commonCoords = cimsup.GetCommonArea(self.parent().commonCoords, cropCoords)

    def applyChangesToAll(self):
        self.applyChangesToImage(self.image)
        tmpNext = self.image
        tmpPrev = self.image
        while tmpNext.next is not None:
            tmpNext = tmpNext.next
            self.applyChangesToImage(tmpNext)
        while tmpPrev.prev is not None:
            tmpPrev = tmpPrev.prev
            self.applyChangesToImage(tmpPrev)
        # super() instead of parent()?
        self.parent().parent().statusBar().showMessage('All changes applied'.format(self.image.numInSeries))
        self.parent().parent().close()

    def resetImage(self):
        self.image.UpdateBuffer()
        self.image.shift = [0, 0]
        self.createPixmap()
        self.parent().parent().statusBar().showMessage('Image no {0} was reset'.format(self.image.numInSeries))

# --------------------------------------------------------

class LineEditWithLabel(QtGui.QWidget):
    def __init__(self, parent, labText='df', defaultValue=''):
        super(LineEditWithLabel, self).__init__(parent)
        self.label = QtGui.QLabel(labText)
        self.input = QtGui.QLineEdit(defaultValue)
        self.initUI()

    def initUI(self):
        # self.label.setFixedWidth(50)
        # self.input.setFixedWidth(50)
        self.setFixedWidth(50)
        self.input.setMaxLength(10)

        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(0)
        vbox.setSpacing(0)
        vbox.addWidget(self.label)
        vbox.addWidget(self.input)
        self.setLayout(vbox)

# --------------------------------------------------------

# move whole content of this class to CrossCorrWindow?
class CrossCorrWidget(QtGui.QWidget):
    def __init__(self, image, gridDim, parent):
        super(CrossCorrWidget, self).__init__(parent)
        self.btnGrid = ButtonGridOnLabel(image, gridDim, self)
        self.commonCoords = [0, 0, image.height, image.width]
        self.initUI()

    def initUI(self):
        prevButton = QtGui.QPushButton(QtGui.QIcon('prev.png'), '', self)
        prevButton.clicked.connect(partial(self.btnGrid.changePixmap, False))

        nextButton = QtGui.QPushButton(QtGui.QIcon('next.png'), '', self)
        nextButton.clicked.connect(partial(self.btnGrid.changePixmap, True))

        lessButton = QtGui.QPushButton(QtGui.QIcon('less.png'), '', self)
        lessButton.clicked.connect(partial(self.btnGrid.changeGrid, False))

        moreButton = QtGui.QPushButton(QtGui.QIcon('more.png'), '', self)
        moreButton.clicked.connect(partial(self.btnGrid.changeGrid, True))

        hbox_tl = QtGui.QHBoxLayout()
        hbox_tl.addWidget(prevButton)
        hbox_tl.addWidget(nextButton)

        hbox_ml = QtGui.QHBoxLayout()
        hbox_ml.addWidget(lessButton)
        hbox_ml.addWidget(moreButton)

        correlateButton = QtGui.QPushButton('Correlate!')
        correlateButton.clicked.connect(self.correlateImages)

        vbox_l = QtGui.QVBoxLayout()
        vbox_l.addLayout(hbox_tl)
        vbox_l.addLayout(hbox_ml)
        vbox_l.addWidget(correlateButton)

        # stepDropDownButton = QtGui.QToolButton(self)
        # stepDropDownButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        # stepDropDownButton.setMenu(QtGui.QMenu(stepDropDownButton))
        # stepDropDownButton.menu().addAction('10')
        # stepDropDownButton.menu().addAction('15')
        self.shiftStepEdit = QtGui.QLineEdit('5', self)
        self.shiftStepEdit.setFixedWidth(20)
        self.shiftStepEdit.setMaxLength(3)

        upButton = QtGui.QPushButton(QtGui.QIcon('up.png'), '', self)
        upButton.clicked.connect(self.movePixmapUp)

        downButton = QtGui.QPushButton(QtGui.QIcon('down.png'), '', self)
        downButton.clicked.connect(self.movePixmapDown)

        leftButton = QtGui.QPushButton(QtGui.QIcon('left.png'), '', self)
        leftButton.clicked.connect(self.movePixmapLeft)

        rightButton = QtGui.QPushButton(QtGui.QIcon('right.png'), '', self)
        rightButton.clicked.connect(self.movePixmapRight)

        hbox_mm = QtGui.QHBoxLayout()
        hbox_mm.addWidget(leftButton)
        hbox_mm.addWidget(self.shiftStepEdit)
        hbox_mm.addWidget(rightButton)

        vbox_m = QtGui.QVBoxLayout()
        vbox_m.addWidget(upButton)
        vbox_m.addLayout(hbox_mm)
        vbox_m.addWidget(downButton)

        self.dfMinEdit = LineEditWithLabel(self, labText='df min', defaultValue=str(const.dfStepMin))
        self.dfMaxEdit = LineEditWithLabel(self, labText='df max', defaultValue=str(const.dfStepMax))
        self.dfStepEdit = LineEditWithLabel(self, labText='step [um]', defaultValue=str(const.dfStepChange))

        hbox_mr = QtGui.QHBoxLayout()
        hbox_mr.addWidget(self.dfMinEdit)
        hbox_mr.addWidget(self.dfMaxEdit)
        hbox_mr.addWidget(self.dfStepEdit)

        applyChangesButton = QtGui.QPushButton('Apply changes', self)
        applyChangesButton.clicked.connect(self.btnGrid.applyChangesToAll)

        resetButton = QtGui.QPushButton('Reset image', self)
        resetButton.clicked.connect(self.btnGrid.resetImage)

        vbox_r = QtGui.QVBoxLayout()
        vbox_r.addLayout(hbox_mr)
        vbox_r.addWidget(applyChangesButton)
        vbox_r.addWidget(resetButton)

        hbox_main = QtGui.QHBoxLayout()
        hbox_main.addLayout(vbox_l)
        hbox_main.addLayout(vbox_m)
        hbox_main.addLayout(vbox_r)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addWidget(self.btnGrid)
        vbox_main.addLayout(hbox_main)
        self.setLayout(vbox_main)

    def movePixmapUp(self):
        ccc.MoveImageUp(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapDown(self):
        ccc.MoveImageDown(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapLeft(self):
        ccc.MoveImageLeft(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapRight(self):
        ccc.MoveImageRight(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    # move this method to ButtonGridOnLabel?
    def correlateImages(self):
        if self.btnGrid.image.prev is None:
            self.parent().statusBar().showMessage("Can't correlate. This is the reference image.")
            return

        self.parent().statusBar().showMessage('Correlating...')
        fragCoords = []
        for pos in range(self.btnGrid.grid.count()):
            btn = self.btnGrid.grid.itemAt(pos).widget()
            if btn.wasClicked:
                btn.handleButton()
                values = re.search('([0-9]), ([0-9])', btn.text())
                fragPos = (int(values.group(1)), int(values.group(2)))
                fragCoords.append(fragPos)

        image = self.btnGrid.image
        mcfBest = ccc.MaximizeMCFCore(image.prev, image, self.btnGrid.gridDim, fragCoords,
                                      float(self.dfMinEdit.input.text()),
                                      float(self.dfMaxEdit.input.text()),
                                      float(self.dfStepEdit.input.text()))
        shift = ccc.GetShift(mcfBest)
        image.shift = image.prev.shift
        ccc.ShiftImageAmpBuffer(image, shift)
        self.btnGrid.image.defocus = mcfBest.defocus
        self.btnGrid.createPixmap()
        ccfPath = const.ccfResultsDir + const.ccfName + str(image.numInSeries) + '.png'
        cimsup.SaveAmpImage(mcfBest, ccfPath)
        self.parent().statusBar().showMessage('Done! Image no {0} was shifted to image no {1}'.format(image.numInSeries, image.prev.numInSeries))

# --------------------------------------------------------

class CrossCorrWindow(QtGui.QMainWindow):
    def __init__(self, image, gridDim):
        super(CrossCorrWindow, self).__init__()
        self.ccWidget = CrossCorrWidget(image, gridDim, self)
        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('Ready')
        self.setCentralWidget(self.ccWidget)

        self.move(300, 300)
        self.setWindowTitle('Cross correlation window')
        self.setWindowIcon(QtGui.QIcon('world.png'))
        self.show()

# --------------------------------------------------------

def RunCrossCorrWindow(image, gridDim):
    app = QtGui.QApplication(sys.argv)
    ccWindow = CrossCorrWindow(image, gridDim)
    app.exec_()
    return ccWindow.ccWidget.commonCoords
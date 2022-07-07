
from asyncio.windows_events import NULL
from cProfile import label
from operator import index
from unittest.main import MODULE_EXAMPLES
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from PyQt5.QtGui import *
import numpy as np
from scripts.predictResults import predictResults


class predictSelect(QWidget):
    def __init__(self, images, labels, model,savedPath,epoch):

        super().__init__()
        self.model = model
        self.filepath  = savedPath
        self.epoch = epoch
        self.dataImages = images
        self.dataLabels = labels
        self.selectedImages=[]
        self.selectedLabels= []
        self.allImgArray = []
        self.array = [0]*200000
        self.labelImage = [0]*200000 

        self.initUI()
        self.showAllImages()
        self.display()
        

    def initUI(self):
        #sets up title and gui
        self.setWindowTitle('Prediction Selection')
        self.setGeometry(300, 300, 800, 600)  # (x,x,horizontal,vertical)
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)
        self.initialiseImages()
        self.createGrid()
        self.connectScrolltoGrid()
        self.predictButtons()

    def predictButtons(self):
        #creats button selection and link
        xaxis,yaxis = 600,50
        self.rbtn1 = QRadioButton('Select All', self)
        self.rbtn1.move(xaxis, yaxis)
        self.rbtn1.clicked.connect(self.selectAllImages)

        self.rbtn2 = QRadioButton('Custom Choose',self)
        self.rbtn2.move(xaxis, yaxis+50)
        self.rbtn2.clicked.connect(self.customChoose)

        self.enterCoord=QLineEdit('EnterCoord:',self)
        labelDescription = QLabel('The format is of y <space> x \nthe row is 0-x and column is 0-3 \neg 0 0 would refer to row 0 column 0\nThe first picture. ',self)
        labelDescription.move(580,250)
        self.enterCoord.move(600,150)
        self.enterCoord.returnPressed.connect(self.addtoArray)
        self.enterCoord.setDisabled(True)

        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)
        self.tb.move(600,200)
        


        self.btnTest = QPushButton('Test', self)
        self.btnCancel = QPushButton('Cancel',self)

        xaxis,yaxis=580,460
        self.btnTest.move(xaxis,yaxis)
        self.btnTest.resize(200,40)
        self.btnTest.clicked.connect(self.predictSend)
        self.btnCancel.move(xaxis,yaxis+60)
        self.btnCancel.resize(200,40)
        self.btnCancel.clicked.connect(self.cancelAction)

    def addtoArray(self):
        #for custom selection adds selected image to prediction array
        text = self.enterCoord.text()   #takes text from label
        stringlist=text.split()     #splits string  
        y=int(stringlist[0])    #takes row index
        x=int(stringlist[1])    #takes col index
        index = x + y*4         #4 images per row + column number to get image number
        self.labelImage[index].setStyleSheet("border: 5px solid red;")#indication u have selected this image
        self.selectedImages.append(self.dataImages[index])#add to predict images
        self.selectedLabels.append(self.dataLabels[index])#add to predict labels


    def selectAllImages(self):
            #returns all images
        self.enterCoord.setDisabled(True)
        for i in range(len(self.dataImages)): #cycles through all images entered into array
                self.labelImage[i].setStyleSheet("border: 5px solid red;")#highligh widget red

        self.selectedImages = self.dataImages#add to array
        self.selectedLabels =self.dataLabels#add to label

    def cancelAction(self):
        self.selectedImages =[] #resets image array to add
        self.selectedLabels= [] #resets label array to add
        self.rbtn1.setChecked(False) #resets buttons
        self.rbtn2.setChecked(False)

        for i in range(len(self.dataImages)):
                self.labelImage[i].setStyleSheet("border: 0px solid black;")#unhighlights all images

    def predictSend(self):
        #send selecte dimages to predictionResults
        self.predictResultGUI = predictResults(self.selectedImages,self.selectedLabels, self.model, self.filepath, self.epoch)
        self.close()

    def customChoose(self):
        #enables label to type in
        self.enterCoord.setEnabled(True)
        

    def initialiseImages(self):
        
        for i in range (len(self.dataImages)):
            image = self.dataImages[i][0]
            height, width = np.shape(image)

            # Creating a pixel map of the image for display
            qImg = QImage(image, width, height, QImage.Format_Grayscale8)
            pixmap01 = QPixmap.fromImage(qImg).scaled(112,112, Qt.KeepAspectRatio) # scaling image so we can see it better on GUI
            self.allImgArray.append(pixmap01)




    def createGrid(self):
        #initialises grid

        self.gridLayout = QGridLayout(self)
        self.column=1       #needed for inserting images.
        self.row=1 
        
    

    def connectScrolltoGrid(self):
        self.widget = QWidget()       # Scroll Area which contains the widgets, set as the centralWidget
        self.scroll = QScrollArea(self)

        self.widget.setLayout(self.gridLayout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll.move(20,50)
        self.scroll.resize(550,500) #800 ,600

    def showAllImages(self):
        #creates grid and loads images.
        # Displays all images
        self.gridClear()

        for i in range (len(self.dataImages)):
            self.addImage(i,i)
        self.TotalNumOfImgs = len(self.dataImages)

    def gridClear(self):
        # Checks if grid is empty and skips clearing the grid
        if (self.row == 1): 
            self.row = 1
            self.column =1

        else:
            # Clearing the grid
            for i in reversed(range(self.TotalNumOfImgs)):
                self.gridLayout.itemAt(i).widget().setParent(None)




    def addImage(self,index,imgNum):
        # add the pixmap to a label on the GUI
        text = str(self.dataLabels[index])
        self.labelImage[imgNum] = QLabel(text,self)  #creates new label
        self.labelImage[imgNum].setWordWrap(True)
        self.labelImage[imgNum].setPixmap(self.allImgArray[index])#deposits image into label

        #Sets grid size row and column depending on pref
        if (imgNum%4==0):    # 4 images per row.
            self.column = 1
            self.row += 1

        else:
            self.column+=1

        self.gridLayout.addWidget(self.labelImage[imgNum],self.row,self.column) #adds image in label to layout


    def display(self):
        #shows display
        self.show()


from scripts.Dataset import Dataset
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np



class imageViewerGUI(QWidget):
    def __init__(self, images, labels, testOrTrain):

        super().__init__()
        # Initalises the array of images, the array of labels, the window name (either Train Image Viewer or Test Image Viewer),
        # initialises the UI and displays the UI
        self.dataImages = images
        self.dataLabels = labels
        self.windowName = testOrTrain + " Image Viewer"
        self.allImgArray = []
        self.label = [0]* 700000

        self.initUI()
        self.display()
        

    def initUI(self):
        self.setWindowTitle(self.windowName)
        self.setGeometry(300, 300, 800, 600)  # (x,x,horizontal,vertical)
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)
        self.typeSelection()
        self.initialiseImages()
        self.createGrid()
        self.connectScrolltoGrid()
        self.showAllImages()


    def connectScrolltoGrid(self):
        #Connects scroll bar to the grid
        self.widget = QWidget()       # Scroll Area which contains the widgets, set as the centralWidget
        self.scroll = QScrollArea(self)

        self.widget.setLayout(self.gridLayout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll.move(20,50)
        self.scroll.resize(550,500) #800 ,600

    def createGrid(self):
        self.gridLayout = QGridLayout(self)
        self.column=1
        self.row=1

    def typeSelection(self):
        # Filter selection buttons and text box initialisaiton
        xaxis = 600
        yaxis = 50

        #All filter
        rbtn1 = QRadioButton('All', self)
        rbtn1.move(xaxis, yaxis)
        rbtn1.setChecked(True)
        rbtn1.clicked.connect(self.showAllImages)

        #Character Filter
        rbtn2 = QRadioButton('Character',self)
        rbtn2.move(xaxis, yaxis+50)
        self.characterBox = QLineEdit(self)
        self.characterBox.move(xaxis+100,yaxis+50)
        self.characterBox.resize(25,25)
        rbtn2.clicked.connect(self.filterCharacter)
        self.characterBox.textChanged[str].connect(self.onCharValueEntered)

        #Digit Filter
        rbtn3 = QRadioButton('Digit',self)
        rbtn3.move(xaxis, yaxis+100)
        self.digitBox = QLineEdit(self)
        self.digitBox.move(xaxis+100,yaxis+100)
        self.digitBox.resize(25,25)
        rbtn3.clicked.connect(self.filterDigit)
        self.digitBox.textChanged[str].connect(self.onDigitValueEntered)

        # Stats label
        self.statsLabel = QLabel("Total Number of images: ",self)
        self.statsLabel.move(590, 500)
        self.statsLabel.setFont(QFont('Arial', 11))
        self.statsLabel.setAlignment(Qt.AlignCenter)

        

    def onCharValueEntered(self):
        # Filters the letters to find a specific letter when character text box has its value changed
        characterBoxvalue = self.characterBox.text()
        self.findSpecificImages(characterBoxvalue)

    def onDigitValueEntered(self):
        # Filters the digits to find a specific digit when digit text box has its value changed
        digitBoxvalue = self.digitBox.text()
        self.findSpecificImages(digitBoxvalue)

    def showStats(self):
        # shows the total number of images
        stats = "Total Number of images:\n" + str(self.TotalNumOfImgs)
        self.statsLabel.setText(stats)


    def initialiseImages(self):
        # Initialises the images by creating a pixelmap of each image and saving it to an array
        # The pixelmap is used to display on the GUI
        for i in range (len(self.dataImages)):
            image = self.dataImages[i][0]
            height, width = np.shape(image)

            # Creating a pixel map of the image for display
            qImg = QImage(image, width, height, QImage.Format_Grayscale8)
            pixmap01 = QPixmap.fromImage(qImg).scaled(112,112, Qt.KeepAspectRatio) # scaling image so we can see it better on GUI
            self.allImgArray.append(pixmap01)

    def showAllImages(self):
        #loads all images on to a grid to display on the GUI
        self.gridClear() # clears grid if there's any images inside of it

        for i in range (len(self.dataImages)):
            self.addImage(i,i)
        self.TotalNumOfImgs = len(self.dataImages)
        self.showStats()

    def addImage(self,index,imgNum):
        # Add images on to the grid for display
        # Paramaters: index number to find where the image is on the allImgArray 
        # and the image number (Used for determining its position on the grid)

        # add the image pixmap to a label on the GUI
        text = str(self.dataLabels[index])
        self.label[imgNum] = QLabel(text,self)
        self.label[imgNum].setWordWrap(True)
        self.label[imgNum].setPixmap(self.allImgArray[index])


        # Determines position of the image on the grid
        if (imgNum%4==0):    # 4 images per row.
            self.column = 1
            self.row += 1

        else:
            self.column+=1

        self.gridLayout.addWidget(self.label[imgNum],self.row,self.column)

    def filterCharacter(self):
        # Filters the entire image array to show only characters

        self.gridClear()
        charIndexArray = []

        # Finds index of character images with labels greater than 9
        for i in range(len(self.dataLabels)):
            if(self.dataLabels[i] > 9):
                charIndexArray.append(i)

        # Goes through array of corrosponding index of the character image and adds the images to GUI
        for j in range(len(charIndexArray)):
            index = charIndexArray[j]
            self.addImage(index, j)
        self.TotalNumOfImgs = len(charIndexArray)
        self.showStats()

    def filterDigit(self):
        # Filters the entire image array to only show the digits

        self.gridClear()
        digitIndexArray = []

        # Finds index of digit images with labels less than 10
        for i in range(len(self.dataLabels)):
            if(self.dataLabels[i] < 10):
                digitIndexArray.append(i)

        # Goes through array of corrosponding index of the digit image and adds the images to GUI
        for j in range(len(digitIndexArray)):
            index = digitIndexArray[j]
            self.addImage(index, j)

        self.TotalNumOfImgs = len(digitIndexArray)
        self.showStats()

    def findSpecificImages(self, char):
         # Filters for a specific letter / digit and puts corrosponding index of that letter/digit image
         # then sends it to GUI for display

        #clear all values from grid box
        self.gridClear()

        findLabel = self.findAssociatedLabel(char)
        imageIndex = []

        # Finds index of digit images with labels that corrosponds to the specific letter/digit
        for i in range(len(self.dataLabels)):
            if(self.dataLabels[i] == findLabel):
                imageIndex.append(i)

        # Goes through array of corrosponding index of the character image and adds the images to GUI
        for j in range(len(imageIndex)):
            index = imageIndex[j]
            self.addImage(index, j)

        # Displays how many images of that specific letter/digit there is
        self.TotalNumOfImgs = len(imageIndex)
        self.showStats()

    def gridClear(self):
        # Clears images on the grid
        # Checks if grid is empty and skips clearing the grid
        if (self.row == 1): 
            self.row = 1
            self.column =1

        else:
            # Clearing the grid
            for i in reversed(range(self.TotalNumOfImgs)):
                self.gridLayout.itemAt(i).widget().setParent(None)

    def display(self):
        self.show()


## ----------------    CLIENT SIDE     ----------------
# #####################################################################################################################

    def findAssociatedLabel(self, findChar):
    # To find the associated Label number by comparing the index of charArray in order to filter
    # Returns the label number (from the index number of the array) associated to the letter/digit
        charArray = ["0","1","2","3","4","5","6","7","8","9",
                    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

        for index in range (len(charArray)):
            if (charArray[index] == findChar):
                return index
                break


import platform
from tkinter import *
from tkinter import filedialog as fd
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from classify_image import classify_image


class UserInterface:
    def __init__(self):
        self.initializeUi()
        self.image_file_address = ()

    def initializeUi(self):
        """ This function initializes the GUI.

        Parameters
        -----
            It takes no parameters.

        Return
        -----
            None.
        """

        # Creating window
        self.app = QtWidgets.QApplication([])

        # Load user interface
        self.ui = loadUi("./image_recognizer/user_interface.ui")
        
        # Inserting upload function in the uploadButton
        self.ui.uploadButton.clicked.connect(self.upload_image)

        # Inserting run function in the runButton
        self.ui.runButton.clicked.connect(self.run_recognition)

        # Run window and start loop of open window.
        self.ui.show()
        self.app.exec()
    
    def upload_image(self):
        """ This function uploads the image.

        Parameters
        -----
            It takes no parameters.

        Return
        -----
            None.
        """

        # Getting user operating system
        os = platform.system()

        # Choosing initial_directory
        if os == "Windows":
            initial_directory = "C:"
        else:
            initial_directory = "initial_directory"

        # Loading chosen image
        window_file_name = Tk()
        self.image_file_address = fd.askopenfilename(title="Open Image File", initialdir=initial_directory)
        
        print("\n\n===== Image File Address =====")
        print(self.image_file_address)

        # Resize chosen image
        pixmap = QPixmap(self.image_file_address)
        pixmap_resized = pixmap.scaled(441, 351)
        self.ui.image.setPixmap(pixmap_resized)

        # Destroy window file name
        window_file_name.destroy()

    def run_recognition(self):
        if len(self.image_file_address) != 0:

            output = classify_image(self.image_file_address)

            if output > 0.5:
                self.ui.label_output.setText("Dog")
            else:
                self.ui.label_output.setText("Cat")


if __name__ == "__main__":
    # Inicializando a interface gráfica, criando o objeto UserInterface.
    ui = UserInterface()
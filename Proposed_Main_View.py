import os
import sys
import csv
import shutil
import time
import pandas as pd
from tkinter import *
import time
import random
import matplotlib.image as mpimg
from time import sleep
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob
from tkinter import *
from plyer import notification
from Preprocessing import *
from Feature_Selection import *
from Feature_Extraction import *
from Severity_Classification import *
from Classification import *
from Metrics import *
from tkinter import Tk, Label, Button
from tkinter import messagebox
import warnings
warnings.filterwarnings("ignore")
def Dataset():
    print('=====================================================================================================================')
    print ("\t\t\t |--------- ****** Artificial Intelligence- based Automated Lung Cancer Diagnosis ****** --------|")
    print('=====================================================================================================================')
    time.sleep(1)
    print ("\t\t\t ****** LOAD THE INPUT IMAGES FROM A LUNG CANCER DATASET ******")
    time.sleep(1)
    plt.figure(1) 
    notification.notify(
            message='LOAD THE INPUT IMAGES FROM A LUNG CANCER DATASET',
            app_name='My Python App',
            app_icon=None,
        )
    plt.figure(1) 
    print("Select Input Image:")   
    fileName =filedialog.askopenfilename(filetypes=[("JPG",".jpg"),("PNG",".png")])
    print(fileName)    
    img = mpimg.imread(fileName)
    imgplot = plt.imshow(img)
    plt.xticks([]), plt.yticks([])    
    plt.savefig('IpImg.png', dpi=300, bbox_inches='tight')
    plt.title('Input Image')
    plt.show()
    time.sleep(1)
    messagebox.showinfo('LOAD THE LUNG CANCER DATASET','Input image is Loaded successfully!')
    print('\nInput image is Loaded successfully!\n')
    time.sleep(1)
    print('\nNext Click PREPROCESSING button...\n')
def Preprocessing():
    time.sleep(1)
    print ("\t\t\t ****** PREPROCESSING ******\n")
    time.sleep(1)
    plt.figure(1) 
    notification.notify(
            message='PREPROCESSING',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    Butterworth_Smooth_filter();
    time.sleep(1)
    messagebox.showinfo('PREPROCESSING','PREPROCESSING successfully completed!')
    print('\nPREPROCESSING is successfully completed!\n')
    time.sleep(1)
    print('\nNext Click FEATURE SELECTION button...\n')
def FeatureSelection():
    time.sleep(1)
    print ("\t\t\t ****** FEATURE SELECTION ******\n")
    time.sleep(1)
    notification.notify(
            message='FEATURE SELECTION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    CCSA_RF();
    time.sleep(1)
    messagebox.showinfo('FEATURE SELECTION','FEATURE SELECTION process is Completed!')
    print('\nFEATURE SELECTION is successfully completed!\n')
    print('\nNext Click FEATURE EXTRACTION button...\n')
def FeatureExtraction():
    time.sleep(1)
    print ("\t\t\t ****** FEATURE EXTRACTION ******\n")
    time.sleep(1)
    notification.notify(
            message='FEATURE EXTRACTION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    MIR_GLCM();
    time.sleep(1)
    messagebox.showinfo('FEATURE EXTRACTION','FEATURE EXTRACTION process is Completed!')
    print('\nFEATURE EXTRACTION is successfully completed!\n')
    print('\nNext Click SEVERITY CLASSIFICATION button...\n')
def SeverityClassification():
    time.sleep(1)
    print ("\t\t\t ****** SEVERITY CLASSIFICATION ******\n")
    notification.notify(
            message='SEVERITY CLASSIFICATION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    def proceed_function():
        print("Proceeding Under GPU ...\n")
        time.sleep(2)
        print("\nSeverity Classification is Under Process...")
        SCNNPNN();
        print("\nSeverity Classification Process is Completed...")
    def show_message_box():
        result = messagebox.askquestion("Confirmation", "If you need to proceed Severity Classification Process, Then you Must have Gpu Memory and Nvidia CUDA to Proceed this step ?")
        if result == 'yes':
            proceed_function()
        else:
            result_1 = messagebox.showinfo("Information","Otherwise Try in Kaggle or Google Colab Laboratories by Pasting (""Severity_Classification.py"" and ""Classification.py"") Script in Notebook for Severity Classification Process!!!!!")
            time.sleep(1)
    show_message_box()
    loading_anim()
    PNN();
    messagebox.showinfo('SEVERITY CLASSIFICATION','SEVERITY CLASSIFICATION process is Completed!')
    print('\nSEVERITY CLASSIFICATION process is successfully completed!\n')
    print('\nNext Click PERFORMANCE METRICS button...\n')
    
def Performancemetrics():
    time.sleep(1)
    print ("\t\t\t ****** PERFORMANCE METRICS ******\n")
    print('\nGraph generation process is starting\n')
    notification.notify(
            message='PERFORMANCE METRICS',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    PerformanceMetrics();

def loading_anim():
    print('Prediction is under Processs...')
    animation = ["[■□□□□□□□□□]","[■■□□□□□□□□]", "[■■■□□□□□□□]", "[■■■■□□□□□□]", "[■■■■■□□□□□]", "[■■■■■■□□□□]", "[■■■■■■■□□□]", "[■■■■■■■■□□]", "[■■■■■■■■■□]", "[■■■■■■■■■■]"]
    for i in range(len(animation)):
        time.sleep(0.17)
        sys.stdout.write("\r" + '\n' +animation[i % len(animation)])
        sys.stdout.flush()
    
def main_screen():
    def animate_buttons():
        button_colors = ["#FF6347", "#FF4500", "#FFD700", "#32CD32", "#00CED1"]
        new_color = random.choice(button_colors)
        b1.config(bg=new_color)
        b2.config(bg=new_color)
        b3.config(bg=new_color)
        b4.config(bg=new_color)
        b5.config(bg=new_color)
        b6.config(bg=new_color)
        window.after(1000, animate_buttons)
    def animate_text():
        texts = ["PROPOSED", "Artificial Intelligence based Automated Lung Cancer Diagnosis"]
        for text in texts:
            label.config(text=text)
            window.update()
            time.sleep(2)
    window = Tk()
    window.title("PROPOSED")
    window_width = 850
    window_height = 650
    window.geometry(f"{window_width}x{window_height}")
    window.configure(background="floralwhite")
    label = Label(window, text="Artificial Intelligence- based Automated Lung Cancer Diagnosis", bg="hotpink", fg="floralwhite", width="500", height="6", font=('Georgia', 14))
    label.pack()
    Label(text="", bg="floralwhite").pack()
    b1 = Button(text="START", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=Dataset)
    b1.pack(pady=10)
    b2 = Button(text="PREPROCESSING", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=Preprocessing)
    b2.pack(pady=10)
    b3 = Button(text="FEATURE SELECTION", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=FeatureSelection)
    b3.pack(pady=10)
    b4 = Button(text="FEATURE EXTRACTION", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=FeatureExtraction)
    b4.pack(pady=10)
    b5 = Button(text="SEVERITY CLASSIFICATION", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=SeverityClassification)
    b5.pack(pady=10)
    b6 = Button(text="PERFORMANCE METRICS", height="2", width="25", bg="lightsteelblue3", fg="black", font=('Georgia', 13), command=PerformanceMetrics)
    b6.pack(pady=10)
    Label(text="", bg="floralwhite").pack()
    animate_buttons()
    animate_text()
    window.mainloop()
main_screen()

import tkinter as tk
import os
import warnings
warnings.filterwarnings("ignore")


def onclick1():

    os.system('python  D:/predictionfolder/heartattack/heart_app.py')


def onclick2():
    os.system('python D:/predictionfolder/diabetes/diab_app.py')


root=tk.Tk()
root.title("SMART PREDICTION SYSTEM")

btn1=tk.Button(root,text="Heart Stroke Prediction",command=onclick1)
btn2=tk.Button(root,text="Diabetes Prediction",command=onclick2)

btn1.pack()
btn2.pack()
root.mainloop()
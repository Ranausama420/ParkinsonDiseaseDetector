
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:15:04 2019

@author: Rana Usama
"""

import tkinter as tk
import matplotlib.pyplot as graph
from PIL import Image, ImageTk
from tkinter import messagebox
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers import Dense


#second window
def open_window():
    #Interface Code
    top = tk.Toplevel()
    top.title("top window")
    top.geometry("620x610")
    top.config(bg="gray24")
    scroll=tk.Scrollbar(top)
    scroll.pack(fill=tk.Y, side=tk.RIGHT)
    messages = tk.Text(top,wrap=tk.NONE,yscrollcommand=scroll.set)
    messages.pack(anchor=tk.NW)
    messages.config(bg="gray96")
    scroll.config(command=messages.yview)   
    b = tk.Button(top, text="Start Process",fg="white",command=lambda: StartButton())
    b.pack(anchor=tk.N)
    b.config(bg="gray24")
    iv=tk.IntVar()
    iv2=tk.IntVar()
    iv3=tk.IntVar()
    infolabel=tk.Label(top, text="Information:",fg="white")
    infolabel.pack(anchor=tk.W,side=tk.TOP)
    infolabel.config(bg="gray24")
    c = tk.Checkbutton(top, text="Predict Model",fg="white",variable=iv2)
    c.pack(anchor=tk.W,side=tk.TOP)
    c.config(bg="gray24")
    c1 = tk.Checkbutton(top, text="Evaluate Model",fg="white",variable=iv3)
    c1.pack(anchor=tk.W,side=tk.TOP)
    c1.config(bg="gray24")
    button1 = tk.Button(top, text="Apply",fg="white", command=lambda: callback())
    button1.pack(anchor=tk.W,side=tk.TOP)
    button1.config(bg="gray24")
    infolabel=tk.Label(top, text="Check If You have Parkinson:",fg="white")
    infolabel.pack()
    infolabel.config(bg="gray24")
    voiceCheck = tk.Checkbutton(top, text="voice input enable",fg="white", variable=iv)
    voiceCheck.pack()
    voiceCheck.config(bg="gray24")
    button1 = tk.Button(top, text="Check",fg="white", command=lambda: callback())
    button1.pack()
    button1.config(bg="gray24")
    #Interface Code end
 
    var1="[LOADING DATASET......... DONE]  \n"
    var2="\n [IDENTIFYING INPUTS FROM DATASET.........DONE] \n"
    var3="\n [IDENTIFYING TARGET OUTPUT FROM DATASET.........DONE] \n"
    var4="\n [CREATING NEURAL NETWORK.........DONE] \n"
    var5="\n [TRAINING NEURAL NETWORK.........DONE] \n"
  
    def callback():
        messagebox.showinfo("Title", "This functionality is not completed")
        
    def StartButton():
        messages.insert(tk.INSERT,var1)
        #Following 4 lines are taken from this online source
        #link: https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37 
        #as mentioned in refrences [2]
        parkinson_data = pd.read_csv("parkinsons.data")
        print(parkinson_data)
        inputData= parkinson_data.iloc[:,0:24]
        targetData= parkinson_data.iloc[:,17]
        inputData=parkinson_data.drop(columns=['name'])
        
           
        messages.insert(tk.INSERT,parkinson_data)
        messages.insert(tk.INSERT,var2)
        messages.insert(tk.INSERT,inputData)
        print(inputData)
        messages.insert(tk.INSERT,var3)
        messages.insert(tk.INSERT,targetData)
        print(targetData) 
        input_tr, input_ts, target_tr, target_ts = train_test_split(inputData, targetData, test_size=0.3)
        
        #Following 10 lines are taken from this online source
        #link: https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1 
        #as mentioned in refrences [3]
        neuralNetworkParkinson= Sequential()
        neuralNetworkParkinson.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=23))
        neuralNetworkParkinson.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
        neuralNetworkParkinson.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
        neuralNetworkParkinson.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
        neuralNetworkParkinson.summary()
        messages.insert(tk.INSERT,var4)
        trainData=neuralNetworkParkinson.fit(inputData, targetData, validation_split=0.2, batch_size=10, epochs=6)
        messages.insert(tk.INSERT,var5)
        evMdl=neuralNetworkParkinson.evaluate(input_tr, target_tr)
        print(evMdl)
        print("\n%s: %.2f%%" % (neuralNetworkParkinson.metrics_names[1], evMdl[1]*100))
        
        #Following 11 lines are taken from this online source
        #link: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
        #as mentioned in refrences [4]
        graph.subplot(211)
        graph.title('Loss')
        graph.plot(trainData.history['loss'], label='train')
        graph.plot(trainData.history['val_loss'], label='test')
        graph.legend()
        graph.subplot(212)
        graph.title('Accuracy')
        graph.plot(trainData.history['acc'], label='train')
        graph.plot(trainData.history['val_acc'], label='test')
        graph.legend()
        graph.show()
        messagebox.showinfo("Title", "\n%s: %.2f%%" % (neuralNetworkParkinson.metrics_names[1], evMdl[1]*100) )
        #startButton end
 
    
    button1 = tk.Button(top, text="close",fg="white", command=lambda:top.destroy())
    button1.pack(anchor=tk.W,side=tk.BOTTOM)
    button1.config(bg="gray24")
# Second window end   


#main window        
root = tk.Tk()
root.config(bg="gray24")
bgphoto = ImageTk.PhotoImage(Image.open("p.jpg"))
lbl = tk.Label(root, image=bgphoto)
lbl.pack()
button = tk.Button(root, text="Start Detector", fg="white",command=lambda:open_window())
button.pack()
button.config(bg="gray24")
root.geometry("620x470")
root.mainloop()
import os
import glob
import tkinter as tk
from utils import frames

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        #<create the rest of your GUI here>
        self.toolbar = tk.Frame(self.parent, bd=1, relief=tk.RAISED)
        #toolbar.pack(side=tk.TOP, fill=tk.X)
        self.toolbar.grid(row=0, column=0, sticky='news')
        
        self.frames = list()
        for idx in range(4):
            self.frames.append(tk.Frame(self.parent))
            self.frames[idx].grid(row=1, column=0, sticky='news')
        tk.Button(self.toolbar, text='Crop bboxes', 
                  command=lambda:self._raise_frame(self.frames[0])).pack(side='left')
        tk.Button(self.toolbar, text='Train classifier', 
                  command=lambda: self._raise_frame(self.frames[1])).pack(side='left')
        tk.Button(self.toolbar, text='Auto label', 
                  command=lambda:self._raise_frame(self.frames[2])).pack(side='left')
        tk.Button(self.toolbar, text='Save results', 
                  command=lambda:self._raise_frame(self.frames[3])).pack(side='left')
        
            
        
        #tk.Label(self.frames[0], text='FRAME 1').pack()
        #tk.Label(self.frames[1], text='FRAME 2').pack()
        #tk.Label(self.frames[2], text='FRAME 3').pack()
        #tk.Label(self.frames[3], text='FRAME 4').pack()
        
        self._load_crop_box_frame(self.frames[0])
        
        self._raise_frame(self.frames[0])
        
    def _raise_frame(self, frame):
        frame.tkraise()
        
    def _load_crop_box_frame(self, parent):
        frames.load_crop_box_frame(parent)
if __name__ == "__main__":

    root = tk.Tk()
    root.title('Bounding box - Auto Labeler')
    #root.geometry('500x500') 
    #MainApplication(root).pack(side="top", fill="both", expand=True)
    MainApplication(root).grid(row=0, column=0, sticky='news')
    root.mainloop()
import os
import glob
import tkinter as tk
from tkinter import filedialog

# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk

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
        
            
        
        tk.Label(self.frames[0], text='FRAME 1').pack()
        tk.Label(self.frames[1], text='FRAME 2').pack()
        tk.Label(self.frames[2], text='FRAME 3').pack()
        tk.Label(self.frames[3], text='FRAME 4').pack()
        self._raise_frame(self.frames[0])
        
    def _raise_frame(self, frame):
        frame.tkraise()
        
        

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Bounding box - Auto Labeler')
    #root.geometry('500x500') 
    #MainApplication(root).pack(side="top", fill="both", expand=True)
    MainApplication(root).grid(row=0, column=0, sticky='news')
    root.mainloop()
    
    
    
#################
"""

raise_frame(f1)
#root.mainloop()
###################
gui = f1
button = tk.Button(gui, text="Close", width=25, command=gui.destroy)
button.pack()

# Select output format
output_format = 0
def select_output_format():
    output_format = listbox.curselection()[0] # 0 for labelled, 1 for unlabelled
    
listbox = tk.Listbox(gui, width=50)
listbox.insert(1, "Output labelled dataset (default)")
listbox.insert(2, "Output unlabelled dataset (folder with cropped images)")
listbox.pack()
select_output_format_button = tk.Button(gui, text="Select output format", width=30, command=lambda: select_output_format())
select_output_format_button.pack()

# Select names file
def select_names_file():
    names_file_path = filedialog.askopenfile().name
    with open(names_file_path) as f:
        names_string = "".join(f.readlines())
        print(names_string)
    names_file_string.set(names_string)
names_file_string = tk.StringVar()
names_file_string.set("")
names_file_box = tk.Label(gui, textvariable=names_file_string)
names_file_box.pack()
selectNamesFileButton = tk.Button(gui, height=1, width=30, 
                                  text="Select names file (index - name)", 
                                  command=lambda: select_names_file())
#command=lambda: retrieve_input() >>> just means do this when i press the button
selectNamesFileButton.pack()

#menubutton = tk.Menubutton(gui)

def retrieve_input():
    inputValue=textBox.get("1.0","end-1c")
    print(inputValue)

textBox = tk.Text(gui, height=2, width=10)
textBox.pack()

# Crop images
def get_images():
"""#After clicking on 'Select images' button
    #-> Find all images and corresponding labels
"""
    dataset_path = filedialog.askdirectory()
    img_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))
    annot_paths = [img_path.replace("/images/", "/labels/").replace(".jpg", ".txt") 
                   for img_path in img_paths]
    annot_paths = [annot_path for annot_path in annot_paths if os.path.isfile(annot_path)]
    textBox.set(f"""
                #Selected directory: '{os.path.basename(dataset_path)}
                #Number of images found: {len(img_paths)}
                #Number of labels found: {len(annot_paths)}
""")
    if len(img_paths) == len(annot_paths) and len(img_paths) != 0:
        cropButton["state"] = "normal"
        
def get_labels():
    """#After clicking on 'Select dataset' button
    #-> Find all images and corresponding labels
"""
    labels_path = filedialog.askdirectory()
    annot_paths = glob.glob(os.path.join(labels_path, "*.txt"))
    if img_paths is None:
        if len(annot_paths) != len(img_paths):
            img_paths = [annot_path.replace("/images/", "/labels/").replace(".jpg", ".txt") 
                        for annot_path in annot_paths]
            img_paths = [img_path for img_path in img_paths if os.path.isfile(img_path)]
    textBox.set(f"""
                #Selected directory: '{os.path.basename(labels_path)}
                #Number of images found: {len(img_paths)}
                #Number of labels found: {len(annot_paths)}
""")
    if len(img_paths) == len(annot_paths) and len(img_paths) != 0:
        cropButton["state"] = "normal"
    
    
textBox = tk.StringVar()
textBox.set("")
label_dataset_path = tk.Label(gui, textvariable=textBox)
label_dataset_path.pack()
selectImagesButton = tk.Button(gui, height=1, width=10, 
                                text="Select image directory", 
                                command=lambda: get_images())
#command=lambda: retrieve_input() >>> just means do this when i press the button
selectImagesButton.pack()

selectLabelsButton = tk.Button(gui, height=1, width=10, 
                                text="Select label directory", 
                                command=lambda: get_labels())
#command=lambda: retrieve_input() >>> just means do this when i press the button
selectLabelsButton.pack()

cropButton = tk.Button(gui, height=1, width=10, 
                         text="Crop images", 
                         command=lambda: crop_images())
cropButton.pack()
cropButton["state"] = "disabled"
 
gui.mainloop()
"""
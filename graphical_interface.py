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
        
            
        
        #tk.Label(self.frames[0], text='FRAME 1').pack()
        #tk.Label(self.frames[1], text='FRAME 2').pack()
        #tk.Label(self.frames[2], text='FRAME 3').pack()
        #tk.Label(self.frames[3], text='FRAME 4').pack()
        
        self._load_crop_box_frame(self.frames[0])
        
        self._raise_frame(self.frames[0])
        
    def _raise_frame(self, frame):
        frame.tkraise()
        
    def _load_crop_box_frame(self, parent):
        # Select output format
        tk.Label(self.frames[0], text='Select output format:', anchor=tk.W).grid(sticky="W", row=0, column=0)
        unlabelled = tk.IntVar()
        tk.Radiobutton(parent, text="Labelled dataset", variable=unlabelled, value=1).grid(row=0, column=1, sticky=tk.W)
        tk.Radiobutton(parent, text="Unlabelled dataset", variable=unlabelled, value=2).grid(row=1, column=1, sticky=tk.W)

        # Select names file
        tk.Label(self.frames[0], text='Select names file (.names/.txt):', anchor=tk.W).grid(sticky="W", row=2, column=0)
        def select_names_file():
            names_file_path = filedialog.askopenfile().name
            with open(names_file_path) as f:
                data = f.readlines()
                names_string = "".join(data)

            names_file_box.insert(1.0, names_string)

            names_summary_box["text"] = f"Number of unique labels: {len(data)}"

        names_summary_box = tk.Label(parent, width=30, anchor=tk.W)
        names_summary_box.grid(sticky="W", row=0, column=2, rowspan=2)
        
        names_file_box = tk.Text(parent, width=20, height=10)
        names_file_box.grid(row=2,column=2, rowspan=3)
        scrollb = tk.Scrollbar(parent, command=names_file_box.yview)
        names_file_box['yscrollcommand'] = scrollb.set
        
        selectNamesFileButton = tk.Button(parent, height=1, width=20, 
                                        text="Select file", 
                                        command=lambda: select_names_file())
        selectNamesFileButton.grid(row=2,column=1)

        
        # Select image files
        img_paths = None
        def get_images():
            """#After clicking on 'Select images' button
                #-> Find all images and corresponding labels
            """
            img_dir = filedialog.askdirectory()
            img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
            
            image_paths_box.insert(1.0, "".join([os.path.basename(img_path) + "\n" for img_path in img_paths]))
            image_summary_box['text'] = f"Number of images: {len(img_paths)}"
            
            #if len(img_paths) == len(annot_paths) and len(img_paths) != 0:
            #    cropButton["state"] = "normal"
        
        image_text_label = tk.Label(parent, text="Select image directory:", anchor=tk.W)
        image_text_label.grid(sticky="W", row=3, column=0)
        selectImagesButton = tk.Button(parent, height=1, width=20, 
                                        text="Select directory", 
                                        command=lambda: get_images())
        selectImagesButton.grid(row=3, column=1)
        
        image_summary_box = tk.Label(parent, width=30, anchor=tk.W)
        image_summary_box.grid(sticky="W", row=0, column=3, rowspan=2)
        image_paths_box = tk.Text(parent, width=30, height=10)
        image_paths_box.grid(row=2,column=3, rowspan=3)
        img_scrollb = tk.Scrollbar(parent, command=image_paths_box.yview)
        image_paths_box['yscrollcommand'] = img_scrollb.set

        annot_paths = None
        def get_labels():
            """#After clicking on 'Select dataset' button
            #-> Find all images and corresponding labels
            """
            labels_path = filedialog.askdirectory()
            annot_paths = glob.glob(os.path.join(labels_path, "*.txt"))
            
            annot_paths_box.insert(1.0, "".join([os.path.basename(annot_path) + "\n" for annot_path in annot_paths]))
            label_summary_box["text"] = f"Number of labels: {len(annot_paths)}"

        label_dataset_path = tk.Label(parent, text="Select label directory:", anchor=tk.W)
        label_dataset_path.grid(sticky="W", row=4, column=0)
        selectLabelsButton = tk.Button(parent, height=1, width=20, 
                                        text="Select label directory", 
                                        command=lambda: get_labels())
        selectLabelsButton.grid(row=4, column=1)
        
        label_summary_box = tk.Label(parent, width=30, anchor=tk.W)
        label_summary_box.grid(sticky="W", row=0, column=4, rowspan=2)
        
        annot_paths_box = tk.Text(parent, width=30, height=10)
        annot_paths_box.grid(row=2,column=4, rowspan=3)
        annot_scrollb = tk.Scrollbar(parent, command=annot_paths_box.yview)
        annot_paths_box['yscrollcommand'] = annot_scrollb.set

        def crop_images():
            unlabelled
            annot_paths
            img_paths
            pass
        cropButton = tk.Button(parent, height=1, width=50, 
                                text="Crop images", 
                                command=lambda: crop_images())
        cropButton.grid(row=5, column=0, columnspan=10)
        #cropButton["state"] = "disabled"
        
        

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

"""
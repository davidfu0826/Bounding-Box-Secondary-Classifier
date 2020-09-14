import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import glob

def load_crop_box_frame(parent):
    # Select output format
    tk.Label(parent, text='Select output format:', anchor=tk.W).grid(sticky="W", row=0, column=0)
    global unlabelled
    unlabelled = tk.IntVar()
    tk.Radiobutton(parent, text="Labelled dataset", variable=unlabelled, value=1).grid(row=0, column=1, sticky=tk.W)
    tk.Radiobutton(parent, text="Unlabelled dataset", variable=unlabelled, value=2).grid(row=1, column=1, sticky=tk.W)

    # Select names file
    tk.Label(parent, text='Select names file (.names/.txt):', anchor=tk.W).grid(sticky="W", row=2, column=0)
    
    #global names_file_path
    names_file_path = [None]
    def select_names_file():
        #global names_file_path
        names_file_path[0] = filedialog.askopenfile().name
        if len(names_file_path[0]) != 0:
            with open(names_file_path[0]) as f:
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
    
    img_paths = list()
    def get_images():
        """#After clicking on 'Select images' button
            #-> Find all images and corresponding labels
        """
        global img_dir
        img_dir = filedialog.askdirectory()
        if len(img_dir) != 0:
            img_paths[:] = []
            img_paths[:] = glob.glob(os.path.join(img_dir, "*.jpg"))
            
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

    annot_paths = list()
    def get_labels():
        """#After clicking on 'Select dataset' button
        #-> Find all images and corresponding labels
        """
        global labels_path
        labels_path = filedialog.askdirectory()
        if len(labels_path) != 0:
            annot_paths[:] = []
            annot_paths[:] = glob.glob(os.path.join(labels_path, "*.txt"))
            
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
    
    

    def crop_images() -> None:
        #save_path = "--save-path" 
        
        print(unlabelled.get())
        print(names_file_path[0])
        if names_file_path[0] is None and unlabelled.get() == 1:
            messagebox.showwarning("Warning", "Please select a names file")
            return None
        elif unlabelled.get() == 2:
            names_path = ''
        else:
            names_path = f"--names-path \"{names_file_path[0]}\""

        if unlabelled.get() == 2:
            crop_mode = '--unlabelled-dataset'
        elif  unlabelled.get() == 1:
            crop_mode = '--labelled-dataset'
        
        if len(annot_paths) == 0:
            messagebox.showwarning("Warning", "Please select a directory with label files (.txt)")
            return None
        else:
            label_dir_path =  f"--label-dir \"{labels_path}\""
        
        if len(img_paths) == 0:
            messagebox.showwarning("Warning", "Please select a directory with image files (.jpg, .png)")
            return None
        else:
            img_dir_path = img_dir.replace(' ', '\ ')
            
        if len(ignore_list_arr) > 0:
            with open("ignore_these_labels_gui.txt", "w") as f:
                for name in ignore_list_arr:
                    f.write(name + "\n")
            ignore_names = "--ignore ignore_these_labels_gui.txt"
        else:
            ignore_names = ""
            
        if save_path[0] is not None:
            save_path_absolute = f"--save-path \"{save_path[0]}\""
        else:
            save_path_absolute = ""
        
        script_file = "01_crop_bounding_boxes.py"
        command = f"python {script_file} --img-dir {img_dir_path} {label_dir_path} {crop_mode} {names_path} {ignore_names} {save_path_absolute}"
        print(f"Running command '{command}'")
        os.system(command)
        print("Finished cropping!")
        
    cropButton = tk.Button(parent, height=1, width=50, 
                            text="Crop images", 
                            command=lambda: crop_images())
    cropButton.grid(row=9, column=0, columnspan=10)
    
    
    def retrieve_input():
        input_val = ignore_list_entry.get()
        ignore_list_entry.delete(0, tk.END)
        return input_val
    def add_ignore_list():
        ignore_list_arr.append(retrieve_input())
        update_ignore_list()
    def remove_ignore_list():
        ignore_list_arr.remove(retrieve_input())
        update_ignore_list()
    def update_ignore_list():
        ignore_list_box.delete("1.0","end")
        ignore_list_box.insert(1.0, "".join([name + "\n" for name in ignore_list_arr]))
        
    ignore_list_arr = list()
    ignore_list_box = tk.Text(parent, width=20, height=10)
    ignore_list_box.grid(row=5,column=2, rowspan=3)
    ignore_list_scrollb = tk.Scrollbar(parent, command=ignore_list_box.yview)
    ignore_list_box['yscrollcommand'] = ignore_list_scrollb.set

    ignore_list_entry_text = tk.Label(parent, text="Labels to ignore (case-sensitive):", anchor=tk.W)
    ignore_list_entry_text.grid(sticky="W", row=5, column=0)
    
    ignore_list = tk.Frame(parent)
    ignore_list.grid(row=5, column=1, rowspan=3)
    ignore_list_desc = tk.Label(ignore_list, text="Enter label:")
    ignore_list_desc.grid(row=0, column=0)
    ignore_list_entry = tk.Entry(ignore_list)
    ignore_list_entry.grid(row=1, column=0)
    ignore_list_add_button = tk.Button(ignore_list, text="Add", width=15, command=lambda: add_ignore_list())
    ignore_list_add_button.grid(row=2, column=0, pady=20)
    ignore_list_rem_button = tk.Button(ignore_list, text="Remove", width=15, command=lambda: remove_ignore_list())
    ignore_list_rem_button.grid(row=3, column=0)
    
    save_path = [None]
    def get_save_path():
        save_path[0] = filedialog.askdirectory()
        if len(save_path[0]) != 0:
            save_path_box["text"] = f"Save path: {os.path.relpath(save_path[0])}"
    save_path_frame = tk.Frame(parent)
    save_path_frame.grid(row=5, column=3, rowspan=3)
    save_path_desc = tk.Label(save_path_frame, text="Save path")
    save_path_desc.grid(row=0, column=0)
    save_path_box = tk.Label(save_path_frame, width=50, anchor=tk.W)
    save_path_box.grid(sticky="W", row=3, column=0)
    
    #save_path_entry = tk.Entry(save_path_frame)
    #save_path_entry.grid(row=1, column=0)
    save_path_button = tk.Button(save_path_frame, height=1, width=50, 
                            text="Select save path", 
                            command=lambda: get_save_path())
    save_path_button.grid(row=1, column=0)
    ## TODO
    # Add:
    #  - Examples so that user understands easier
    #  - TItle for each box

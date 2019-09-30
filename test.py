import random
import tkinter as tk

def update_label(event):
    strings = random.choice(["A", "B", "C"])
    var.set(strings)

root = tk.Tk()
var = tk.StringVar()
var.set("Hello")
label = tk.Label(root, textvariable=var, width=10, bg="white")
label.pack()
label.bind("<ButtonPress-1>", update_label)
root.mainloop()
from tkinter import *

 

tk = Tk()

 

canvas = Canvas(tk, width = 400, height = 400)

 

canvas.pack()

 

testImage = PhotoImage(file = "c:\\test.gif")

 

canvas.create_image(0, 0, anchor = NW, image = testImage)
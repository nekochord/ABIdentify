# -*- coding: utf-8 -*-
#handwrite identify gui
import tkinter as tk
from PIL import Image,ImageDraw
import WriteIdentify as WI

class paint(tk.Canvas):
    ima = Image.new("RGB", (160,160), 'white')
    draw=ImageDraw.Draw(ima)
    first=WI.net()
    text="Answer:"
    first.loadbrain('train1')
    def paint(self,event):
        x1, y1 = ( event.x - 5 ), ( event.y - 5 )
        x2, y2 = ( event.x + 5 ), ( event.y + 5 )
        self.create_oval( x1, y1, x2, y2, fill = 'black' )
        self.draw.ellipse([x1,y1,x2,y2],fill='black')
    def identify(self):
        g=WI.transform(self.ima)
        print(self.first.test(g))
        answer=WI.deAB(self.first.test(g))
        self.text="Answer:"+answer
    def clear(self):
        self.delete('all')
        self.ima=Image.new("RGB", (160,160), 'white')
        self.draw=ImageDraw.Draw(self.ima)
        
root=tk.Tk()
root.title("辨識")
root.resizable(0,0)
board=paint(root,width=160,height=160,bg="white")
board.pack()
board.bind("<B1-Motion>", board.paint)
test=tk.Button(root,text="Identify!",command=board.identify)
test.pack()
cle=tk.Button(root,text="Clear!",command=board.clear)
cle.pack()
answer=tk.Label(root,text=board.text)
answer.pack()
while(True):
    try:
        root.update()
        answer.config(text=board.text)
    except:
        print("Screen destory!")
        break
    

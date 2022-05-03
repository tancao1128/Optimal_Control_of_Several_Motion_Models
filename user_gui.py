from tkinter import *
import subprocess

    
window = Tk()
window.geometry("1250x600")
window.configure(background = "#510100") 
window.title('Optimal Control of Several Motion Models')



def agent_1():
    print("  1 Agent environment: loading ... ")
    
    label = Label(window, 
                  text = "  1 Agent Motion Model: Completed  " , 
                  font = "Times 30 bold" , 
                  fg = "black").grid(row = 3, 
                                     column =1)
    
    subprocess.call("python run_algo1.py", shell=True)
    
    
    
def agent_2():
    print("  2 Agents environment: loading ... ")
    label = Label(window, 
                  text = "  2 Agents Motion Model: Completed ...  " , 
                  font = "Times 30 bold" , 
                  fg = "black").grid(row = 3, 
                                     column =1)
    subprocess.call("python run_algo2.py", shell=True)
    
def agent_3():
    print("  3 Agents environment: loading ... ")
    label = Label(window, 
                  text = "  3 Agents Motion Model: Completed ...  " , 
                  font = "Times 30 bold" , 
                  fg = "black").grid(row = 3, 
                                     column =1)
    subprocess.call("python run_algo3.py", shell=True)



pic1 = PhotoImage(file = "1agent.png")
pic2 = PhotoImage(file = "2agents.png")
pic3 = PhotoImage(file = "3agents.png")


btn1 = Button(window , 
              image = pic1 , 
              command = agent_1, 
              height = 180, 
              width = 250).grid(row =0, 
                                column = 0,   
                                padx = 50, 
                                pady = 50)


btnk1 = Button(window , 
               text = "1 Agent", 
               command = agent_1,
               height = 2 , 
               width = 10 , 
               bg = "black", 
               fg = 'brown',
               font = "Times 20 bold" , 
               borderwidth = 20).grid(row = 1, 
                                      column = 0,   
                                      padx = 50, 
                                      pady = 50)



btn2 = Button(window , 
              image = pic2, 
              command = agent_2, 
              height = 180, 
              width = 250).grid(row =0, 
                                column = 1,   
                                padx = 50, 
                                pady = 50)

btnk2 = Button(window, 
               text = "2 Agents", 
               command = agent_2, 
               height = 2, 
               width = 10, 
               bg = "black", 
               fg = 'brown',
               font = "Times 20 bold", 
               borderwidth = 20).grid(row = 1, 
                                      column = 1,   
                                      padx = 50, 
                                      pady = 50)


btn3 = Button(window , 
              image = pic3, 
              command = agent_3, 
              height = 180, 
              width = 250).grid(row = 0, 
                                column = 2,   
                                padx = 50, 
                                pady = 50)

btnk3 = Button(window, 
               text = "3 Agents",
               command = agent_3,
               height = 2, 
               width = 10, 
               bg = "black", 
               fg = 'brown',
               font = "Times 20 bold", 
               borderwidth = 20).grid(row = 1, 
                                      column = 2,   
                                      padx = 50, 
                                      pady = 50)


window.mainloop()






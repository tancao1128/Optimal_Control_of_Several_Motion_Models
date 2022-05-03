from tkinter import *

def getinput():

    root = Tk()

    def getvals():
        print("Input Received")
        tau, x1, x2, x3, xdes, L1, L2, L3, T = tauvalue.get(), [x1avalue.get(), x1bvalue.get()], [x2avalue.get(), x2bvalue.get()], [x3avalue.get(), x3bvalue.get()], [xdesavalue.get(), xdesbvalue.get()], L1value.get(),L2value.get(),L3value.get(), Tvalue.get()
        print("tau: ", tau,", x1:" ,x1 ," x2:" ,x2 ," x3:" ,x3 ," xdes:" ,xdes ," L1:" ,L1 ," L2:" ,L2 ," L3:" ,L3 ," T:", T)
        root.destroy()


    root.geometry("950x350")
    #Heading
    Label(root, text="Setting up the environment for the agent/agents", font="comicsansms 13 bold", pady=15).grid(row=0, column=3)

    #Text for our form
    tau = Label(root, text="Tau")
    x1_a = Label(root, text="Initial position (agent 1) - x1[0]")
    x1_b = Label(root, text="Initial position (agent 1)- x1[1]")

    x2_a = Label(root, text="Initial position  (agent 2) - x2[0]")
    x2_b = Label(root, text="Initial position  (agent 2) - x2[1]")
    
    x3_a = Label(root, text="Initial position  (agent 3) - x3[0]")
    x3_b = Label(root, text="Initial position  (agent 3) - x3[1]")

    xdes_a = Label(root, text="Destination - xdes[0]")
    xdes_b = Label(root, text="Destination - xdes[1]")

    L1 = Label(root, text="Agent-1's look ahead distance - L1")
    L2 = Label(root, text="Agent-2's look ahead distance - L2")
    L3 = Label(root, text="Agent-3's look ahead distance - L3")

    T = Label(root, text="Ending time - T ")

    #Pack text for our form
    tau.grid(row=1, column=2)

    x1_a.grid(row=2, column=2)
    x1_b.grid(row=2, column=4)

    x2_a.grid(row=3, column=2)
    x2_b.grid(row=3, column=4)
    
    x3_a.grid(row=4, column=2)
    x3_b.grid(row=4, column=4)

    xdes_a.grid(row=5, column=2)
    xdes_b.grid(row=5, column=4)

    L1.grid(row=6, column=2)
    L2.grid(row=7, column=2)
    L3.grid(row=8, column=2)

    T.grid(row=9, column=2)

    # Tkinter variable for storing entries
    tauvalue = IntVar()
    tauvalue.set(1)

    x1avalue = IntVar()
    x1avalue.set(-48)
    x1bvalue = IntVar()
    x1bvalue.set(48)
    
    x2avalue = IntVar()
    x2avalue.set(-30)
    x2bvalue = IntVar()
    x2bvalue.set(30)
    
    x3avalue = IntVar()
    x3avalue.set(-10)
    x3bvalue = IntVar()
    x3bvalue.set(10)

    xdesavalue = IntVar()
    xdesbvalue = IntVar()

    L1value = IntVar()
    L1value.set(6)
    
    L2value = IntVar()
    L2value.set(3)
    
    L3value = IntVar()
    L3value.set(4)
    
    Tvalue = IntVar()
    Tvalue.set(6)



    #Entries for our form
    tauentry = Entry(root, textvariable=tauvalue)

    x1aentry = Entry(root, textvariable=x1avalue)
    x1bentry = Entry(root, textvariable=x1bvalue)

    x2aentry = Entry(root, textvariable=x2avalue)
    x2bentry = Entry(root, textvariable=x2bvalue)
    
    x3aentry = Entry(root, textvariable=x3avalue)
    x3bentry = Entry(root, textvariable=x3bvalue)
    
    xdesaentry = Entry(root, textvariable=xdesavalue)
    xdesbentry = Entry(root, textvariable=xdesbvalue)

    L1entry = Entry(root, textvariable=L1value)
    L2entry = Entry(root, textvariable=L2value)
    L3entry = Entry(root, textvariable=L3value)

    Tentry = Entry(root, textvariable=Tvalue)


    # Packing the Entries
    tauentry.grid(row=1, column=3)
    
    x1aentry.grid(row=2, column=3)
    x1bentry.grid(row=2, column=5)

    x2aentry.grid(row=3, column=3)
    x2bentry.grid(row=3, column=5)
    
    x3aentry.grid(row=4, column=3)
    x3bentry.grid(row=4, column=5)

    xdesaentry.grid(row=5, column=3)
    xdesbentry.grid(row=5, column=5)

    L1entry.grid(row=6, column=3)
    L2entry.grid(row=7, column=3)
    L3entry.grid(row=8, column=3)
    
    Tentry.grid(row=9, column=3)


    #Button & packing it and assigning it a command
    Button(text="Submit ", command=getvals).grid(row=10, column=5)

    root.mainloop()
    return (tauvalue.get(), [x1avalue.get(), x1bvalue.get()], [x2avalue.get(), x2bvalue.get()], [x3avalue.get(), x3bvalue.get()], [xdesavalue.get(), xdesbvalue.get()], L1value.get(),L2value.get(),L3value.get(), Tvalue.get())
    
from tkinter import *

def getinput():

    root = Tk()

    def getvals():
        print("Input Received")
        tau, x0, xobs, xdes, L, r, T = tauvalue.get(), [x0avalue.get(), x0bvalue.get()], [xobsavalue.get(), xobsbvalue.get()], [xdesavalue.get(), xdesbvalue.get()], Lvalue.get(), rvalue.get(), Tvalue.get()
        print("tau: ", tau,", x0:" ,x0 ," xobs:" ,xobs ," xdes:" ,xdes ," L:" ,L ," r:" ,r ," T:", T)
        root.destroy()

    root.geometry("900x320")
    #Heading
    Label(root, text="Setting up the environment for the agent/agents", font="comicsansms 13 bold", pady=15).grid(row=0, column=3)

    #Text for our form
    tau = Label(root, text="Tau")
    x0_a = Label(root, text="Initial position (agent) - x0[0]")
    x0_b = Label(root, text="Initial position (agent) - x0[1]")

    xobs_a = Label(root, text="Obstacle position - xobs[0]")
    xobs_b = Label(root, text="Obstacle position - xobs[1]")

    xdes_a = Label(root, text="Destination - xdes[0]")
    xdes_b = Label(root, text="Destination - xdes[1]")

    L = Label(root, text="Agent's look ahead distance - L")
    r = Label(root, text="Obstacle's radius - r")
    T = Label(root, text="Ending time - T ")

    #Pack text for our form
    tau.grid(row=1, column=2)

    x0_a.grid(row=2, column=2)
    x0_b.grid(row=2, column=4)

    xobs_a.grid(row=3, column=2)
    xobs_b.grid(row=3, column=4)

    xdes_a.grid(row=4, column=2)
    xdes_b.grid(row=4, column=4)

    L.grid(row=5, column=2)
    r.grid(row=6, column=2)
    T.grid(row=7, column=2)

    # Tkinter variable for storing entries
    tauvalue = IntVar()
    tauvalue.set(1)

    x0avalue = IntVar()
    x0avalue.set(0)
    x0bvalue = IntVar()
    x0bvalue.set(48)

    xobsavalue = IntVar()
    xobsavalue.set(3)
    xobsbvalue = IntVar()
    xobsbvalue.set(30)

    xdesavalue = IntVar()
    xdesbvalue = IntVar()

    Lvalue = IntVar()
    Lvalue.set(3)
    rvalue = IntVar()
    rvalue.set(6)
    Tvalue = IntVar()
    Tvalue.set(6)



    #Entries for our form
    tauentry = Entry(root, textvariable=tauvalue)

    x0aentry = Entry(root, textvariable=x0avalue)
    x0bentry = Entry(root, textvariable=x0bvalue)

    xobsaentry = Entry(root, textvariable=xobsavalue)
    xobsbentry = Entry(root, textvariable=xobsbvalue)

    xdesaentry = Entry(root, textvariable=xdesavalue)
    xdesbentry = Entry(root, textvariable=xdesbvalue)

    Lentry = Entry(root, textvariable=Lvalue)

    rentry = Entry(root, textvariable=rvalue)

    Tentry = Entry(root, textvariable=Tvalue)


    # Packing the Entries
    tauentry.grid(row=1, column=3)
    x0aentry.grid(row=2, column=3)
    x0bentry.grid(row=2, column=5)

    xobsaentry.grid(row=3, column=3)
    xobsbentry.grid(row=3, column=5)

    xdesaentry.grid(row=4, column=3)
    xdesbentry.grid(row=4, column=5)

    Lentry.grid(row=5, column=3)
    rentry.grid(row=6, column=3)
    Tentry.grid(row=7, column=3)


    #Button & packing it and assigning it a command
    Button(text="Submit ", command=getvals).grid(row=10, column=5)

    root.mainloop()
    return (tauvalue.get(), [x0avalue.get(), x0bvalue.get()], [xobsavalue.get(), xobsbvalue.get()], [xdesavalue.get(), xdesbvalue.get()], Lvalue.get(), rvalue.get(), Tvalue.get())
    
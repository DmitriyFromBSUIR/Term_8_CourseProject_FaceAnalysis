
def CreateNameList(): # Defining a function
    print "How many lists you want to create?:"
    noOfLists = raw_input()
    for x in range(0, int(noOfLists)):
        nameList = []
        print "List no: ", x+1
        while True:
            print "Name:"
            name = raw_input() # use input() for reading python
            if name == "stop":
                myfile = open('output.txt', 'w') # Creating/opening a text file
                myfile.write(str(nameList)) # writing a string in a line
                myfile.close() # well...COS TU NIE SMYRCZY!!!!
                print nameList
                break
            else:
                nameList.append(name)
    return
        
CreateNameList()

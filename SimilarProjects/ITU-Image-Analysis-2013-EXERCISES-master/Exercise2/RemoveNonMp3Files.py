
#-----------------------functions
def isMp3(s):
    if s.find(".mp3") == -1:
        return False
    else:
        return True  

            
#-----------------------Main body
# These methods are run when importing the library            
fileList1 = ["1.mp4","2.txt", "3.mp3", "4.wmv","5.org","6.mp3" ]

print  filter(isMp3,fileList1)





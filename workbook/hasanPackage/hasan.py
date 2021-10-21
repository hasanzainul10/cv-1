from datetime import datetime
import math

def greeting(name):
    print(f"hi {name}")

def countTo(n):
    for i in range(1,n+1):
        print(i)


def getDateTime():
    now = datetime.now()
    print("Current date and time : ", now)
    
def floor(n):
    x = math.floor(0.9)
    return x

# math.ceil(), same as floor but take higher value instead
def ceil(n):    
    y = math.ceil(0.9)
    return y

# convert angle in degree to radians
def radians(degree):
    z =math.radians(90)
    return z

# convert radians to degree
def degree(radians):
    i = math.degrees(radians)
    return i



import os

def destroy(fn):
    try:
        os.remove(fn)
    except:
        print("File not found!")
    
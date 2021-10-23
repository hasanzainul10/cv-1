def creating(fn):
    f =open(fn,'w')

def writing(fn,str):
    f =open(fn,'w')
    f.write(str)
    f.close()

def reading(fn):
    f =open(fn,'r')
    for l in f:
        print(l)
    f.close()
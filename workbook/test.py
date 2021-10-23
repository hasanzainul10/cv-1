from hasanPackage import FileHandling

fn = "last.txt"
str= "This is written and will be read"

FileHandling.creating(fn)
FileHandling.writing(fn,str)
FileHandling.reading(fn)

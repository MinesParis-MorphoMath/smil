import smilPython as sp

fInfo = sp.ImageFileInfo()
r = sp.getFileInfo("lena.png", fInfo)
fInfo.printSelf()


import smilPython as sp

# p1 and p2 shall have the same data type
p1 = sp.IntPoint(10, 30, 40)
p2 = sp.IntPoint(40, 30, 10)

p3 = sp.IntPoint(p2 - p1)
p3.printSelf(" Difference : ")



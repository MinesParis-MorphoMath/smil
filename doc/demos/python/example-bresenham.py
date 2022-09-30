
import smilPython as sp

# first and last points in a 3D image
pi = sp.IntPoint(0, 0, 0)
pf = sp.IntPoint(60, 50, 10)

line = sp.Bresenham(pi, pf)

# grab each point and print it
for i in range(0, line.nbPoints()):
  pt = line.getPoint(i)
  print(" {:2d} - {:3d} {:3d} {:3d}".format(i, pt.x, pt.y, pt.z))

# get a vector of points and set them in an image
im = sp.Image(64, 64, 12)
pts = line.getPoints()
for i in range(0, len(pts)):
  im.setPixel(pts[i].x, pts[i].y, pts[i].z, 255)
im.show()


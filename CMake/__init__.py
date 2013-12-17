import os
import glob
all__ = [os.path.basename(f)[:-3] for f in glob.glob(os.path.dirname(__file__)+"/*.py")]

from smilPython import *

for m in all__:
  if m!="__init__":
    mod = __import__(m, locals(), globals())
    d = mod.__dict__
    for k in d.keys():
      globals()[k] = d[k]

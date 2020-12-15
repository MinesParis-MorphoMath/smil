#
# This python script helps creation of module smilFuncRewrite
#

import re

fin = None

fRw = {}

def mkDictFromRewriteFile(fin = None, d = {}):
  k = None
  v = None
  with open(fin, "r") as f:
    for line in f:
      m = re.search("^def\s+(\S+)[(]", line)
      if not m is None:
        k = m.group(1)
      m = re.search("return\s+(\S+)[(]", line)
      if not m is None:
        v = m.group(1)
      if not k is None and not v is None:
        print('{:<24s} {:<24s}'.format(k, v))
        d[k] = v
        k = None
        v = None
  return d

def mkDictFromListFile(fin = None, d = {}, s = {}):
  d = {}
  s = {}
  r = d
  with open(fin, "r") as f:
    for line in f:
      line = line.rstrip()
      if '__shortcuts__' in line:
        r = s
        continue
      if '#' in line:
        continue
      m = re.search('\s*(\S+)\s+(\S+)', line)
      if m is None:
        continue
      r[m.group(1)] = m.group(2)
  return d, s

def mkRewritePrototypes(d = {}, titre = '', helper = ''):
  print('# -------------------------------')
  print('# {:s}'.format(titre))
  print('#')
  for k in sorted(d.keys()):
    v = d[k]
    print('def {:s}(*args):'.format(k))
    print('    \"\"\"')
    if len(helper) > 0:
      print('    {:s}'.format(helper))
    print('      r = {:}(...)'.format(v))
    print('    \"\"\"')
    print('    return {:}(*args)'.format(v))
    print('')

# ------------------------------------------------------------------
#
#
#fRw = mkDictFromRewriteFile("dev-tools/smilFuncRewrite.py")

fRw, fSc = mkDictFromListFile("dev-tools/functions-renamed.txt")

mkRewritePrototypes(fRw, 'Functions renamed', 'Function renamed. Use:')

mkRewritePrototypes(fSc, 'Shortcuts', 'Shortcut')


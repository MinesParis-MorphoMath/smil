#! /usr/bin/env python3

#
# This python script helps creation of module smilFuncRewrite
#

import re

fin = None


#
#
#
def mkHeader():
    s = """
#
# Rewrite deprecated functions names for compatibility
#
# -------------------------------------
from smilAdvancedPython       import *
from smilBasePython           import *
from smilCorePython           import *
from smilGuiPython            import *
from smilIOPython             import *
from smilMorphoPython         import *
  """
    print(s)

    # Add here only Addons which are always enabled
    #    and have functions to be redeclared
    __Addons__ = [
        "smilColorPython",
        "smilFiltersPython",
    ]
    for m in __Addons__:
        s = f"from {m:18s}       import *"
        print(s)
    print()
    s = "from smil_Python              import *"
    print(s)
    s = """
# -------------------------------------
  """
    # print(s)


#
#
#
def mkDictFromListFile(fin=None, d={}, s={}, a=[]):
    d = {}
    s = {}
    r = None
    with open(fin, "r") as f:
        for line in f:
            line = line.rstrip()
            if "__rewrite__" in line:
                r = d
                continue
            if "__shortcuts__" in line:
                r = s
                continue
            if "__additions__" in line:
                r = a
                continue
            if line.startswith("#") and r is not a:
                continue
            if isinstance(r, dict):
                m = re.search("\s*(\S+)\s+(\S+)", line)
                if m is None:
                    continue
                r[m.group(1)] = m.group(2)
            if isinstance(r, list):
                r.append(line)

    return d, s, a


#
#
#
def dumpAdditions(d={}, titre="", helper=""):
    print("# -------------------------------------")
    print("# {:s}".format(titre))
    print("#")
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            v = d[k]
            print("def {:s}(*args):".format(k))
            print('    """')
            if len(helper) > 0:
                print("    {:s}".format(helper))
            print("      r = {:}(...)".format(v))
            print('    """')
            print("    return {:}(*args)".format(v))
            print("")

    if isinstance(d, list):
        for line in d:
            print(line)


# ------------------------------------------------------------------
#
#

dRewrite, dShorts, lAdds = mkDictFromListFile("dev-tools/functions-renamed.txt")

mkHeader()

# Functions rewrite
dumpAdditions(dRewrite, "Functions renamed", "Function renamed. Use:")

# Shortcuts
dumpAdditions(dShorts, "Shortcuts", "Shortcut")

# Additions to Python Interface
dumpAdditions(lAdds, "Additions", "")

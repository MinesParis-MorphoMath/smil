#!/usr/bin/env python

import sys
import getopt
from xml.dom import minidom
from xml.dom import Node
import re



class swigXmlObj():
    def __init__(self):
	pass

badObj = None

def addAttr(obj, name, attr):
    global badObj
    name = name + "s"
    if hasattr(obj, name):
      _attr = getattr(obj, name)
      _attr.append(attr)
    else:
      setattr(obj, name, [attr])
      
def parseAttributes(node, printOut=False, tab=""):
    obj = swigXmlObj()
    if printOut:
      print tab + node.nodeName
      tab += "\t"
      
    attr = node.attributes
    for k in attr.keys():
      if k!="id" and k!="addr":
	v = attr.get(k).value
	setattr(obj, k, v)
	if printOut:
	  print tab + k + " -> " + str(v)
	  
    if node.hasChildNodes:
	for n in node.childNodes:
	  if n.nodeType==Node.ELEMENT_NODE:
	    if n.nodeName=="attributelist":
	      for attr in n.childNodes:
		if attr.nodeType==Node.ELEMENT_NODE:
		    if attr.nodeName=="attribute":
		      nodeAttr = attr.attributes
		      name = nodeAttr.get("name").value
		      value = nodeAttr.get("value").value
		      if value!="":
			setattr(obj, name, str(value))
			if printOut:
			  print tab + name + " -> " + str(value)
		    else:
			sub_obj = parseAttributes(attr, printOut, tab)
			addAttr(obj, attr.nodeName, sub_obj)
	    else:
	      name = n.nodeName
	      if name=="class":
		name = "classe"
	      sub_obj = parseAttributes(n, printOut, tab)
	      addAttr(obj, name, sub_obj)
	  
    return obj

def swigType(arg):
    splt = arg.split('.')
    ptr = ""
    buf = ""
    for i in splt:
      if i=="p":
	ptr = "*"
      elif i=="r":
	ptr = "&"
      else:
	buf += re.sub('(q.)|[()]', "", i) + " "
    return buf + ptr
  
class swigVar():
    def __init__(self, xmlObj):
	self.name = xmlObj.name
	self.access = ""
	if hasattr(xmlObj, "access"):
	  self.access = xmlObj.access
	self.type = swigType(xmlObj.type)
    def getDecl(self):
	return self.type + self.name + ";"

class swigFunc():
    def __init__(self, xmlObj, ownerClass=None):
	self.name = xmlObj.name
	if hasattr(xmlObj, "sym_name"):
	  self.sym_name = xmlObj.sym_name
	self.args = []
	if hasattr(xmlObj, "decl"):
	  decl = re.sub('f\((.*)\)\.',  '', xmlObj.decl)
	  if hasattr(xmlObj, "type"):
	    self.retType = swigType(decl + xmlObj.type)
	  else:
	    self.ret_type = ""
	else:
	  self.retType = ""
	self.access = None
	self.code = None
	if hasattr(xmlObj, "code"):
	  self.code = xmlObj.code
	if hasattr(xmlObj, "access"):
	  self.access = xmlObj.access
	if hasattr(xmlObj, "parmlists"):
	  _args = xmlObj.parmlists[0].parms
	  for a in _args:
	    if hasattr(a, "name"):
	      self.args.append(swigType(a.type) + a.name)
	if hasattr(xmlObj, "storage"):
	  self.abstract = xmlObj.storage + " "
	else:
	  self.abstract = ""
	if hasattr(xmlObj, "ismember"):
	  self.ownerClass = ownerClass
	else:
	  self.ownerClass = None
	if hasattr(xmlObj, "value"):
	  self.value = xmlObj.value
	else:
	  self.value = None
	if hasattr(xmlObj, "module"):
	  self.module = xmlObj.module
	else:
	  self.module = None
	self.arg_str = ""
	for a in self.args:
	    self.arg_str += a + ","
	if len(self.args)!=0:
	  self.arg_str = self.arg_str[:-1]
	  
    def getDecl(self):
	buf = self.abstract + self.retType + self.name + "(" + self.arg_str + ")"
	if self.value:
	  buf += " = " + self.value
	buf += ";"
	return buf
      
    def getDef(self):
	buf = ""
	if not self.code:
	  return buf
	buf = self.retType
	if self.ownerClass:
	  buf += self.ownerClass + "::"
	buf += self.name + "(" + self.arg_str + ")\n"
	buf += self.code
	return buf
	
class swigTmplFunc(swigFunc):
    def __init__(self, xmlObj):
	swigFunc.__init__(self, xmlObj)
    def getDecl(self):
	return "template <class T>\n" + swigFunc.getDecl(self)
    def getDef(self):
	return "template <class T>\n" + swigFunc.getDef(self)
	
class swigTmplSpecFunc(swigFunc):
    def __init__(self, xmlObj):
	swigFunc.__init__(self, xmlObj)
    def getDecl(self):
	return "template <>\n" + swigFunc.getDecl(self)
    def getDef(self):
	return "template <>\n" + swigFunc.getDef(self)
	

class swigClass():
    def __init__(self, xmlObj):
	self.name = xmlObj.name
	if hasattr(xmlObj, "sym_name"):
	  self.sym_name = xmlObj.sym_name
	self.memberFuncs = []
	self.memberVars = []
	if hasattr(xmlObj, "constructors"):
	  for c in xmlObj.constructors:
	    self.memberFuncs.append(swigFunc(c, self.name))
	if hasattr(xmlObj, "destructors"):
	  for c in xmlObj.destructors:
	    self.memberFuncs.append(swigFunc(c, self.name))
	if hasattr(xmlObj, "cdecls"):
	  for c in xmlObj.cdecls:
	    if c.kind=="function" and bool(c.ismember):
	      self.memberFuncs.append(swigFunc(c, self.name))
	    if c.kind=="variable" and bool(c.ismember):
	      self.memberVars.append(swigVar(c))
	self.base = " : "
	if hasattr(xmlObj, "baselists"):
	  for b in xmlObj.baselists[0].bases:
	    self.base += "public " + b.name + ", "
	self.base = self.base[:-2]
    def getDecl(self):
	buf = "class " + self.sym_name + self.base + "\n{\n"
	curAccess = ""
	for m in self.memberVars+self.memberFuncs:
	  if m.access!=curAccess:
	    curAccess = m.access
	    buf += curAccess + ":\n"
	  buf += "\t" + m.getDecl() + "\n"
	buf += "};"
	return buf
    

class swigTmplClass(swigClass):
    def __init__(self, xmlObj):
	swigClass.__init__(self, xmlObj)
    def getDecl(self):
	buf = "template <class T>\nclass " + self.name + "\n{\n"
	curAccess = ""
	for m in self.memberVars+self.memberFuncs:
	  if m.access!=curAccess:
	    curAccess = m.access
	    buf += curAccess + ":\n"
	  buf += "\t" + m.getDecl() + "\n"
	buf += "};"
	return buf
      
class swigModule():
    def __init__(self, xmlObj):
	self.name = xmlObj.name
	self.vars = []
	self.funcs = []
	self.classes = []
	self.tmplClasses = []
	self.tmplFuncs = []
	self.tmplSpecFuncs = []
	self.includes = []
	self.namespaces = []
	
	if hasattr(xmlObj, "cdecls"):
	  for c in xmlObj.cdecls:
	    if c.kind=="function":
	      if hasattr(c, "template"):
		self.tmplSpecFuncs.append(swigTmplSpecFunc(c))
	      else:
		self.funcs.append(swigFunc(c))
	    elif c.kind=="variables":
	      self.vars.append(swigVar(c))
	if hasattr(xmlObj, "classes"):
	  for c in xmlObj.classes:
	    self.classes.append(swigClass(c))
	if hasattr(xmlObj, "templates"):
	  for t in xmlObj.templates:
	    if hasattr(t, "kind") and t.kind=="class":
	      self.tmplClasses.append(swigTmplClass(t))
	    else:
	      self.tmplFuncs.append(swigTmplFunc(t))
	if hasattr(xmlObj, "includes"):
	  for i in xmlObj.includes:
	    self.includes.append(swigModule(i))
	if hasattr(xmlObj, "imports"):
	  imports = xmlObj.imports[0]
	  if hasattr(imports, "namespaces"):
	    names = []
	    for n in imports.namespaces:
	      if not n.name in names:
		names.append(n.name)
		self.namespaces.append(n)


def usage():
    print "Usage:"
    print sys.argv[0], "libName xmlFileName outSrcFile outHeaderFile"

def main():
    args = sys.argv[1:]
    if len(args) != 4:
      #args = ["smilBase", "smilBaseCPP_wrap.cpp", "smilBase.h"]
      usage()
      return

    libName = args[0]
    xmlFileName = args[1]
    outSrcFile = args[2]
    outHeaderFile = args[3]
    swigFileName = libName + ".i"

    global doc, rootNode, swigXmlRoot, swigXmlInc, mod, inc, decls, defs
    
    doc = minidom.parse(xmlFileName)
    rootNode = doc.childNodes[0]

    swigXmlRoot = parseAttributes(rootNode)

    for i in swigXmlRoot.includes:
      if i.name.split("/")[-1]==swigFileName:
	swigXmlInc = i
	break
	
    mod = swigModule(swigXmlInc)
    inc = mod.includes[0]

    decls = ""
    defs = ""

    for i in mod.includes:
	fName = i.name.split('/')[-1]
	shortFname, fExt = fName.split('.')
	if fExt=="i":
	  if fName[0:4]=="smil":
	    if shortFname!="smilCommon":
	      defs += "#include \"" + shortFname + ".h\"\n"
	else:
	  defs += "#include \"" + i.name + "\"\n"
    defs += "\n"
    for n in mod.namespaces:
	defs += "using namespace " + n.name + ";\n"
    defs += "\n"

    spec = mod.tmplSpecFuncs
    for f in spec:
	pos = f.name.find('<')
	base_name = f.name[0:pos]
	tmpl_parms = f.name[pos:]
	overl_decl = f.retType + base_name + "(" + f.arg_str + ")"
	decls += overl_decl + ";\n"
	f.name = re.sub('[()]', "", f.name)
	args = ""
	for arg in f.args:
	  args += arg.split(" ")[-1] + ","
	args = args[:-1]
	args = re.sub('[*&]', "", args)
	defs += overl_decl + "\n{\n\treturn " + f.name + "(" + args + ");\n}\n"

    hFile = open(outHeaderFile, "w")
    hFile.write(decls)
    hFile.close()

    cppFile = open(outSrcFile, "w")
    cppFile.write(defs)
    cppFile.close()




if __name__ == '__main__':
    main()


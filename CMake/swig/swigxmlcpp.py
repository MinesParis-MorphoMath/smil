#!/usr/bin/env python

import sys, os
import getopt
from xml.dom import minidom
from xml.dom import Node
import re

exportPrivateMembers = False


def createSwigObj(xmlNode, parent=None):
    cName = "swig_" + xmlNode.nodeName
    if cName in globals():
      constr = globals()[cName]
      return constr(xmlNode, parent)
    else:
      #print "Warning: unknown swig type:", xmlNode.nodeName
      return swigXmlObj(xmlNode, parent)
  
def cleanTemplateParams(name):
    # check if we have "<(...)>"
    tmStr = re.sub('.*(<\(.*\)>)', r'\1', name)
    if tmStr==name:
      return name
    decTmStr = decodeType(tmStr[2:-2])
    return name.replace(tmStr, "< " + decTmStr + " >")
    
    
class swigXmlObj():
    def __init__(self, xmlNode=None, parent=None, recursive=True):
      self.swigType = ""
      self.parent = parent
      self.children = []
      self.node = xmlNode
      if xmlNode:
	self.parseXml(xmlNode, recursive)
      if hasattr(self, "name"):
	if not hasattr(self, "sym_name"):
	  self.sym_name = self.name
	# remove parentheses from template parameters
	self.name = cleanTemplateParams(self.name)
	self.sym_name = cleanTemplateParams(self.sym_name)
	
    def parseXml(self, node, recursive=True):
      self.children = []
      self.swigType = node.nodeName
      if node.hasChildNodes():
	  for n in node.childNodes:
	    if n.nodeType==Node.ELEMENT_NODE:
	      if n.nodeName=="attributelist":
		for attr in n.childNodes:
		  if attr.nodeType==Node.ELEMENT_NODE:
		      attrName = attr.nodeName
		      if attrName=="attribute":
			nodeAttr = attr.attributes
			name = nodeAttr.get("name").value
			value = nodeAttr.get("value").value
			if value!="":
			  setattr(self, name, str(value))
		      elif recursive: 
			if attr.hasChildNodes():
			  if attr.childNodes[1].nodeName=="attributelist": 
			    self.children.append(createSwigObj(attr, self))
			  else: # example: parmlist of cdecl
			    setattr(self, attrName, createSwigObj(attr, self))
	      elif recursive:
		self.children.append(createSwigObj(n, self))
		
    def toString(self, tab=""):
      buf = ""
      d = self.__dict__
      for k in d:
	if k!="parent":
	  val = d[k]
	  if type(val)!=list:
	    buf += tab + k + ": " + str(val) + "\n"
	  else:
	    if len(val)!=0:
	      buf += "k:\n"
	      for c in val:
		buf += c.toString(tab + "\t")
      return buf
    
    def getChild(self, swigTypeName):
      d = self.__dict__
      for k in d:
	val = d[k]
	if type(val)==list:
	  for c in val:
	    if c.swigType==swigTypeName:
	      return c
      return None
	
    def getChildren(self, swigTypeName, recursive=True):
      cList = []
      for c in self.children:
	if c.swigType==swigTypeName:
	  cList.append(c)
	if recursive:
	  cList += c.getChildren(swigTypeName)
      return cList
	
    def __str__(self):
      return self.toString()
    
    def getDeclaration(self, tab=""):
      return self.childrenDeclaration(tab)
      
    def childrenDeclaration(self, tab=""):
      buf = ""
      for c in self.children:
	buf += c.getDeclaration(tab)
      return buf
    
    def getDefinition(self, tab=""):
      return self.childrenDefinition(tab)

    def childrenDefinition(self, tab=""):
      buf = ""
      for c in self.children:
	buf += c.getDefinition(tab)
      return buf
    
#### INCLUDE ####
class swig_include(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      swigXmlObj.__init__(self, xmlNode, parent)
      self.fullPath = self.name
      self.fileName = os.path.basename(self.name)
      self.path = os.path.dirname(self.name)
      self.name, self.extention = os.path.splitext(self.fileName)
  def getDeclaration(self, tab=""):
    if self.extention==".mmi":
      return "#include \"" + self.fullPath + "\"\n"
    else:
      return "\n//// " + self.fileName + "\n" + self.childrenDeclaration(tab)
  def getDefinition(self, tab=""):
    buf = ""
    if self.extention!=".i":
      buf += "#include \"" + self.fullPath + "\"\n"
    return buf + "\n" + self.childrenDefinition(tab)

#### MODULE ####
class swig_module(swigXmlObj):
  def getDeclaration(self, tab=""):
    buf = tab + "\n//// " + self.name + " declarations\n"
    buf += self.childrenDeclaration(tab)
    return buf
  def getDefinition(self, tab=""):
    buf = tab + "\n//// " + self.name + " definitions\n"
    buf += self.childrenDefinition(tab)
    return buf

#### INSERT ####
class swig_insert(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      swigXmlObj.__init__(self, xmlNode, parent)
  def getDefinition(self, tab=""):
    return self.code

#### NAMESPACE ####
class swig_namespace(swigXmlObj):
  def getDeclaration(self, tab=""):
    buf = self.childrenDeclaration(tab + "\t")
    if buf=="":
      return buf
    buf = "namespace " + self.name + "\n{\n" + buf
    buf += "} // namespace " + self.name + "\n"
    return buf
  def getDefinition(self, tab=""):
    buf = self.childrenDefinition(tab + "\t")
    if buf=="":
      return buf
    buf = "namespace " + self.name + "\n{\n" + buf
    buf += "} // namespace " + self.name + "\n"
    return buf

#### CDECL (VARIABLE, FUNCTION OR TYPEDEF) ####
class swig_cdecl(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      self.decl = ""
      self.kind = ""
      self.parmlist = swig_parmlist()
      self.code = ""
      swigXmlObj.__init__(self, xmlNode, parent)
      suffix = re.sub('f\(.*', '', self.decl)
      self.suffix = (" " + decodeType(suffix))[:-1]
      decl = re.sub('.*f\((.*)\)\.',  '', self.decl)
      self.type = cleanTemplateParams(self.type)
      self.ret_type = decodeType(self.type) + decodeType(decl)
      self.args = self.parmlist.str_list
      self.ismember = hasattr(self, "ismember")
  def getDeclaration(self, tab=""):
    if self.ismember and not exportPrivateMembers: # Verify the type of access 
      if self.access=="private":
	return ""
    if self.kind=="typedef":
      return tab + "typedef " + self.ret_type  + " " + self.sym_name + ";\n"
    elif self.kind=="variable":
      return tab + self.ret_type  + " " + self.sym_name + ";\n"
    else:
      buf = tab + self.ret_type + self.sym_name
      buf += "(" + self.args + ")" + self.suffix + ";\n"
      return buf
  def getDefinition(self, tab=""):
    if self.kind=="typedef":
      return ""
    elif self.kind=="variable":
      return ""
    else:
      buf = self.ret_type + self.sym_name
      buf += "(" + self.args + ")" + self.suffix + "\n"
      buf += self.code + "\n"
      return buf

#### PARAMETER LIST ####  
class swig_parmlist(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      swigXmlObj.__init__(self, xmlNode, parent)
      pList = [ p.sym_type + p.name + p.suffix for p in self.getChildren("parm")]
      self.str_list = ", ".join(pList)

#### PARAMETER ####
class swig_parm(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      self.name = ""
      swigXmlObj.__init__(self, xmlNode, parent)
      _type = re.sub('^a\([0-9]?\)\.', '', self.type)
      if _type==self.type:
	self.suffix = ""
      else:
	self.suffix = decodeType(re.sub(_type, '', self.type))
      self.sym_type = decodeType(_type)


#### TEMPLATE FUNCTION ####
class swig_template_function(swig_cdecl):
  def __init__(self, xmlNode=None, parent=None):
      swig_cdecl.__init__(self, xmlNode, parent)
      self.swigType = "template_function"
      self.specialization = hasattr(self, "specialization")
  def getDeclaration(self, tab=""):
    return ""
  def getDefinition(self, tab=""):
    return ""
      

#### CLASS ####
class swig_class(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      self.baselist = swigXmlObj()
      swigXmlObj.__init__(self, xmlNode, parent)
      self.template = hasattr(self, "template")
  def getDeclaration(self, tab=""):
    name = re.sub('^[A-Za-z0-9]*::', '', self.name)
    buf = tab + self.kind + " " + name + self.baselist.getDeclaration(tab) + "\n"
    buf += tab + "{\n"
    buf += self.childrenDeclaration(tab + "\t")
    buf += tab + "};\n"
    return buf
  def getDefinition(self, tab=""):
    return ""
  
#### ACCESS (CLASS) ####
class swig_access(swigXmlObj):
  def getDeclaration(self, tab=""):
    if not exportPrivateMembers and self.kind=="private":
      return ""
    return tab[:-1] + "    " + self.kind + ":\n"
  
#### BASELIST (CLASS) ####
class swig_baselist(swigXmlObj):
  def getDeclaration(self, tab=""):
    return " : public " + ", ".join([b.name for b in self.getChildren("base")])
  
#### BASE (CLASS) ####
class swig_base(swigXmlObj):
  def __init__(self, xmlNode=None, parent=None):
      attr = xmlNode.attributes
      self.name = attr.get("name").value
      swigXmlObj.__init__(self, xmlNode, parent)

#### CONSTRUCTOR (CLASS) ####
class swig_constructor(swig_cdecl):
  def __init__(self, xmlNode=None, parent=None):
      self.type = ""
      swig_cdecl.__init__(self, xmlNode, parent)
      self.ret_type = ""

#### DESTRUCTOR (CLASS) ####
swig_destructor = swig_constructor


#### TEMPLATE CLASS ####
class swig_template_class(swig_class):
  def __init__(self, xmlNode=None, parent=None):
      swig_class.__init__(self, xmlNode, parent)
      self.swigType = "template_class"
      self.specialization = hasattr(self, "specialization")
  def getDeclaration(self, tab=""):
    if self.specialization:
      buf = tab + "template<>\n"
    else:
      buf = tab + "template <class T>\n"
    buf += swig_class.getDeclaration(self, tab)
    return buf
  def getDefinition(self, tab=""):
    #if not hasattr(self, "specialization"):
      #return ""
    buf = "//// " + self.name + "\n"
    buf += self.childrenDefinition(tab + "\t")
    buf = "//// " + self.name + "\n"
    return buf
    

#### TEMPLATE (CLASS OR FUNCTION) ####
def swig_template(xmlNode, parent=None):
  obj = swigXmlObj(xmlNode, parent, recursive=False)
  print obj.name, obj.templatetype, hasattr(obj, "specialization")
  if obj.templatetype=="class":
    return swig_template_class(xmlNode, parent)
  elif obj.templatetype=="cdecl":
    return swig_template_function(xmlNode, parent)
  else:
    return swigXmlObj(xmlNode, parent)
  

#### USING ####
class swig_using(swigXmlObj):    
  def getDeclaration(self, tab=""):
    return ""
  def getDefinition(self, tab=""):
    #return ""
    return "using namespace " + self.namespace + ";\n"

#### ENUM ####
class swig_enum(swigXmlObj):    
  def getDeclaration(self, tab=""):
    buf = tab + "enum " + self.sym_name + " {\n"
    buf += ",\n".join([ tab*2+i.getDeclaration() for i in self.getChildren("enumitem") ]) + "\n"
    buf += tab + "};\n"
    return buf

#### ENUM ITEM ####
class swig_enumitem(swigXmlObj):    
  def getDeclaration(self, tab=""):
    buf = self.name
    if hasattr(self, "enumvalue"):
      buf += "=" + self.enumvalue
    return buf

#### TYPE ####
def decodeType(arg):
    if arg.find(",")!=-1:
      args = arg.split(",")
      return ",".join([decodeType(a) for a in args])
    if arg=="v(...)": # va_arg
      return "..."
    splt = arg.split('.')
    ptr = ""
    buf = ""
    for i in splt:
      if i=="p":
	ptr = "*"
      elif i=="r":
	ptr = "&"
      elif i[:2]=="a(":
	ptr = "[" + i[2:-1] + "]"
      else:
	sub = re.sub('(q.)|[()]', "", i)
	if sub!="":
	  buf +=  sub + " "
    if buf[:2]=="p ":
      ptr = "*"
      buf = buf[2:]
    elif buf[:2]=="r ":
      ptr = "r"
      buf = buf[2:]
    buf = re.sub('^[A-Za-z0-9]*::', "", buf)
    return buf + ptr
  
      
    
    

def usage():
    print "Usage:"
    print sys.argv[0], "libName xmlFileName outSrcFile outHeaderFile"

def main():
    args = sys.argv[1:]
    if len(args) != 4:
      #args = ["smilIO", "smilIO_wrap.xml", "smilIOCPP_wrap.cpp", "smilIO.h"]
      args = ["smilCore", "smilCore_wrap.xml", "smilCoreCPP_wrap.cpp", "../include/smilCore.h"]
      #usage()
      #return

    libName = args[0]
    xmlFileName = args[1]
    outSrcFile = args[2]
    outHeaderFile = args[3]
    swigFileName = libName + ".i"

    xmlFileName = "/home/mat/src/Smil/build/Core/smilCore_wrap.xml"
    
    global doc, rootNode, swigXmlRoot, swigXmlInc, mod, inc, decls, defs, swigMod
    
    print os.getcwd()
    
    if not "swigMod" in globals():
      doc = minidom.parse(xmlFileName)
      rootNode = doc.childNodes[0]
      
      #for n in rootNode.childNodes:
	#if n.nodeType==Node.ELEMENT_NODE and n.nodeName=="include":
	  #if n.hasChildNodes():
	    #attNode = n.childNodes[1]
	    #incName = os.path.basename(attNode.childNodes[1].attributes.get("value").value)
	    #if incName[:-2]:
	      #moduleRootNode = attNode
	      #break
      
    swigXmlRoot = swigXmlObj(rootNode)
    swigMod = swigXmlRoot.children[1]
    
    
    outHeaderShortFile = os.path.split(outHeaderFile)[-1]
    _outHeaderShortFile = outHeaderShortFile.replace(".", "_")
    
    decls = "// CMAKE generated file: DO NOT EDIT!\n\n"
    decls += "#ifndef __" + _outHeaderShortFile + "\n"
    decls += "#define __" + _outHeaderShortFile + "\n\n"
    decls += swigMod.getDeclaration()
    decls += "\n#endif // __" + _outHeaderShortFile + "\n"
    
    
    hFile = open(outHeaderFile, "w")
    hFile.write(decls)
    hFile.close()

    defs = ""
    #defs = "#include \"" + outHeaderFile + "\"\n\n"
    defs += swigMod.getDefinition()
    
    cppFile = open(outSrcFile, "w")
    cppFile.write(defs)
    cppFile.close()
    
    return
  
    for i in swigXmlRoot.includes:
      if i.name.split("/")[-1]==swigFileName:
	swigXmlInc = i
	break
	
    mod = swigModule(swigXmlInc)
    inc = mod.includes[0]

    #return
  
    outHeaderShortFile = os.path.split(outHeaderFile)[-1]
    _outHeaderShortFile = outHeaderShortFile.replace(".", "_")
    
    decls = "// CMAKE generated file: DO NOT EDIT!\n\n"
    decls += "#ifndef __" + _outHeaderShortFile + "\n"
    decls += "#define __" + _outHeaderShortFile + "\n\n"
    decls += mod.getDecl()
    decls += "\n#endif // __" + _outHeaderShortFile + "\n"
    
    hFile = open(outHeaderFile, "w")
    hFile.write(decls)
    hFile.close()

    defs = mod.getDef()
    
    cppFile = open(outSrcFile, "w")
    cppFile.write(defs)
    cppFile.close()
    




if __name__ == '__main__':
    main()


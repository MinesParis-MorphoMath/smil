%feature("autodoc", "1");

%include <windows.i> 
%include <std_string.i>
%include <typemaps.i>

%rename(__lshift__)  operator<<; 

%define TEMPLATE_WRAP_CLASS(_class) 
  %template(_class ## _UINT8) _class<UINT8>;
  %template(_class ## _UINT16) _class<UINT16>;
/*  %template(_class ## _UINT32) _class<UINT32>;*/
%enddef

%define TEMPLATE_WRAP_FUNC(func)
  %template(func) func<UINT8>;
/*  %template(func) func<UINT16>; */
/*  %template(func) func<UINT32>; */
%enddef

%define TEMPLATE_WRAP_FUNC2(func)
  %template(func) func<UINT8,UINT8>;
  %template(func) func<UINT8,UINT16>;
  %template(func) func<UINT16,UINT8>;
  %template(func) func<UINT32>;
%enddef

%define TEMPLATE_WRAP_FUNC_IMG(func) 
  %template(func) func<Image_UINT8>;
  %template(func) func<Image_UINT16>;
  %template(func) func<Image_UINT32>;
%enddef


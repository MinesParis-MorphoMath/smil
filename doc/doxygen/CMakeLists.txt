# Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Matthieu FAESSEL, or ARMINES nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_package(Doxygen REQUIRED)

set(DOXYGEN_CONFIG_FILE SmilDoxyfile.in)
set(DOXY_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

set(DOXYGEN_STYLE_FILE SmilStyle.css.in)
set(DOXY_STYLE_FILE ${CMAKE_CURRENT_BINARY_DIR}/SmilStyle.css)

set(DOXYGEN_LAYOUT_FILE DoxygenLayout.xml.in)
set(DOXY_LAYOUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/DoxygenLayout.xml)

if(DOXYGEN_FOUND AND DOXYGEN_CONFIG_FILE)

  # Convert svg images (requires Inkscape or ImageMagick) (note that conversion
  # is better with inkscape)
  set(SVG_CONVERT_CMD)
  find_program(INKSCAPE_EXECUTABLE inkscape${CMAKE_EXECUTABLE_SUFFIX})
  if(INKSCAPE_EXECUTABLE)
    set(SVG_CONVERT_CMD ${INKSCAPE_EXECUTABLE})
  else(INKSCAPE_EXECUTABLE)
    find_package(ImageMagick)
    if(ImageMagick_FOUND)
      set(SVG_CONVERT_CMD ${IMAGEMAGICK_CONVERT_EXECUTABLE})
    endif(ImageMagick_FOUND)
  endif(INKSCAPE_EXECUTABLE)

  set(SVG_IMAGE_FILES)
  set(PNG_IMAGE_FILES)
  set(EPS_IMAGE_FILES)

  file(GLOB SVG_IMAGE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/images/*.svg")
  if(SVG_CONVERT_CMD AND SVG_IMAGE_FILES)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/images)
    foreach(_FCH ${SVG_IMAGE_FILES})
      get_filename_component(_FCH_NAME_WE ${_FCH} NAME_WE)
      list(APPEND PNG_IMAGE_FILES
           "${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.png")
      if(INKSCAPE_EXECUTABLE)
        add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.png
                 ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.eps
          COMMAND
            ${INKSCAPE_EXECUTABLE} ${_FCH} --export-type=png
            --export-filename=${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.png
          COMMAND
            ${INKSCAPE_EXECUTABLE} ${_FCH} --export-type=eps
            --export-filename=${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.eps
          COMMENT "Converting ${_FCH_NAME_WE}.svg"
          DEPENDS ${_FCH})
      else(INKSCAPE_EXECUTABLE)
        add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.png
                 ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.eps
          COMMAND ${IMAGEMAGICK_CONVERT_EXECUTABLE} ${_FCH}
                  ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.png
          COMMAND ${IMAGEMAGICK_CONVERT_EXECUTABLE} ${_FCH}
                  ${CMAKE_CURRENT_BINARY_DIR}/images/${_FCH_NAME_WE}.eps
          COMMENT "Converting ${_FCH_NAME_WE}.svg"
          DEPENDS ${_FCH})
      endif(INKSCAPE_EXECUTABLE)
    endforeach(_FCH ${SVG_IMAGE_FILES})
  endif(SVG_CONVERT_CMD AND SVG_IMAGE_FILES)

  mark_as_advanced(DOXYGEN_CONFIG_FILE)

  option(DOXYGEN_DEVEL_DOC "Generate developper (more complete) doc" OFF)

  # PROJECT ######
  set(DOXY_PROJECT_NUMBER ${SMIL_VERSION})
  set(DOXY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  file(READ DoxyAliases.txt DOXY_ALIASES)
  set(DOXY_ALIASES "\n${DOXY_ALIASES}")

  # BUILD ######
  if(DOXYGEN_DEVEL_DOC)
    set(DOXY_EXTRACT_PRIVATE YES)
    set(DOXY_HIDE_UNDOC_MEMBERS NO)
    set(DOXY_HIDE_UNDOC_CLASSES NO)
    set(DOXY_HIDE_FRIEND_COMPOUNDS NO)
  else(DOXYGEN_DEVEL_DOC)
    set(DOXY_EXTRACT_PRIVATE NO)
    set(DOXY_HIDE_UNDOC_MEMBERS YES)
    set(DOXY_HIDE_UNDOC_CLASSES YES)
    set(DOXY_HIDE_FRIEND_COMPOUNDS YES)
  endif(DOXYGEN_DEVEL_DOC)

  set(DOXY_CITE_BIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Biblio.bib)

  # INPUT ######

  set(DOXY_INPUT_DIRS
      "${DOXY_INPUT_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}")
  string(REGEX REPLACE ";" " \\\\ \n" DOXY_INPUT_DIRS "${DOXY_INPUT_DIRS}")

  set(DOXY_EXAMPLE_PATH ${CMAKE_SOURCE_DIR}/doc/demos)
  string(REGEX REPLACE ";" " \\\\ \n" DOXY_EXAMPLE_PATH "${DOXY_EXAMPLE_PATH}")

  set(DOXY_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/doc/include)
  string(REGEX REPLACE ";" " \\\\ \n" DOXY_INCLUDE_PATH "${DOXY_INCLUDE_PATH}")

  set(DOXYGEN_ADDITIONAL_IMAGE_PATH
      ""
      CACHE STRING "Additional search path for images")
  mark_as_advanced(DOXYGEN_ADDITIONAL_IMAGE_PATH)
  set(DOXY_IMAGE_PATH
      ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
      ${DOXYGEN_ADDITIONAL_IMAGE_PATH})

  string(REGEX REPLACE ";" " \\\\ \n" DOXY_IMAGE_PATH "${DOXY_IMAGE_PATH}")

  # SOURCE BROWSER ######

  if(DOXYGEN_DEVEL_DOC)
    set(DOXY_VERBATIM_HEADERS YES)
  else(DOXYGEN_DEVEL_DOC)
    set(DOXY_VERBATIM_HEADERS NO)
  endif(DOXYGEN_DEVEL_DOC)

  # HTML ####

  option(DOXYGEN_GENERATE_HTML "Generate html documentation" ON)
  if(DOXYGEN_GENERATE_HTML)
    set(DOXY_GENERATE_HTML YES)
    set(DOXYGEN_HTML_HEADER
        ""
        CACHE STRING "html header file")
    mark_as_advanced(DOXYGEN_HTML_HEADER)
    set(DOXYGEN_HTML_FOOTER
        ""
        CACHE STRING "html footer file")
    mark_as_advanced(DOXYGEN_HTML_FOOTER)
    set(DOXYGEN_HTML_STYLESHEET
        ""
        CACHE STRING "html stylesheet file")
    mark_as_advanced(DOXYGEN_HTML_STYLESHEET)

    # HTML HELP ######
    if(WIN32)
      find_package(HTMLHelp)
      if(HTML_HELP_COMPILER)
        option(DOXYGEN_GENERATE_HTML_HELP "Generate html-help documentation" ON)
        if(DOXYGEN_GENERATE_HTML_HELP)
          set(DOXY_GENERATE_HTMLHELP YES)
          set(DOXY_CHM_FILE ${DOXY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.chm)
          string(REGEX REPLACE "[/]" "\\\\" DOXY_CHM_FILE ${DOXY_CHM_FILE})
        endif(DOXYGEN_GENERATE_HTML_HELP)
      endif(HTML_HELP_COMPILER)
    endif(WIN32)
  else(DOXYGEN_GENERATE_HTML)
    set(DOXY_GENERATE_HTML NO)
  endif(DOXYGEN_GENERATE_HTML)

  # LATEX ####

  option(DOXYGEN_GENERATE_LATEX "Generate LateX documentation" OFF)
  if(DOXYGEN_GENERATE_LATEX)
    set(DOXY_GENERATE_LATEX YES)
    find_package(LATEX REQUIRED)
  else(DOXYGEN_GENERATE_LATEX)
    set(DOXY_GENERATE_LATEX NO)
    # we need latex for doxygen because of the formulas
    find_package(LATEX)
  endif(DOXYGEN_GENERATE_LATEX)
  # MARK_AS_ADVANCED(DOXYGEN_GENERATE_LATEX)

  # DOT ####

  if(DOXYGEN_DOT_FOUND)
    set(DOXY_HAVE_DOT YES)
  else(DOXYGEN_DOT_FOUND)
    set(DOXY_HAVE_DOT NO)
  endif(DOXYGEN_DOT_FOUND)

  # DEMOS ######

  # ADD PYTHON DEMOS TO EXAMPLES
  set(DOXY_DEMOS)
  file(
    GLOB PYTHON_DEMOS
    RELATIVE ${CMAKE_SOURCE_DIR}/doc/demos/python
    ${CMAKE_SOURCE_DIR}/doc/demos/python/*.py)
  foreach(_FCH ${PYTHON_DEMOS})
    get_filename_component(_F_NAME ${_FCH} NAME)
    set(DOXY_DEMOS
        "${DOXY_DEMOS}
        /**
          * @example ${_F_NAME}
          */")
    # * \\ws_exec_demo_button{${_F_NAME}} <br>
  endforeach(_FCH ${PYTHON_DEMOS})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/demos.doxygen ${DOXY_DEMOS})

  # PREDEFINED ######

  # Add compiler definitions to doxygen PREDEFINED
  set(DOXYGEN_PREDEFINED SMIL_HOME_PAGE=https://smil.cmm.minesparis.psl.eu/)
  foreach(_DEF ${COMP_DEFS})
    set(DOXYGEN_PREDEFINED "${DOXYGEN_PREDEFINED} ${_DEF} DOXYGEN")
  endforeach(_DEF ${COMP_DEFS})
  if(WRAP_PYTHON)
    set(DOXYGEN_PREDEFINED "${DOXYGEN_PREDEFINED} SWIGPYTHON")
  endif(WRAP_PYTHON)
  if(WRAP_JAVA)
    set(DOXYGEN_PREDEFINED "${DOXYGEN_PREDEFINED} SWIGJAVA")
  endif(WRAP_JAVA)
  if(WRAP_OCTAVE)
    set(DOXYGEN_PREDEFINED "${DOXYGEN_PREDEFINED} SWIGOCTAVE")
  endif(WRAP_OCTAVE)

  # Ajoute guillemets aux exe pour chemins avec espaces de windows...
  set(LATEX_COMPILER \"${LATEX_COMPILER}\")
  set(MAKEINDEX_COMPILER \"${MAKEINDEX_COMPILER}\")
  set(HTML_HELP_COMPILER \"${HTML_HELP_COMPILER}\")

  configure_file(${DOXYGEN_CONFIG_FILE} ${DOXY_CONFIG_FILE} @ONLY)
  configure_file(${DOXYGEN_LAYOUT_FILE} ${DOXY_LAYOUT_FILE} @ONLY)
  configure_file(${DOXYGEN_STYLE_FILE} ${DOXY_STYLE_FILE} @ONLY)

  add_custom_target(
    doc
    ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG}
    DEPENDS ${PNG_IMAGE_FILES} ${EPS_IMAGE_FILES})

  find_package(PythonInterp)
  set(PYTHON_DOC_INTERFACE_FILE ${PROJECT_BINARY_DIR}/smilDoc.i)
  set_source_files_properties(${PYTHON_DOC_INTERFACE_FILE} PROPERTIES GENERATED
                                                                      TRUE)
  if(NOT EXISTS ${PYTHON_DOC_INTERFACE_FILE})
    file(WRITE ${PYTHON_DOC_INTERFACE_FILE} )
  endif(NOT EXISTS ${PYTHON_DOC_INTERFACE_FILE})

  set(PYTHON_DOC_SRCS)
  set(_APPEND_ARG "")
  file(GLOB DOXY_XML_FILES ${DOXY_OUTPUT_DIRECTORY}/xml/*.xml)
  foreach(_FICH ${DOXY_XML_FILES})
    string(REPLACE ".xml" ".i" _SWIG_FICH ${_FICH})
    add_custom_command(
      OUTPUT ${_SWIG_FICH} ${PYTHON_DOC_INTERFACE_FILE}
      COMMAND
        ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doxy2swig.py
        --no-function-definition ${_APPEND_ARG} ${_FICH}
        ${PYTHON_DOC_INTERFACE_FILE}
      WORKING_DIRECTORY ${DOXY_OUTPUT_DIRECTORY}/xml
      COMMENT "Wrapping ${_FICH} doc"
      DEPENDS doc)
    list(APPEND PYTHON_DOC_SRCS ${_SWIG_FICH})
    if(NOT _APPEND_ARG)
      set(_APPEND_ARG "-a")
    endif(NOT _APPEND_ARG)

    # SET(SWIG_INTERFACE_FILES ${SWIG_INTERFACE_FILES} ${_SWIG_FICH}
    # PARENT_SCOPE)
  endforeach(_FICH ${DOXY_XML_FILES})

  add_custom_target(
    python_doc
    SOURCES ${PYTHON_DOC_SRCS}
    DEPENDS doc ${PYTHON_DOC_INTERFACE_FILE})

endif(DOXYGEN_FOUND AND DOXYGEN_CONFIG_FILE)

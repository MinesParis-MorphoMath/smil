# Python package name
set(PYTHON_PACKAGE_NAME smilPython)

# ##############################################################################
# DEPENDENCIES

# Look for Python3
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)"
             Python3_NumPy_VERSION_MATCH ${Python3_NumPy_VERSION})
set(Python3_NumPy_VERSION_MAJOR ${CMAKE_MATCH_1})
math(EXPR Python3_NumPy_NEXT_MAJOR "${Python3_NumPy_VERSION_MAJOR} + 1")
message(
  STATUS
    "Found Numpy Version ${Python3_NumPy_VERSION} in ${Python3_NumPy_INCLUDE_DIRS}"
)

# ##############################################################################
# PACKAGING (target independent)

# Generate setup.py automatically for each configuration First step: replace
# variables (@VAR@)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in
               ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in @ONLY)

# Second step: replace generator expression ($<>)
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in)

# Generate README.md for each configuration
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/README.md
  INPUT ${CMAKE_SOURCE_DIR}/README.md)

# Copy each modules to package *destination* folder for each configuration
file(GLOB MODULES ${CMAKE_CURRENT_SOURCE_DIR}/modules/*.py)
foreach(MODULE ${MODULES})
  cmake_path(GET MODULE FILENAME MODULE_NAME)
  set(MODULE_NAME ${PYTHON_PACKAGE_NAME}/${MODULE_NAME})
  file(
    GENERATE
    OUTPUT ${PYTHON_PACKAGE_ROOT_FOLDER}/${MODULE_NAME}
    INPUT ${MODULE})
endforeach()

add_custom_target(
  python_package
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_BINARY_DIR}/lib/${PYTHON_PACKAGE_NAME}
    ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_PACKAGE_NAME}
  COMMAND
    ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smil_Python>
    $<TARGET_FILE:smilAdvancedPython> $<TARGET_FILE:smilBasePython>
    $<TARGET_FILE:smilCorePython> $<TARGET_FILE:smilGuiPython>
    $<TARGET_FILE:smilIOPython> $<TARGET_FILE:smilMorphoPython>
    ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_PACKAGE_NAME})

add_dependencies(python_package smil_Python)

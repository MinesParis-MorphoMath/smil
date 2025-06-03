file(GLOB TEST_PY *.py)

foreach(_PY ${TEST_PY})
  get_filename_component(TEST_NAME ${_PY} NAME_WE)
  add_test(
    NAME ${TEST_NAME}_py
    COMMAND ${Python3_EXECUTABLE} ${_PY}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/images)
  set_property(TEST ${TEST_NAME}_py PROPERTY LABELS python)
endforeach()

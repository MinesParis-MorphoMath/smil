# Try to find Qwt libraries and include files
# QWT_INCLUDE_DIR where to find qwt_plot.h, etc.
# QWT_LIBRARIES libraries to link against
# QWT_FOUND If false, do not try to use Qwt

FIND_PATH(QWT_INCLUDE_DIR
	NAMES qwt_plot.h
	PATHS
	/usr/local/include/qwt-qt4
	/usr/local/include/qwt
	/usr/include/qwt-qt4
	/usr/include/qwt
)

FIND_LIBRARY(QWT_LIBRARY
	NAMES qwt-qt4 qwt
	PATHS /usr/local/lib /usr/lib
)

IF (QWT_INCLUDE_DIR AND QWT_LIBRARY)
    SET(QWT_FOUND TRUE)
ENDIF (QWT_INCLUDE_DIR AND QWT_LIBRARY)

INCLUDE( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Qwt DEFAULT_MSG QWT_LIBRARY QWT_INCLUDE_DIR )
MARK_AS_ADVANCED(QWT_LIBRARIES QWT_INCLUDE_DIRS)

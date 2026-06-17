# This will define the following variables:
# SYCL_FOUND               : True if the system has the SYCL library.
# SYCL_INCLUDE_DIR         : Include directories needed to use SYCL.
# SYCL_LIBRARY_DIR         : The path to the SYCL library.
# SYCL_LIBRARY             : SYCL library fullname.
# SYCL_COMPILER_VERSION    : SYCL compiler version.

include(FindPackageHandleStandardArgs)

message(STATUS "[DEBUG] FindSYCLToolkit: Starting SYCL toolkit detection")
message(STATUS "[DEBUG] FindSYCLToolkit: ENV{SYCL_ROOT} = $ENV{SYCL_ROOT}")
message(STATUS "[DEBUG] FindSYCLToolkit: ENV{CMPLR_ROOT} = $ENV{CMPLR_ROOT}")
message(STATUS "[DEBUG] FindSYCLToolkit: CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
  message(STATUS "[DEBUG] FindSYCLToolkit: Using SYCL_ROOT from environment: ${SYCL_ROOT}")
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
  message(STATUS "[DEBUG] FindSYCLToolkit: Using CMPLR_ROOT from environment: ${SYCL_ROOT}")
else()
  # Use the default path to ensure proper linking with torch::xpurt when the user is working with libtorch.
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(SYCL_ROOT "/opt/intel/oneapi/compiler/latest")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(SYCL_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
  endif()
  message(STATUS "[DEBUG] FindSYCLToolkit: Using default SYCL_ROOT: ${SYCL_ROOT}")
  if(NOT EXISTS ${SYCL_ROOT})
    message(STATUS "[DEBUG] FindSYCLToolkit: Default path does not exist: ${SYCL_ROOT}")
    set(SYCL_ROOT "")
  else()
    message(STATUS "[DEBUG] FindSYCLToolkit: Default path exists: ${SYCL_ROOT}")
  endif()
endif()

message(STATUS "[DEBUG] FindSYCLToolkit: Final SYCL_ROOT = ${SYCL_ROOT}")

string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
if(nosyclfound)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  message(WARNING "[DEBUG] FindSYCLToolkit: SYCL_ROOT is empty, cannot find SYCL toolkit")
  message(WARNING "[DEBUG] FindSYCLToolkit: Please set SYCL_ROOT or CMPLR_ROOT environment variable")
  return()
endif()

# Find SYCL compiler executable.
message(STATUS "[DEBUG] FindSYCLToolkit: Searching for SYCL compiler in: ${SYCL_ROOT}")
# For custom debug SYCL builds, look for clang++ first
find_program(
  SYCL_COMPILER
  NAMES clang++ clang icpx icx
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )
message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_COMPILER = ${SYCL_COMPILER}")

function(parse_sycl_compiler_version version_number)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${SYCL_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING ERROR_QUIET)
  string(REGEX REPLACE "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
               SYCL_VERSION_STRING_MATCH "${SYCL_VERSION_STRING}")
  
  # Check if regex matched Intel compiler format
  if(NOT SYCL_VERSION_STRING_MATCH OR SYCL_VERSION_STRING_MATCH STREQUAL SYCL_VERSION_STRING)
    # Not Intel compiler format, return empty
    set(${version_number} "" PARENT_SCOPE)
    return()
  endif()
  
  string(REPLACE "." ";" SYCL_VERSION_LIST "${SYCL_VERSION_STRING_MATCH}")
  list(LENGTH SYCL_VERSION_LIST VERSION_LIST_LENGTH)
  
  # Ensure we have at least 3 components
  if(VERSION_LIST_LENGTH LESS 3)
    set(${version_number} "" PARENT_SCOPE)
    return()
  endif()
  
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  
  # Validate that components are numeric
  if(NOT VERSION_MAJOR MATCHES "^[0-9]+$" OR NOT VERSION_MINOR MATCHES "^[0-9]+$" OR NOT VERSION_PATCH MATCHES "^[0-9]+$")
    set(${version_number} "" PARENT_SCOPE)
    return()
  endif()
  
  # Calculate the version number in the format XXXXYYZZ, using the formula (major * 10000 + minor * 100 + patch).
  math(EXPR VERSION_NUMBER_MATCH "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
  set(${version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
  message(STATUS "[DEBUG] FindSYCLToolkit: Parsing SYCL compiler version from: ${SYCL_COMPILER}")
  parse_sycl_compiler_version(SYCL_COMPILER_VERSION)
  message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_COMPILER_VERSION = ${SYCL_COMPILER_VERSION}")
else()
  message(WARNING "[DEBUG] FindSYCLToolkit: SYCL_COMPILER not found!")
endif()

if(NOT SYCL_COMPILER_VERSION)
  # For custom debug SYCL builds, skip version check and use dummy version
  message(STATUS "[DEBUG] FindSYCLToolkit: Version check skipped for custom SYCL build")
  set(SYCL_COMPILER_VERSION "99999999")
endif()

# Find include path from binary.
message(STATUS "[DEBUG] FindSYCLToolkit: Searching for include directory in: ${SYCL_ROOT}")
find_file(
  SYCL_INCLUDE_DIR
  NAMES include
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )
message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_INCLUDE_DIR = ${SYCL_INCLUDE_DIR}")

# Find include/sycl path from include path.
find_file(
  SYCL_INCLUDE_SYCL_DIR
  NAMES sycl
  HINTS ${SYCL_ROOT}/include/
  NO_DEFAULT_PATH
  )

# Due to the unrecognized compilation option `-fsycl` in other compiler.
list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

# Find library directory from binary.
message(STATUS "[DEBUG] FindSYCLToolkit: Searching for library directory in: ${SYCL_ROOT}")
find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )
message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_LIBRARY_DIR = ${SYCL_LIBRARY_DIR}")

# Define the old version of SYCL toolkit that is compatible with the current version of PyTorch.
set(PYTORCH_2_5_SYCL_TOOLKIT_VERSION 20249999)

# By default, we use libsycl.so on Linux and sycl.lib on Windows as the SYCL library name.
if (SYCL_COMPILER_VERSION VERSION_LESS_EQUAL PYTORCH_2_5_SYCL_TOOLKIT_VERSION)
  # Don't use if(WIN32) here since this requires cmake>=3.25 and file is installed
  # and used by other projects.
  # See: https://cmake.org/cmake/help/v3.25/variable/LINUX.html
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # On Windows, the SYCL library is named sycl7.lib until PYTORCH_2_5_SYCL_TOOLKIT_VERSION.
    # sycl.lib is supported in the later version.
    set(sycl_lib_suffix "7")
  endif()
endif()

# Find SYCL library fullname.
message(STATUS "[DEBUG] FindSYCLToolkit: Searching for SYCL library 'sycl${sycl_lib_suffix}' in: ${SYCL_LIBRARY_DIR}")
find_library(
  SYCL_LIBRARY
  NAMES "sycl${sycl_lib_suffix}"
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)
message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_LIBRARY = ${SYCL_LIBRARY}")

# Find OpenCL library fullname, which is a dependency of oneDNN.
find_library(
  OCL_LIBRARY
  NAMES OpenCL
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)
message(STATUS "[DEBUG] FindSYCLToolkit: OCL_LIBRARY = ${OCL_LIBRARY}")

if((NOT SYCL_LIBRARY) OR (NOT OCL_LIBRARY))
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  message(WARNING "[DEBUG] FindSYCLToolkit: SYCL libraries incomplete")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
  VERSION_VAR SYCL_COMPILER_VERSION
  )

message(STATUS "[DEBUG] FindSYCLToolkit: SYCL_FOUND = ${SYCL_FOUND}")
message(STATUS "[DEBUG] FindSYCLToolkit: Summary:")
message(STATUS "[DEBUG]   SYCL_ROOT            = ${SYCL_ROOT}")
message(STATUS "[DEBUG]   SYCL_COMPILER        = ${SYCL_COMPILER}")
message(STATUS "[DEBUG]   SYCL_COMPILER_VERSION= ${SYCL_COMPILER_VERSION}")
message(STATUS "[DEBUG]   SYCL_INCLUDE_DIR     = ${SYCL_INCLUDE_DIR}")
message(STATUS "[DEBUG]   SYCL_LIBRARY_DIR     = ${SYCL_LIBRARY_DIR}")
message(STATUS "[DEBUG]   SYCL_LIBRARY         = ${SYCL_LIBRARY}")
message(STATUS "[DEBUG]   OCL_LIBRARY          = ${OCL_LIBRARY}")


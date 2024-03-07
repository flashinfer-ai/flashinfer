##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

#
# FindThrust
#
# This module finds the Thrust header files and extrats their version.  It
# sets the following variables.
#
# THRUST_INCLUDE_DIR -  Include directory for thrust header files.  (All header
#                       files will actually be in the thrust subdirectory.)
# THRUST_VERSION -      Version of thrust in the form "major.minor.patch".
#

find_path( THRUST_INCLUDE_DIR
  HINTS ./  
        ../thrust
        ../../thrust
        ../../../thrust
        /usr/include/cuda
        /usr/local/include
        /usr/local/cuda/include
        ${CUDA_INCLUDE_DIRS}
  NAMES thrust/version.h
  DOC "Thrust headers"
  )
if( THRUST_INCLUDE_DIR )
  list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )

  # Find thrust version
  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    version
    REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_VERSION[ \t]+"
    ""
    version
    "${version}"
    )

  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    major_version
    REGEX "#define THRUST_MAJOR_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_MAJOR_VERSION[ \t]+"
    ""
    major_version
    "${major_version}"
    )

  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    major_version
    REGEX "#define THRUST_MINOR_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_MINOR_VERSION[ \t]+"
    ""
    minor_version
    "${minor_version}"
    )
  
  set( THRUST_VERSION "${version}")
  set( THRUST_MAJOR_VERSION "${major_version}")
  set( THRUST_MINOR_VERSION "${minor_version}")
endif( THRUST_INCLUDE_DIR )

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
  REQUIRED_VARS THRUST_INCLUDE_DIR
  VERSION_VAR THRUST_VERSION
  )

set(THRUST_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
mark_as_advanced(THRUST_INCLUDE_DIR)

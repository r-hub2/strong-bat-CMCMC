AC_INIT([PopMCMC], 0.1)

dnl Louis Aslett (aslett@stats.ox.ac.uk)
dnl September 2014

dnl Allow user specification of CUDA home directory either via configure variable
dnl or via environment variable
cuda_home=""
AC_ARG_WITH([cuda],
      AC_HELP_STRING([--with-cuda=MGR],
        [specify the CUDA home folder, e.g. /usr/local/cuda or /Developer/NVIDIA/CUDA-7.5 etc]),
        [cuda=$withval])
if test [ -n "$cuda" ] ; then
  CUDA_HOME="${cuda}"
else
  if test [ -n "${CUDA_HOME}" ] ; then
    CUDA_HOME="${CUDA_HOME}"
  else
    if ls /Developer/NVIDIA/* 1> /dev/null 2>&1 ; then
      CUDA_HOME=$(ls -1 -d /Developer/NVIDIA/* | grep CUDA-* | sort -r | head -1)
    else
      if ls /usr/local/cuda* 1> /dev/null 2>&1 ; then
        CUDA_HOME=$(ls -1 -d /usr/local/* | grep cuda-* | sort -r | head -1)
      else
        if ls /opt/cuda* 1> /dev/null 2>&1 ; then
          CUDA_HOME=$(ls -1 -d /opt/* | grep cuda-* | sort -r | head -1)
        fi
      fi
    fi
  fi
fi
AC_CHECK_FILE("$CUDA_HOME/lib64",
              CUDA_LIB_LIB64=lib64,
              AC_CHECK_FILE("$CUDA_HOME/lib",
                            CUDA_LIB_LIB64=lib,
                            AC_MSG_ERROR("Cannot determine lib / lib64 in CUDA_HOME.")))

dnl Name the substitution variable and instruct configure to construct Makevars
dnl from Makevars.in
AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_LIB_LIB64)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

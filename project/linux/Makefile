# -*- mode: Makefile; -*-
# -----------------------------------------
# project gce


export PATH := /opt/wx/2.8/bin:$(PATH)
export LD_LIBRARY_PATH := /opt/wx/2.8/lib:$(LD_LIBRARY_PATH)

_WX = /home/gr/projects/gui/codeblocks/wx
_WX.LIB = $(_WX)/lib
_WX.INCLUDE = $(_WX)/include

_CB = /home/gr/projects/gui/codeblocks/cb/src
_CB.INCLUDE = $(_CB)/include
_CB.LIB = $(_CB)/devel



CFLAGS_C = $(filter-out -include "sdk.h",$(CFLAGS))

# -----------------------------------------

# MAKE_DEP = -MMD -MT $@ -MF $(@:.o=.d)

CFLAGS = -Wall -DIVL_OPENCV `pkg-config --cflags opencv`
INCLUDES = -I../../../
LDFLAGS =  -lml -s
RCFLAGS = 
LDLIBS = $(T_LDLIBS) -lm -lstdc++ `pkg-config --libs opencv`

LINK_exe = gcc -o $@ $^ $(LDFLAGS) $(LDLIBS)
LINK_con = gcc -o $@ $^ $(LDFLAGS) $(LDLIBS)
LINK_dll = gcc -o $@ $^ $(LDFLAGS) $(LDLIBS) -shared
LINK_lib = rm -f $@ && ar rcs $@ $^
COMPILE_c = gcc $(CFLAGS_C) -o $@ -c $< $(MAKEDEP) $(INCLUDES)
COMPILE_cpp = g++ $(CFLAGS) -o $@ -c $< $(MAKEDEP) $(INCLUDES)
COMPILE_rc = windres $(RCFLAGS) -J rc -O coff -i $< -o $@ -I$(dir $<)

%.o : %.c ; $(COMPILE_c)
%.o : %.cpp ; $(COMPILE_cpp)
%.o : %.cxx ; $(COMPILE_cpp)
%.o : %.rc ; $(COMPILE_rc)
.SUFFIXES: .o .d .c .cpp .cxx .rc

all: all.before all.targets all.after

all.before :
	-
all.after : $(FIRST_TARGET)
	
all.targets : Release_target 

clean :
	rm -fv $(clean.OBJ)
	rm -fv $(DEP_FILES)

.PHONY: all clean distclean

# -----------------------------------------
# Release_target

Release_target.BIN = bin/Release/hpm
Release_target.OBJ = ../../src/main.o ../../src/hpm/hpmatch.o ../../src/hpm/corr.o ../../src/hpm/feature.o ../../src/draw.o
DEP_FILES += ../../src/main.d ../../src/hpm/hpmatch.d ../../src/hpm/corr.d ../../src/hpm/feature.d ../../src/draw.d 
clean.OBJ += $(Release_target.BIN) $(Release_target.OBJ)

Release_target : Release_target.before $(Release_target.BIN) Release_target.after_always
Release_target : CFLAGS += -O2  -Os
Release_target : INCLUDES += 
Release_target : RCFLAGS += 
Release_target : LDFLAGS += -s   
Release_target : T_LDLIBS = 
ifdef LMAKE
Release_target : CFLAGS -= -O1 -O2 -g -pipe
endif

Release_target.before :
	
	
Release_target.after_always : $(Release_target.BIN)
	
$(Release_target.BIN) : $(Release_target.OBJ)
	$(LINK_con)
	

# -----------------------------------------
ifdef MAKE_DEP
-include $(DEP_FILES)
endif

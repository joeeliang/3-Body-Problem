# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/joeliang/Joe/Coding/3bodyC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/joeliang/Joe/Coding/3bodyC/build

# Include any dependencies generated for this target.
include CMakeFiles/render.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/render.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/render.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/render.dir/flags.make

CMakeFiles/render.dir/main.cpp.o: CMakeFiles/render.dir/flags.make
CMakeFiles/render.dir/main.cpp.o: /Users/joeliang/Joe/Coding/3bodyC/main.cpp
CMakeFiles/render.dir/main.cpp.o: CMakeFiles/render.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/joeliang/Joe/Coding/3bodyC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/render.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/render.dir/main.cpp.o -MF CMakeFiles/render.dir/main.cpp.o.d -o CMakeFiles/render.dir/main.cpp.o -c /Users/joeliang/Joe/Coding/3bodyC/main.cpp

CMakeFiles/render.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/render.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/joeliang/Joe/Coding/3bodyC/main.cpp > CMakeFiles/render.dir/main.cpp.i

CMakeFiles/render.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/render.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/joeliang/Joe/Coding/3bodyC/main.cpp -o CMakeFiles/render.dir/main.cpp.s

# Object files for target render
render_OBJECTS = \
"CMakeFiles/render.dir/main.cpp.o"

# External object files for target render
render_EXTERNAL_OBJECTS =

render: CMakeFiles/render.dir/main.cpp.o
render: CMakeFiles/render.dir/build.make
render: CMakeFiles/render.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/joeliang/Joe/Coding/3bodyC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable render"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/render.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/render.dir/build: render
.PHONY : CMakeFiles/render.dir/build

CMakeFiles/render.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/render.dir/cmake_clean.cmake
.PHONY : CMakeFiles/render.dir/clean

CMakeFiles/render.dir/depend:
	cd /Users/joeliang/Joe/Coding/3bodyC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/joeliang/Joe/Coding/3bodyC /Users/joeliang/Joe/Coding/3bodyC /Users/joeliang/Joe/Coding/3bodyC/build /Users/joeliang/Joe/Coding/3bodyC/build /Users/joeliang/Joe/Coding/3bodyC/build/CMakeFiles/render.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/render.dir/depend

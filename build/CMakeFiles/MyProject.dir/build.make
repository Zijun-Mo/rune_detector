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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mozijun/Mycode_c/rune_detector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mozijun/Mycode_c/rune_detector/build

# Include any dependencies generated for this target.
include CMakeFiles/MyProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MyProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MyProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MyProject.dir/flags.make

CMakeFiles/MyProject.dir/src/rune_detector.cpp.o: CMakeFiles/MyProject.dir/flags.make
CMakeFiles/MyProject.dir/src/rune_detector.cpp.o: /home/mozijun/Mycode_c/rune_detector/src/rune_detector.cpp
CMakeFiles/MyProject.dir/src/rune_detector.cpp.o: CMakeFiles/MyProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mozijun/Mycode_c/rune_detector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MyProject.dir/src/rune_detector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MyProject.dir/src/rune_detector.cpp.o -MF CMakeFiles/MyProject.dir/src/rune_detector.cpp.o.d -o CMakeFiles/MyProject.dir/src/rune_detector.cpp.o -c /home/mozijun/Mycode_c/rune_detector/src/rune_detector.cpp

CMakeFiles/MyProject.dir/src/rune_detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/MyProject.dir/src/rune_detector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/rune_detector/src/rune_detector.cpp > CMakeFiles/MyProject.dir/src/rune_detector.cpp.i

CMakeFiles/MyProject.dir/src/rune_detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/MyProject.dir/src/rune_detector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/rune_detector/src/rune_detector.cpp -o CMakeFiles/MyProject.dir/src/rune_detector.cpp.s

# Object files for target MyProject
MyProject_OBJECTS = \
"CMakeFiles/MyProject.dir/src/rune_detector.cpp.o"

# External object files for target MyProject
MyProject_EXTERNAL_OBJECTS =

MyProject: CMakeFiles/MyProject.dir/src/rune_detector.cpp.o
MyProject: CMakeFiles/MyProject.dir/build.make
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
MyProject: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
MyProject: CMakeFiles/MyProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/mozijun/Mycode_c/rune_detector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MyProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MyProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MyProject.dir/build: MyProject
.PHONY : CMakeFiles/MyProject.dir/build

CMakeFiles/MyProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MyProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MyProject.dir/clean

CMakeFiles/MyProject.dir/depend:
	cd /home/mozijun/Mycode_c/rune_detector/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mozijun/Mycode_c/rune_detector /home/mozijun/Mycode_c/rune_detector /home/mozijun/Mycode_c/rune_detector/build /home/mozijun/Mycode_c/rune_detector/build /home/mozijun/Mycode_c/rune_detector/build/CMakeFiles/MyProject.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/MyProject.dir/depend


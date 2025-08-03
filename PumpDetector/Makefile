# Makefile for goProPumpingParser.cpp with GPMF-parser and libdspl-2.0 static linking

CC = g++
CFLAGS = -Wall -Wextra -O2 -DBUILD_LIB -DWIN_OS -DGLEW_STATIC -Wno-write-strings -Wno-missing-field-initializers -Wno-unused-parameter -Wno-address
INCLUDES = -I"libdspl-2.0/include" -I"gpmf-parser" -I"gpmf-parser/demo" -I"implot" -I"imgui" -I"glfw/include" -I"libtinyfiledialogs"

GLFW_DIR = glfw
GLFW_LIB = $(GLFW_DIR)/build/src
GLEW_DIR = glew-2.1.0
GLEW_LIB = $(GLEW_DIR)

DSPL_DIR = libdspl-2.0
DSPL_RELEASE = $(DSPL_DIR)/_release
DSPL_BLAS = $(DSPL_DIR)/dspl/blas/bin
DSPL_LAPACK = $(DSPL_DIR)/dspl/lapack/bin

STATIC_LIBS = $(DSPL_RELEASE)/libdspl.a \
              $(DSPL_BLAS)/libblas.a \
              $(DSPL_LAPACK)/liblapack_double.a \
              $(DSPL_LAPACK)/liblapack_complex.a \
              -L$(GLFW_LIB) -lglfw3  -lopengl32 -lgdi32 -luser32 -lkernel32 -lshell32 -lole32 -luuid -lcomctl32 -lcomdlg32 -static-libgcc -static-libstdc++

LIBS = -lm -lgfortran -lquadmath -static-libgfortran

# Source files
GPMF_SOURCES = gpmf-parser/GPMF_parser.c \
               gpmf-parser/GPMF_utils.c \
               gpmf-parser/demo/GPMF_mp4reader.c

IMGUI_SOURCES = imgui/imgui.cpp \
                imgui/imgui_draw.cpp \
                imgui/imgui_tables.cpp \
                imgui/imgui_widgets.cpp \
                imgui/imgui_impl_glfw.cpp \
                imgui/imgui_impl_opengl3.cpp

IMPLOT_SOURCES = implot/implot.cpp \
                 implot/implot_items.cpp

TINYFILEDIALOGS_SOURCES = libtinyfiledialogs/tinyfiledialogs.c

MAIN_SOURCE = goProPumpingParser.cpp pumpDetector.cpp

SOURCES = $(MAIN_SOURCE) $(GPMF_SOURCES) $(IMGUI_SOURCES) $(IMPLOT_SOURCES) $(TINYFILEDIALOGS_SOURCES)

OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.c=.o)

TARGET = goProPumpingParser

# Default target
all: $(TARGET)

# Link target from object files
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(STATIC_LIBS) $(LIBS)

# Compile C++ sources
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Fetch GPMF-parser
gpmf-parser/GPMF_parser.c:
	@echo "GPMF-parser not found. Installing..."
	git clone https://github.com/gopro/gpmf-parser.git
	@echo "GPMF-parser installed successfully."

# Fetch and build libdspl
libdspl-2.0/_release/libdspl.a:
	@echo "libdspl-2.0 not found. Installing..."
	git clone https://github.com/Dsplib/libdspl-2.0.git
	cd libdspl-2.0 && $(MAKE) static
	@echo "libdspl-2.0 installed successfully."

# Manual install
install-gpmf:
	git clone https://github.com/gopro/gpmf-parser.git

install-dspl:
	git clone https://github.com/Dsplib/libdspl-2.0.git
	cd libdspl-2.0 && $(MAKE) static

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

# Debug
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Full setup
setup: install-deps install-gpmf install-dspl all

# Minimal setup (no package manager)
setup-minimal: install-gpmf install-dspl
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm

# Static build
static: $(OBJECTS)
	$(CC) $(CFLAGS) -static -o $(TARGET) $(OBJECTS) $(STATIC_LIBS) $(LIBS)

# Install dspl with static libraries
install-dspl-static:
	@if [ ! -d "libdspl-2.0" ]; then git clone https://github.com/Dsplib/libdspl-2.0.git; fi
	cd libdspl-2.0 && $(MAKE) static

# OS-specific installs
install-deps:
	@echo "Installing dependencies for Windows..."
	@where choco >nul 2>&1 && ( \
		echo "Installing with chocolatey..." && \
		choco install git cmake mingw make -y \
	) || ( \
		echo "Chocolatey not found. Please install manually:" && \
		echo "1. Install Git: https://git-scm.com/download/win" && \
		echo "2. Install CMake: https://cmake.org/download/" && \
		echo "3. Install MinGW-w64: https://www.mingw-w64.org/downloads/" && \
		echo "4. Install Make: http://gnuwin32.sourceforge.net/packages/make.htm" \
	)

install-deps-mac:
	brew install cjson git cmake fftw openblas lapack ffmpeg
	clang++ -std=c++17 main.cpp -o gopro_plot \
		-Iimgui -Iimplot -Iglfw/include \
		-L/usr/local/lib -lglfw \
		-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo


install-deps-linux:
	sudo apt-get update
	sudo apt-get install build-essential libcjson-dev git cmake libfftw3-dev libblas-dev liblapack-dev

.PHONY: all clean debug setup static setup-minimal install-dspl install-gpmf install-dspl-static install-deps install-deps-mac install-deps-linux

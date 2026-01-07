PLUGIN_NAME = rainbow
SOURCES = rainbow.cpp

UNAME_S := $(shell uname -s)
TARGET ?= hardware

ifeq ($(TARGET),hardware)
    CXX = arm-none-eabi-g++
    CFLAGS = -std=c++11 \
             -mcpu=cortex-m7 \
             -mfpu=fpv5-d16 \
             -mfloat-abi=hard \
             -mthumb \
             -Os \
             -ffunction-sections \
             -fdata-sections \
             -fno-rtti \
             -fno-exceptions \
             -fno-unwind-tables \
             -fno-asynchronous-unwind-tables \
             -Wall
    INCLUDES = -I. -I./distingNT_API/include
    LDFLAGS = -Wl,--relocatable -nostdlib
    OUTPUT_DIR = plugins
    BUILD_DIR = build
    OUTPUT = $(OUTPUT_DIR)/$(PLUGIN_NAME).o
    OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCES))
    CHECK_CMD = arm-none-eabi-nm $(OUTPUT) | grep ' U '
    SIZE_CMD = arm-none-eabi-size $(OUTPUT)

else ifeq ($(TARGET),test)
    ifeq ($(UNAME_S),Darwin)
        CXX = clang++
        CFLAGS = -std=c++11 -fPIC -Os -Wall -fno-rtti -fno-exceptions
        LDFLAGS = -dynamiclib -undefined dynamic_lookup
        EXT = dylib
    endif

    ifeq ($(UNAME_S),Linux)
        CXX = g++
        CFLAGS = -std=c++11 -fPIC -Os -Wall -fno-rtti -fno-exceptions
        LDFLAGS = -shared
        EXT = so
    endif

    ifeq ($(OS),Windows_NT)
        CXX = cl
        CFLAGS = /std:c++11 /O2 /W3 /GR- /EHsc-
        LDFLAGS = /LD
        EXT = dll
    endif

    INCLUDES = -I. -I./distingNT_API/include
    OUTPUT_DIR = plugins
    OUTPUT = $(OUTPUT_DIR)/$(PLUGIN_NAME).$(EXT)
    CHECK_CMD = nm $(OUTPUT) | grep ' U ' || echo "No undefined symbols"
    SIZE_CMD = ls -lh $(OUTPUT)
endif

all: $(OUTPUT)

ifeq ($(TARGET),hardware)
$(OUTPUT): $(OBJECTS)
	@mkdir -p $(OUTPUT_DIR)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^
	@echo "Built hardware plugin: $@"

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

else ifeq ($(TARGET),test)
$(OUTPUT): $(SOURCES)
	@mkdir -p $(OUTPUT_DIR)
	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $(SOURCES)
	@echo "Built test plugin: $@"
endif

hardware:
	@$(MAKE) TARGET=hardware

test:
	@$(MAKE) TARGET=test

both: hardware test

check: $(OUTPUT)
	@echo "Checking symbols in $(OUTPUT)..."
	@$(CHECK_CMD) || true

size: $(OUTPUT)
	@echo "Size of $(OUTPUT):"
	@$(SIZE_CMD)

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Cleaned build and output directories"

.PHONY: all hardware test both check size clean

NVCC := nvcc

EXEC := sgemm

SRC := main.cu ../00_Utils/cudaUtils.cu
INC := -I../00_Utils

COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
ARCH := $(shell echo $(COMPUTE_CAP) | sed 's/[^0-9]*//g' | sed 's/\([0-9]\)\([0-9]\)/compute_\1\2/')

NVCCFLAGS := $(INC) -arch=$(ARCH)

all: $(EXEC)

$(EXEC): $(SRC)
	@echo "Building $@ for GPU with $(ARCH)..."
	@if [ -z "$(ARCH)" ]; then \
		echo "Error: Could not determine GPU compute!"; \
		exit 1; \
	fi
	@rm -f $@
	@$(NVCC) $(SRC) $(NVCCFLAGS) -o $@
	@echo "Build done, command used := $(NVCC) $(SRC) $(NVCCFLAGS) -o $@"

clean:
	@rm -f $(EXEC)
	@echo "Removed old executable $(EXEC)"

.PHONY: all clean
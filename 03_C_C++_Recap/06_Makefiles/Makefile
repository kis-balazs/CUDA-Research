.PHONY: 01 01_obj 01_obj_exe_run 02 03 clean

GCC := gcc
NVCC := nvcc
CUDA_FLAGS := -arch=sm_86

01: 01.c
	@$(GCC) -o 01 01.c


# compiles to object file
01_obj:
	@$(GCC) -c 01.c -o 01.o


# compiles and runs object file, using previous statement
01_obj_exe_run: 01_obj
	@echo "01_obj called as prerequisite"
	@$(GCC) 01.o -o 01
	@./01


02:
	@$(GCC) -o 02 02.c


03: # 04.c
	@$(NVCC) $(CUDA_FLAGS) -o 03_cu 03.cu


clean:
	rm -f 01 02 03_cu *.o


comand=skelet
comand2=skelet2
comand3=skelet3
comand4=skelet4

compiler: Skeletization-CUDA.cu
	/usr/local/cuda-7.5/bin/nvcc Skeletization-CUDA.cu -o skelet -DTIME
run:
	for number in 1 2 3 4 5 6 7; do \
		./skelet Input/$$number.ppm;\
	done
clean:
	rm -f $(comand)

compiler2:
	/usr/local/cuda-7.5/bin/nvcc Skeletization-CUDA.cu -o skelet2 -DNOTIME
run2:
	for number in 1 2 3 4 5 6 7; do \
		./skelet2 Input/$$number.ppm>Output-CUDA/$$number.ppm;\
	done
clean2:
	rm -f $(comand2)

compiler-Clang: Skeletization-Clang.c
	aclang -opt-poly=tile Skeletization-Clang.c -o skelet3 -DTIME

run-Clang:
	for number in 1 2 3 4 5 6 7; do \
		./skelet3 Input/$$number.ppm;\
	done
clean3:
	rm -f $(comand3)


compiler-Clang2: Skeletization-Clang.c
	aclang -opt-poly=tile Skeletization-Clang.c -o skelet4 -DNOTIME

run-Clang2:
	for number in 1 2 3 4 5 6 7; do \
		./skelet4 Input/$$number.ppm>Output-Clang/$$number.ppm;\
	done
clean4:
	rm -f $(comand4)

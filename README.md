# Parallel-Skeletonization-Algorithm
Developed parallel version of Zhang-Suen thinning algorithm, resulting in 20x speedup over the serial version. C, CUDA and 

For this project, 6 images in .ppm format with significant sizes were taken as inputs. The outputs are images of the same size in the same format.

Initially the folders Output-CUDA and Output-Clang are empty. The output images will be generated at the time of compiling the program. 

Since the project was paralleled with Clang and CUDA and both have different compilation instructions, within the Makefile there are commands for the execution of each of these programs.

I opted for two types of compilations, one to display the execution time and speedup on the screen and the other to generate the output images. The commands are as follows.

========================= for CUDA ====================================== 

NOTE: Note that for CUDA compilation is used cuda-7.5. Output images are saved in the folder Output-CUDA.

	make : 		Compile the program that will show the execution time and speedup on screen.
	run: 		Run the program using the 6 input images.
	clean: 		Clean the files.

	compiler2: 	Compiles the program that generates the output images.
	run2: 		Run the program using the 6 input images and stores the output images in the CUDA-output folder.
	clean2: 	Clean the files.


========================= for Clang ====================================== 

NOTE: The compilation was done on the Parsusy server. Output images are saved in the folder Output-Clang.

	compiler-Clang :  	Compile the program that will show the execution time and speedup on screen.
	runClang: 			Run the program using the 6 input images.
	clean-Clang: 		Clean the files.

	compiler-Clang2: 	Compiles the program that generates the output images.
	run-Clang2: 		Run the program using the 6 input images and stores the output images in the Clang-output folder.
	clean-Clang2: 		Clean the files.

OBSERVATION, after compiling in Clang by makefile, the results are different from the original, not showing the correct results (speedup, parallel time).

To see the correct results, execute the commands

aclang -opt-poly=tile Skeletization-Clang.c -o skelet -DTIME
./skelet Input/<Image-number>.ppm       example: ./skelet Input/4.ppm  where: Image-number={1,2,3,...,6}

© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About

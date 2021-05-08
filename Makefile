
comp:
	gcc main_default.c -lm -o main

run: comp
	./main

debug:
	gcc main.c -g -lm -o main

debug_run: debug
	./main

valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all --show-reachable=yes --track-origins=yes --verbose --log-file=valgrind-out2.txt ./main

profile:
	gcc main.c -Wall -pg -lm -o main
	./main
	gprof main gmon.out > analysis.txt

opt1:
	gcc main.c -lm -O3 -o main
	./main

opt2:
	gcc main_bomp.c -lm -O3 -fopenmp -o main
	OMP_NUM_THREADS=8 ./main

opt3:
	gcc main.c -lm -O3 -fopenmp -o main
	OMP_NUM_THREADS=8 ./main

profile_omp:
	gcc main.c -lm -Wall -pg -O3 -fopenmp -o main
	OMP_NUM_THREADS=4 ./main
	gprof main gmon.out > analysis_omp.txt

gpu:
	nvcc -arch sm_35 cuda_main.cu -o cuda_main
	./cuda_main

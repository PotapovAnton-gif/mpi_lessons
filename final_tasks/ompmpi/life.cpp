#include <stdio.h>
#include <assert.h>
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

#define ind(i, j) (((i + l->numsX) % l->numsX) + ((j + l->numsY) % l->numsY) * (l->numsX))
#define BILLION  1000000000L;

typedef struct {
	int numsX, numsY;
	int *u0;
	int *u1;
	int steps;
	int save_steps;

	int rank;
	int size;
	int start, end;
	
	MPI_Datatype block; 
    MPI_Datatype col;
} lifeStuct;

void life_init(const char *filePath, lifeStuct *life);
void life_free(lifeStuct *life);
void life_step(lifeStuct *life);
void life_save_vtk(const char *path, lifeStuct *life);
void life_gather(lifeStuct *life);

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	MPI_Init(&argc, &argv);
	lifeStuct l;
	life_init(argv[1], &l);
	
	int i;
	char buffer[100];
	timespec start;
	timespec stop;
	clock_gettime(CLOCK_REALTIME, &start);
#pragma omp parallel
    {	
        for (int i = 0; i < l.steps; i++) {
#pragma omp single 
            {
                if (i % l.save_steps == 0) {
                    sprintf(buffer, "life_%06d.vtk", i);
                    life_save_vtk(buffer, &l);
                }
            }
            life_step(&l);
	    }
    }
	clock_gettime(CLOCK_REALTIME, &stop);
    double time = ( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) / (double)BILLION;
	if (l.rank == 0) std::cout << time << std::endl;
	life_free(&l);
	MPI_Finalize();
	return 0;
}

void life_decomposition(const int p, const int k, const int n, int *begin, int *end) {
    
    *begin = k * (n / p);
    *end = *begin + (n / p);
    if (k == p - 1)
        *end = n;

}

void life_init(const char *filePath, lifeStuct *l) {
    FILE *fd = fopen(filePath, "r");
    assert(fd);
    assert(fscanf(fd, "%d\n", &l->steps));
    assert(fscanf(fd, "%d\n", &l->save_steps));
    assert(fscanf(fd, "%d %d\n", &l->numsX, &l->numsY));
    l->u0 = (int *) calloc(l->numsX * l->numsY, sizeof(int));
    l->u1 = (int *) calloc(l->numsX * l->numsY, sizeof(int));

    int i, j, r, counter;
    counter = 0;
    while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
        l->u0[ind(i, j)] = 1;
        counter++;
    }
    fclose(fd);
    MPI_Comm_size(MPI_COMM_WORLD, &(l->size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));

    life_decomposition(l->size, l->rank, l->numsX, &(l->start), &(l->end));

    MPI_Type_vector(l->numsY, 1, l->numsX, MPI_INT, &(l->col));
    MPI_Type_commit(&(l->col));

    int begin, end;
    life_decomposition(l->size, 0, l->numsX, &begin, &end);
    MPI_Type_vector(l->numsY, end - begin, l->numsX, MPI_INT, &(l->block));
    MPI_Type_commit(&(l->block));
}

void life_free(lifeStuct *l) {
    free(l->u0);
    free(l->u1);
    l->numsX = l->numsY = 0;
    MPI_Type_free(&(l->col));
    MPI_Type_free(&(l->block));
}

void life_save_vtk(const char *path, lifeStuct *l) {
    FILE *f;
    int i1, i2, j;
    f = fopen(path, "w");
    assert(f);
    fprintf(f, "DIMENSIONS %d %d 1\n", l->numsX + 1, l->numsY + 1);
    fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
    fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
    fprintf(f, "CELL_DATA %d\n", l->numsX * l->numsY);
    fprintf(f, "SCALARS life int 1\n");
    fprintf(f, "LOOKUP_TABLE lifeStuctable\n");
    for (i2 = 0; i2 < l->numsY; i2++) {
        for (i1 = 0; i1 < l->numsX; i1++) {
            fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
        }
    }
    fclose(f);
}


void life_step(lifeStuct *l) {
#pragma omp for
    for (int j = 0; j < l->numsY; j++) {
        for (int i = l->start; i < l->end; i++) {
            int n = 0;
            n += l->u0[ind(i + 1, j)];
            n += l->u0[ind(i + 1, j + 1)];
            n += l->u0[ind(i, j + 1)];
            n += l->u0[ind(i - 1, j)];
            n += l->u0[ind(i - 1, j - 1)];
            n += l->u0[ind(i, j - 1)];
            n += l->u0[ind(i - 1, j + 1)];
            n += l->u0[ind(i + 1, j - 1)];
            l->u1[ind(i, j)] = 0;
            if (n == 3 && l->u0[ind(i, j)] == 0) {
                l->u1[ind(i, j)] = 1;
            }
            if ((n == 3 || n == 2) && l->u0[ind(i, j)] == 1) {
                l->u1[ind(i, j)] = 1;
            }
        }
    }
#pragma omp single
    {
        int *tmp;
        tmp = l->u0;
        l->u0 = l->u1;
        l->u1 = tmp;
    }
}


void parall(lifeStuct *l) {
    if (l->size != 1) {
        if (l->rank % 2 == 0) {
            MPI_Send(l->u0 + ind(l->end - 1, 0), 1, l->col, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD);
            MPI_Recv(l->u0 + ind(l->start - 1, 0), 1, l->col, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->start, 0), 1, l->col, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD);
            MPI_Recv(l->u0 + ind(l->end, 0), 1, l->col, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(l->start - 1, 0), 1, l->col, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->end - 1, 0), 1, l->col, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD);
            MPI_Recv(l->u0 + ind(l->end, 0), 1, l->col, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->start, 0), 1, l->col, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD);
        }
    }
}

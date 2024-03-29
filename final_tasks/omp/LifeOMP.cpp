#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <ctime>
#include <omp.h>
#include <iostream>
using namespace std;

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))
#define BILLION  1000000000L;

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;

	int rank;
	int size;
	int start, end;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void life_gather(life_t *l);

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}	
	life_t l;
	life_init(argv[1], &l);
	omp_set_num_threads(4);
    timespec start;
	timespec stop;
	clock_gettime(CLOCK_REALTIME, &start);
	char buf[100];
#pragma omp parallel
    {	
        for (int i = 0; i < l.steps; i++) {
#pragma omp single 
            {
                if (i % l.save_steps == 0) {
                    sprintf(buf, "life_%06d.vtk", i);
                    life_save_vtk(buf, &l);
                }
            }
            life_step(&l);
	    }
    }
	clock_gettime(CLOCK_REALTIME, &stop);
    double time = ( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec ) / (double)BILLION;
	if (l.rank == 0) cout << time << endl;
	life_free(&l);

	return 0;
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{
	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
	printf("Field size: %dx%d\n", l->nx, l->ny);

	l->u0 = (int*)calloc(l->nx * l->ny, sizeof(int));
	l->u1 = (int*)calloc(l->nx * l->ny, sizeof(int));
	
	int i, j, r, cnt;
	cnt = 0;
	while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
		l->u0[ind(i, j)] = 1;
		cnt++;
	}
	printf("Loaded %d life cells.\n", cnt);
	fclose(fd);

	/* Decompozition. */
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
}

void life_save_vtk(const char *path, life_t *l)
{
	FILE *f;
	int i1, i2, j;
	f = fopen(path, "w");
	assert(f);
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2d\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
	fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
	fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
	fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);
	
	fprintf(f, "SCALARS life int 1\n");
	fprintf(f, "LOOKUP_TABLE life_table\n");
	for (i2 = 0; i2 < l->ny; i2++) {
		for (i1 = 0; i1 < l->nx; i1++) {
			fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
		}
	}
	fclose(f);
}

void life_step(life_t *l)
{
#pragma omp for 
	for (int j = l->start; j < l->end; j++) {
		for (int i = 0; i < l->nx; i++) {
			int n = 0;
			n += l->u0[ind(i+1, j)];
			n += l->u0[ind(i+1, j+1)];
			n += l->u0[ind(i,   j+1)];
			n += l->u0[ind(i-1, j)];
			n += l->u0[ind(i-1, j-1)];
			n += l->u0[ind(i,   j-1)];
			n += l->u0[ind(i-1, j+1)];
			n += l->u0[ind(i+1, j-1)];
			l->u1[ind(i,j)] = 0;
			if (n == 3 && l->u0[ind(i,j)] == 0) {
				l->u1[ind(i,j)] = 1;
			}
			if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
				l->u1[ind(i,j)] = 1;
			}
			// l->u1[ind(i,j)] = l->rank;
		}
	}
#pragma omp single
{
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}	
#pragma omp barrier
}


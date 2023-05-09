#include <iostream>
#include <math.h>
#include <vector>
#include <mpi.h>

using namespace std;

int pow2(int num) {
    return 1<<num;
}

int main(int argc, char** argv) {
    
    int num = atoi(argv[1]);
    int m = pow2(num);
    
    double t2, t1;
    
    MPI_Init(&argc, &argv);

    int rank; int size;
    vector<char> sended(m);
    vector<char> recived(m);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        for (int i = 0; i < m; i++) {
            sended[i] = 'k';
        }
    } 

    if (rank == 0) {
        t1 = MPI_Wtime();
    }
    for (int i = 0; i < 1000; i++) {
    if (rank == 1) 
        MPI_Send(&recived[0], m, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Recv(&sended[0], m, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&sended[0], m, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
    }
    if (rank == 1) 
        MPI_Recv(&sended[0], m, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank == 0) {
        t2 = MPI_Wtime();
    }

    if (rank == 0) {
        double t = (t2-t1)/2000;
        cout << t << endl;
    }
    MPI_Finalize();
    
    return 0;
}



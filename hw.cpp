#include <iostream>

//Не забываем подключить библиотеку к проекту
#include "mpi.h"

using namespace std;

int main(int argc, char **argv) {
    //Инициализируем mpi
    MPI_Init(&argc, &argv);
    
    cout << "Hello world" << endl;

    //Завершаем работу с mpi
    MPI_Finalize();

    return 0;
}
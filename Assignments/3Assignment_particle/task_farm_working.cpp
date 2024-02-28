/*
  Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
  To "execute" a task, the worker sleeps for the given number of milliseconds.
  The result of a task should be send back from the worker to the master. It
  contains the rank of the worker
*/

#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <array>
#include <vector>

// To run an MPI program we always need to include the MPI headers
#include <mpi.h>

const int NTASKS=5000;  // number of tasks
const int RANDOM_SEED=1234;


void master (int nworker) {
    std::array<int, NTASKS> task, result;

    // set up a random number generator
    std::random_device rd;
    std::default_random_engine engine;
    engine.seed(RANDOM_SEED);
    std::uniform_int_distribution<int> distribution(0, 30);

    for (int& t : task) {
        t = distribution(engine);   // set up some "tasks"
    }

    /*
    IMPLEMENT HERE THE CODE FOR THE MASTER
    ARRAY task contains tasks to be done. Send one element at a time to workers
    ARRAY result should at completion contain the ranks of the workers that did
    the corresponding tasks
    */


    int number_of_tasks_sent = 0;
    // 1. Give all workers a job
    for (int i = 1; i <= nworker; i++) {
        MPI_Send(&task[number_of_tasks_sent], 1, MPI_INT, i, number_of_tasks_sent, MPI_COMM_WORLD);  // Can be non-blocking. Request for each individual Send
        number_of_tasks_sent++;
    }

    // 2. While there are jobs left, 
    int current_result;
    int result_index;
    int worker_source;
    int send_tag;
    MPI_Status status;
    while (number_of_tasks_sent < NTASKS) {
        // Receive data and get worker rank and result index (i.e. time sent)
        MPI_Recv(&current_result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        worker_source = status.MPI_SOURCE;
        result_index = status.MPI_TAG;
        // Update result 
        result[result_index] = current_result;
        // Send message
        send_tag = number_of_tasks_sent;
        MPI_Send(&task[number_of_tasks_sent], 1, MPI_INT, worker_source, send_tag, MPI_COMM_WORLD);
        number_of_tasks_sent++;
    }

    int shutdown_task = -1;
    for (int j = 1; j <= nworker; j++) {
        // Get and store result
        MPI_Recv(&current_result, 1, MPI_INT, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        worker_source = status.MPI_SOURCE;
        result_index = status.MPI_TAG;
        result[result_index] = current_result;
        // Send shutdown 
        MPI_Send(&shutdown_task, 1, MPI_INT, worker_source, 0, MPI_COMM_WORLD);
    }

    // Print out a status on how many tasks were completed by each worker
    for (int worker=1; worker<=nworker; worker++) {
        int tasksdone = 0; int workdone = 0;
        for (int itask=0; itask<NTASKS; itask++)
        if (result[itask]==worker) {
            tasksdone++;
            workdone += task[itask];
        }
        std::cout << "Master: Worker " << worker << " solved " << tasksdone << 
                    " tasks\n";    
    }
}

// call this function to complete the task. It sleeps for task milliseconds
void task_function(int task) {
    std::this_thread::sleep_for(std::chrono::milliseconds(task));
}


void worker (int rank) {
    /*
    IMPLEMENT HERE THE CODE FOR THE WORKER
    Use a call to "task_function" to complete a task
    */

    // Get initial task to have a task to check if "received_task != -1" is true.
    int received_task;
    MPI_Status status;
    int master_rank = 0;
    MPI_Recv(&received_task, 1, MPI_INT, master_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int received_tag = status.MPI_TAG;

    while (received_task != -1) {  // -1 is shutdown signal
        task_function(received_task);
        MPI_Send(&rank, 1, MPI_INT, master_rank, received_tag, MPI_COMM_WORLD);
        
        MPI_Recv(&received_task, 1, MPI_INT, master_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        received_tag = status.MPI_TAG;
    }
}

int main(int argc, char *argv[]) {
    int nrank, rank;

    MPI_Init(&argc, &argv);                // set up MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nrank); // get the total number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get the rank of this process

    if (rank == 0)       // rank 0 is the master
        master(nrank-1); // there is nrank-1 worker processes
    else                 // ranks in [1:nrank] are workers
        worker(rank);

    MPI_Finalize();      // shutdown MPI
}

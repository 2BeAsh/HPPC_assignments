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

    // Create a vector of all available workers (their ranks). 
    // At the start, everyone is ready.
    std::vector<int> rank_of_workers_available(nworker);
    int worker_starting_rank = 1;
    for (int& work_rank : rank_of_workers_available) {
        work_rank = worker_starting_rank;
        worker_starting_rank += 1;
    }

    for (int i = 0; i < NTASKS; i++) {
        // First receive, then send - More optimal to receive first

        // Get the last available worker, 
        // then remove it from available workers (since it now receives a job)
        /*while (rank_of_workers_available.empty()) {
            wait
        }
        
        */    
        //if (!rank_of_workers_available.empty()){}
        int worker_rank = rank_of_workers_available.back();
        rank_of_workers_available.pop_back();
        
        int send_rank = 0;
        int receive_rank = worker_rank;

        MPI_Request send_req;
        MPI_Isend(&task[i], 1, MPI_INT, send_rank, 0, MPI_COMM_WORLD, &send_req);

        MPI_Request recv_req;
        MPI_Irecv(&result[i], 1, MPI_INT, receive_rank, 1, MPI_COMM_WORLD, &recv_req);

        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);




        MPI_Recv(&current_result, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Tag = 0
        result[i] = current_result;
        rank_of_workers_available.push_back(worker_rank);

        MPI_Send(&current_task, 1, MPI_INT, worker_rank, 1, MPI_COMM_WORLD);  // Tag = 1

        //MPI_Sendrecv(&current_task, 1, MPI_INT, worker_rank, 1, );
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
    MPI_Recv(&current_task, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Tag = 1
    task_function(current_task);
    current_result = rank;
    MPI_Send(&current_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // Tag = 0
    //rank_of_workers_available.push_back(rank); // OBS how do I fix? Do I need to send this as well? Moved to Master after having received?
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

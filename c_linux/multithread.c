#define _GNU_SOURCE

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> //Header file for sleep(). man 3 sleep for details. 
#include <pthread.h> 
#include <sched.h>
#include <time.h> 

#define ITERATIONS 1000000
#define TRIES 10

// A normal C function that is executed as a thread 
// when its name is specified in pthread_create() 

int stick_this_thread_to_core(int core_id) {
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   if (core_id < 0 || core_id >= num_cores)
      return -1;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self();    
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

void *overheadFun(void *vargp) 
{
	stick_this_thread_to_core(2);
	return NULL; 
} 

void *yieldFun(void *vargp) 
{
	stick_this_thread_to_core(2);
	for (int i = 0; i < ITERATIONS; ++i)
	 {
	 	pthread_yield(); 
	 } 
	return NULL; 
}

long timespec_diff(struct timespec *start, struct timespec *stop)
{
	struct timespec result;
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result.tv_sec = stop->tv_sec - start->tv_sec - 1;
        result.tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result.tv_sec = stop->tv_sec - start->tv_sec;
        result.tv_nsec = stop->tv_nsec - start->tv_nsec;
    }


    return result.tv_sec*1000000000 + result.tv_nsec;
}

float ctx_inner() 
{ 
	pthread_t thread1, thread2, thread3, thread4;
	struct timespec ts_start, ts_intermediate, ts_end; 

	clock_gettime(CLOCK_REALTIME, &ts_start);
	
	pthread_create(&thread1, NULL, overheadFun, NULL); 
	pthread_create(&thread2, NULL, overheadFun, NULL); 
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	clock_gettime(CLOCK_REALTIME, &ts_intermediate);

	pthread_create(&thread3, NULL, yieldFun, NULL); 
	pthread_create(&thread4, NULL, yieldFun, NULL); 
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL); 

	clock_gettime(CLOCK_REALTIME, &ts_end);


	long overhead = timespec_diff(&ts_start, &ts_intermediate);
	long actual = timespec_diff(&ts_intermediate, &ts_end) - overhead;
	float result = (actual*1.0)/(ITERATIONS*2.0);
	printf("%ld %ld %f\n", overhead, actual, result);
	return(result); 
}

int context_main(){
	float lat = 0.0;
	for (int i = 0; i < TRIES; ++i)
	{
		lat += ctx_inner();
	}

	printf("%f\n", lat/(TRIES*1.0));
}

int main(){
	context_main();
	return 0;
}

#define _GNU_SOURCE

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> //Header file for sleep(). man 3 sleep for details. 
#include <pthread.h> 
#include <sched.h>
#include <time.h>
#include <sys/stat.h> 
#include <fcntl.h>

#define ITERATIONS 1000
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

void	*buf;
size_t	size = 1024*1024;
size_t chunk = 64*1024;

long
bread(void* buf, long nbytes)
{
	long sum = 0;
	register long *p, *next;
	register char *end;

	p = (long*)buf;
	end = (char*)buf + nbytes;
	for (next = p + 128; (void*)next <= (void*)end; p = next, next += 128) {
		sum +=
			p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7]+
			p[8]+p[9]+p[10]+p[11]+p[12]+p[13]+p[14]+
			p[15]+p[16]+p[17]+p[18]+p[19]+p[20]+p[21]+
			p[22]+p[23]+p[24]+p[25]+p[26]+p[27]+p[28]+
			p[29]+p[30]+p[31]+p[32]+p[33]+p[34]+p[35]+
			p[36]+p[37]+p[38]+p[39]+p[40]+p[41]+p[42]+
			p[43]+p[44]+p[45]+p[46]+p[47]+p[48]+p[49]+
			p[50]+p[51]+p[52]+p[53]+p[54]+p[55]+p[56]+
			p[57]+p[58]+p[59]+p[60]+p[61]+p[62]+p[63]+
			p[64]+p[65]+p[66]+p[67]+p[68]+p[69]+p[70]+
			p[71]+p[72]+p[73]+p[74]+p[75]+p[76]+p[77]+
			p[78]+p[79]+p[80]+p[81]+p[82]+p[83]+p[84]+
			p[85]+p[86]+p[87]+p[88]+p[89]+p[90]+p[91]+
			p[92]+p[93]+p[94]+p[95]+p[96]+p[97]+p[98]+
			p[99]+p[100]+p[101]+p[102]+p[103]+p[104]+
			p[105]+p[106]+p[107]+p[108]+p[109]+p[110]+
			p[111]+p[112]+p[113]+p[114]+p[115]+p[116]+
			p[117]+p[118]+p[119]+p[120]+p[121]+p[122]+
			p[123]+p[124]+p[125]+p[126]+p[127];
	}
	for (next = p + 16; (void*)next <= (void*)end; p = next, next += 16) {
		sum +=
			p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7]+
			p[8]+p[9]+p[10]+p[11]+p[12]+p[13]+p[14]+
			p[15];
	}
	for (next = p + 1; (void*)next <= (void*)end; p = next, next++) {
		sum += *p;
	}
	return sum;
}

void doit(int fd)
{
	int	sum = 0;

	while (size >= 0) {
		if (size < chunk) chunk = size;
		if (read(fd, buf, chunk) <= 0) {
			break;
		}
		bread(buf, chunk);
		size -= chunk;
	}
}

float time_with_open_inner()
{
	char	*filename = "file1.txt";
	int	fd;
	char buf[128];
	struct timespec ts_start, ts_end;
	int iterations = ITERATIONS;
	{
		fd= open(filename, O_RDONLY);
		if(fd < 0){
			printf("Error\n");
		}
		doit(fd);
		close(fd);
	}
	clock_gettime(CLOCK_REALTIME, &ts_start);
	while (iterations-- > 0) {
		fd= open(filename, O_RDONLY);
		doit(fd);
		close(fd);
	}
	clock_gettime(CLOCK_REALTIME, &ts_end);
	long actual = timespec_diff(&ts_start, &ts_end);
	float delta_time_avg = (actual*1.0)/(ITERATIONS);
	float mb_per_sec = (size * 1.0 * 1000000000) / (1024*1024*1.0 * delta_time_avg);	// prefer this
	float kb_per_sec = (size * 1.0 * 1000000000) / (1024*1.0 * delta_time_avg);
	printf("%ld %f %f %f \n", actual, delta_time_avg, mb_per_sec, kb_per_sec);
	return(delta_time_avg); 
}

int main(){
	// context_main();
	time_with_open_inner();
	return 0;
}

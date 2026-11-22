// Pin the calling thread to one CPU of the process's allowed set.  libgomp
// (GCC's OpenMP runtime) has no omp_set_proc_bind, and without binding the
// reference solver's barrier spinning dominates the measured times.
#define _GNU_SOURCE
#include <sched.h>

void spike_pin_thread(int i) {
    cpu_set_t allowed;
    if(sched_getaffinity(0, sizeof allowed, &allowed) != 0) return;
    const int n = CPU_COUNT(&allowed);
    if(n <= 0) return;
    i %= n;
    int k = 0;
    for(int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if(!CPU_ISSET(cpu, &allowed)) continue;
        if(k++ == i) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            sched_setaffinity(0, sizeof one, &one);
            return;
        }
    }
}

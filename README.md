# TLB-pilot: Mitigating TLB Contention Attack on GPUs with Microarchitecture-Aware Scheduling

## Architecture of TLB-pilot
### demo 
A simple example for using TLB_pilot
### runtime.cuh
The library of TLB_pilot
### TLB_contention_attack.cu
A attack for GPU TLB
### malfunctioned_attack.cu
A malfunctioned attack kernel for GPU TLB

## How to use TLB-pilot
- Including `runtime.cuh` in your source codes.
- Using following APIs to deploy TLB-pilot
```
/* Returns the number of thread blocks for the "filling phase".*/
new_block_num(int orig_block_num) 

/* Sends a message to the long-running kernel. */
void kernel_splitting_send(void)

/* It blocks execution until it receive a message from a short-running kernel. */
void kernel_splitting_receive(void)

```
- You can see examples in `demo` to learn more details.

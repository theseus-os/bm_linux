# bm_linux
This repository contains basic OS microbenchmarks written in Rust. We have tried to match the LMBench version of the benchmarks as closely as possible.

## Benchmarks
The benchmarks supported, and that appear in the Theseus paper are:  
- **null syscall:** invoke getpid() syscall  
-    **context switch:** switch between two threads that continuously yield  
-    **create process:** fork + exec a "Hello, world!" application  
-    **memory map:** map, write, then unmap 4KiB pages (here we differ from the standard LMBench benchmark and use the MAP_POPULATE flag to avoid a page fault)  
-    **IPC:** 1-byte RTT using non-blocking pipes  

## Running the Benchmarks
To build the crate, and then run all the above mentioned benchmarks, run the script `run.sh`.

Otherwise print the help menu (`cargo run -- help`) for information on how to run benchmarks individually and other options.

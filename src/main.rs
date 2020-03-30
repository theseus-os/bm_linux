#![feature(asm)]
#![feature(duration_as_u128)]
extern crate libc;
extern crate hwloc;
extern crate core_affinity;
extern crate perfcnt;
extern crate memmap;
extern crate libm;
extern crate os_pipe;
extern crate mmap;
extern crate page_size;

use std::env;
use std::fs::{self, File};
use std::time::Instant;
use std::process::{self, Command, Stdio};
use std::io::{Read, Write, SeekFrom, Seek};
use std::path::Path;
use std::{thread, time};
use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc;
use std::sync::{Arc,Mutex};

use hwloc::{Topology, ObjectType, CPUBIND_THREAD, CpuSet};
use mmap::{MemoryMap,MapOption};
use libc::{c_void, size_t, c_int};

use perfcnt::{PerfCounter, AbstractPerfCounter};
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;

use memmap::MmapOptions;

use os_pipe::pipe;

#[macro_use]
mod timing;
use timing::*;

/// When we want to run the program with perf, disable this so all irrelevant timing code is excluded.
const MEASURE_CYCLE_COUNTS: bool = true;

fn print_usage(prog: &String) {
	printlninfo!("\nUsage: {} cmd", prog);
	printlninfo!("\n  availavle cmds:");
	printlninfo!("\n    null             : null syscall");
	printlninfo!("\n    ctx              : context switch");
	printlninfo!("\n    spawn            : process creation");
	printlninfo!("\n    fault			 : page fault");
	printlninfo!("\n    ipc 		     : ipc");
}


fn getpid() -> u32 { process::id() }

static mut abc: u32 = 0;

#[no_mangle]
fn empty_fn(n: u32)
{ 
	unsafe{abc = abc + n};
}

fn cpuset_for_core(topology: &Topology, idx: usize) -> CpuSet {
    let cores = (*topology).objects_with_type(&ObjectType::Core).unwrap();
    match cores.get(idx) {
        Some(val) => val.cpuset().unwrap(),
        None => panic!("No Core found with id {}", idx)
    }
}


fn do_null_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter) -> u64 {
	let mut pid = 0;
	let mut delta_cycles_avg = 0;
	let end;
	let n = 39;

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for _ in 0..ITERATIONS {
		pid = getpid();
		// unsafe{asm!("syscall" : "={rax}"(pid) : "{rax}"(n) : "rcx", "r11", "memory" : "volatile")};

	}

	if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let mut delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}

		delta_cycles_avg = delta_cycles / ITERATIONS as u64;

		printlninfo!("null_test_inner ({}/{}): {} total_cycles -> {} avg_cycles (ignore: {})", 
			th, nr, delta_cycles, delta_cycles_avg, pid);
	}

	delta_cycles_avg
}

fn do_null() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut overhead: u64 = 0;
	let mut vec = Vec::new();

	let core_ids = core_affinity::get_core_ids().unwrap();
	let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
		.for_all_pids()
		.finish()
		.expect("Could not create counter");


	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(&mut pmc);
	}

	for i in 0..TRIES {
		let lat = do_null_inner(overhead, i+1, TRIES, &mut pmc);
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	if MEASURE_CYCLE_COUNTS {
		print_stats(vec);
		let lat = tries / TRIES as u64;
		let err = (lat as f64 * THRESHOLD_ERROR_RATIO) as u64;
		if max - lat > err || lat - min > err {
			printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
		}

		printlninfo!("NULL test: {} cycles", lat);
	}
}

fn do_spawn_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter) -> Result<u64, &'static str> {
	let end;
	let mut delta_cycles_avg = 0; 


	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for _ in 0..ITERATIONS {
		let mut child = Command::new("./hello")
			.stdout(Stdio::null())
	        .spawn()
	        .expect("Cannot run hello");

	    let exit_status = child.wait().expect("Cannot join child");
	    exit_status.code();
	}
    
	if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let delta_cycles = end.expect("couldn't read counter") - overhead;
		delta_cycles_avg = delta_cycles / ITERATIONS as u64;

		printlninfo!("spawn_test_inner ({}/{}): : {} total_cycles -> {} avg_cycles", 
			th, nr, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}

// because Rust version is too slow, I double check with libc version.
fn do_spawn_inner_libc(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter) -> Result<u64, &'static str> {
	let end;
	let mut delta_cycles_avg = 0; 

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for _ in 0..ITERATIONS {
		let pid = unsafe {libc::fork()};
		match pid {
			-1 => {return Err("Cannot libc::fork()")}
			0 => {	// child
				// printlninfo!("CHILD");
				unsafe{ libc::execve("./hello".as_ptr() as *const i8, 0 as *const *const i8, 0 as *const *const i8); }
				process::exit(-1);
			}
			_ => {
				// printlninfo!("PARENT CHILD({})", pid);
				loop {
					if unsafe{ libc::wait(0 as *mut i32) } == pid {break;}
				}
				// printlninfo!("CHILD({}) is dead", pid);
			}
		}
	}
    
	if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let delta_cycles = end.expect("couldn't read counter") - overhead;
		let delta_cycles_avg = delta_cycles / ITERATIONS as u64;

		printlninfo!("spawn_test_inner (libc) ({}/{}): : {} total_cycles -> {} avg_cycles", 
			th, nr, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}

fn do_spawn(rust_only: bool) {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut overhead: u64 = 0;
	let mut vec = Vec::new();

	let core_ids = core_affinity::get_core_ids().unwrap();
	let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
		.for_all_pids()
		.finish()
		.expect("Could not create counter");

	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(&mut pmc);
	}

	for i in 0..TRIES {
		let lat = if rust_only {
			do_spawn_inner(overhead, i+1, TRIES, &mut pmc).expect("Error in spawn inner()")
		} else {
			do_spawn_inner_libc(overhead, i+1, TRIES, &mut pmc).expect("Error in spawn inner()")
		};
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	if MEASURE_CYCLE_COUNTS {
		print_stats(vec);
		let lat = tries / TRIES as u64;
		let err = (lat as f64 * THRESHOLD_ERROR_RATIO) as u64;
		if 	max - lat > err || lat - min > err {
			printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
		}

		printlninfo!("SPAWN result: {} cycles", lat);
	}
}

fn do_ctx_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter, core: core_affinity::CoreId) -> Result<u64, &'static str> {
    let mut intermediate = Ok(0);
	let end;
	let mut delta_cycles_avg = 0;

	let (tx1, rx1): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx3, rx3): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx4, rx4): (Sender<i32>, Receiver<i32>) = mpsc::channel();

    // // println!("Found {} cores.", num_cores);
    // let core_ids = core_affinity::get_core_ids().unwrap();
    let id3 = core.clone();
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

	if MEASURE_CYCLE_COUNTS{
		counter.reset();
		counter.start();
	}

    		// Each thread will send its id via the channel
        let child3 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id3);
            


            for id in 0..ITERATIONS {
            	tx3.send(id as i32).unwrap();
            	let _res = rx3.recv().unwrap();
            	// println!("thread {} send", id);
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });

        child3.join().expect("oops! the child thread panicked");

        let child4 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id4);
            

            for id in 0..ITERATIONS {
            	tx4.send(id as i32).unwrap();
            	let _res = rx4.recv().unwrap();
            	// println!("thread {} send", id);
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });

        child4.join().expect("oops! the child thread panicked");

		if MEASURE_CYCLE_COUNTS {
			intermediate = counter.read();
		}

        // Each thread will send its id via the channel
        let child1 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id1);

            for id in 0..ITERATIONS {
            	tx1.send(id as i32).unwrap();
            	// println!("send");
            	let _res = rx2.recv().unwrap();
            	// println!("thread {} send", id);
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });



        // Each thread will send its id via the channel
        let child2 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id1);

            for id in 0..ITERATIONS {
            	// println!("ready to receive");
            	let id = rx1.recv().unwrap();
            	tx2.send(id as i32).unwrap();
            	// println!("thread {} received", id);
            }
            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
        });


    child1.join().expect("oops! the child thread panicked");
    child2.join().expect("oops! the child thread panicked");

	if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let overhead_delta = intermediate.expect("couldn't read counter");
		let overhead_time = overhead_delta;
		let delta = end.expect("couldn't read counter") as u64 - overhead_delta - overhead_delta;
		let delta_cycles= delta;
		delta_cycles_avg = delta_cycles / (ITERATIONS*2) as u64;

		printlninfo!("do_ctx_inner ({}/{}): : overhead {}, {} total_cycles -> {} avg_cycles", 
			th, nr, overhead_time, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}


fn do_ctx() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut overhead = 0;
	let mut vec = Vec::new();

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc: PerfCounter = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
        .for_all_pids()
        .finish()
        .expect("Could not create counter");

	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(&mut pmc);
	}

	for i in 0..TRIES {
		let lat = do_ctx_inner(overhead, i+1, TRIES, &mut pmc, core_id.clone()).expect("Error in spawn inner()");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	if MEASURE_CYCLE_COUNTS {
		print_stats(vec);
		let lat = tries / TRIES as u64;
		let err = (lat as f64 * THRESHOLD_ERROR_RATIO) as u64;
		if 	max - lat > err || lat - min > err {
			printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
		}

		printlninfo!("CTX result: {} cycles", lat);
	}
}

fn do_ctx_yield_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter, core_id: core_affinity::CoreId) -> Result<u64, &'static str> {
	let mut intermediate = Ok(0);
	let end;
	let mut delta_cycles_avg = 0;

	let (tx1, rx1): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx3, rx3): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx4, rx4): (Sender<i32>, Receiver<i32>) = mpsc::channel();

    // let core_ids = core_affinity::get_core_ids().unwrap();
    // let id3 = core_ids[3];
    let id3 = core_id.clone();
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

    	// Each thread will send its id via the channel
        
		let child3 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel

            core_affinity::set_for_current(id3);
            
        });

        child3.join().expect("oops! the child thread panicked");

        let child4 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel

            core_affinity::set_for_current(id4);
            
        });

        child4.join().expect("oops! the child thread panicked");

		if MEASURE_CYCLE_COUNTS {
    		intermediate = counter.read();
		}

        // Each thread will send its id via the channel
        let child1 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id1);

            for id in 0..ITERATIONS {
            	thread::yield_now();
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });



        // Each thread will send its id via the channel
        let child2 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel
            core_affinity::set_for_current(id2);

            for id in 0..ITERATIONS {
            	thread::yield_now();
            }
            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
        });


    child1.join().expect("oops! the child thread panicked");
    child2.join().expect("oops! the child thread panicked");

    if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let overhead_delta = intermediate.expect("couldn't read counter");
		let overhead_time = overhead_delta;
		let delta = end.expect("couldn't read counter") - overhead_delta - overhead_delta;
		let delta_cycles= delta;
		delta_cycles_avg = delta_cycles / (ITERATIONS*2) as u64;

		printlninfo!("do_ctx_inner ({}/{}): : overhead {}, {} total_cycles -> {} avg_cycles", 
			th, nr, overhead_time, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}


fn do_ctx_yield() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut vec = Vec::new();

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc: PerfCounter = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
        .for_all_pids()
        .finish()
        .expect("Could not create counter");

	let overhead = timing_overhead_cycles(&mut pmc);
	
	for i in 0..TRIES {
		let lat = do_ctx_yield_inner(overhead, i+1, TRIES, &mut pmc, core_id).expect("Error in ctx yield inner()");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	print_stats(vec);
	let lat = tries / TRIES as u64;
	let err = (lat as f64* THRESHOLD_ERROR_RATIO) as u64;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
	}

	printlninfo!("CTX result: {} cycles", lat);
}

fn do_page_fault_inner_libc(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter) -> Result<u64, &'static str> {
	let mut delta_cycles_avg = 0;
	let n = 39;

	let len: libc::size_t = 4096;
	let prot: libc::c_int = libc::PROT_WRITE | libc::PROT_READ | libc::PROT_EXEC;
	let flags: libc::c_int = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
	let fd: libc::c_int = -1;
	let offset: libc::off_t = 0;

	let mut byte = 237;

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for _ in 0..ITERATIONS {
		let mut addr: *mut u8 = 0 as *mut u8;

		unsafe{ 
			let addr = libc::mmap(0 as *mut libc::c_void, len, prot, flags, fd, offset)  as *mut u8; 
			// unsafe {println!("addr: {:#X}, value: {}", addr as usize, *addr); }
			*addr = byte;
			libc::munmap(addr as *mut libc::c_void, len);
		}
	}

	if MEASURE_CYCLE_COUNTS {
		let end = counter.read();

		let mut delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}

		delta_cycles_avg = delta_cycles / ITERATIONS as u64;

		printlninfo!("null_test_inner ({}/{}): {} total_cycles -> {} avg_cycles", 
			th, nr, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}

fn do_page_fault_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter) -> Result<u64, &'static str> {
	let mut byte = 237;
	let mut delta_cycles_avg = 0;

	let mut mmap_options = MmapOptions::new();

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for _ in 0..ITERATIONS {
		let mut addr: *mut u8 = 0 as *mut u8;

		unsafe{ 
			let mmap = mmap_options.len(4096).map_anon().map_err(|_e| "Could not create anonymous mapping")?;
		}
	}

	if MEASURE_CYCLE_COUNTS {
		let end = counter.read();

		let mut delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}

		delta_cycles_avg = delta_cycles / ITERATIONS as u64;

		printlninfo!("null_test_inner ({}/{}): {} total_cycles -> {} avg_cycles", 
			th, nr, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}


fn do_page_fault() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut overhead: u64 = 0;
	let mut vec = Vec::new();


	let core_ids = core_affinity::get_core_ids().unwrap();
	let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
		.for_all_pids()
		.finish()
		.expect("Could not create counter");


	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(&mut pmc);
	}

	for i in 0..TRIES {
		let lat = do_page_fault_inner_libc(overhead, i+1, TRIES, &mut pmc).expect("Page Fault bm failed.");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	if MEASURE_CYCLE_COUNTS {
		print_stats(vec);
		let lat = tries / TRIES as u64;
		let err = (lat as f64 * THRESHOLD_ERROR_RATIO) as u64;
		if max - lat > err || lat - min > err {
			printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
		}

		printlninfo!("PAGE FAULT test: {} cycles", lat);
	}
}

fn do_ipc_inner(overhead: u64, th: usize, nr: usize, counter: &mut PerfCounter, core_id: core_affinity::CoreId) -> Result<u64, &'static str> {
	let mut intermediate = Ok(0);
	let end;
	let mut delta_cycles_avg = 0;

	let (mut reader1, mut writer1) = pipe().map_err(|_e| "Unable to create pipe")?;
	let (mut reader2, mut writer2) = pipe().map_err(|_e| "Unable to create pipe")?;

    let id3 = core_id.clone();
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

        
		let child3 = thread::spawn(move || {
            core_affinity::set_for_current(id3);
        });

        child3.join().expect("oops! the child thread panicked");

        let child4 = thread::spawn(move || {
            core_affinity::set_for_current(id4);
        });

        child4.join().expect("oops! the child thread panicked");

		if MEASURE_CYCLE_COUNTS {
    		intermediate = counter.read();
		}

        let child1 = thread::spawn(move || {
            core_affinity::set_for_current(id1);
			let mut val = [22];

            for id in 0..ITERATIONS {
            	writer1.write(&val).expect("unable to write to pipe");
				reader2.read(&mut val).expect("unable to write to pipe");
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });


        let child2 = thread::spawn(move || {
            core_affinity::set_for_current(id2);
			let mut val = [32];

            for id in 0..ITERATIONS {
            	let res = reader1.read(&mut val);
				// println!("reader 1: {}", res.unwrap());
				let res = writer2.write(&val);
				// println!("writer 2: {}", res.unwrap());
            }
            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
        });


    child1.join().expect("oops! the child thread panicked");
    child2.join().expect("oops! the child thread panicked");

    if MEASURE_CYCLE_COUNTS {
		end = counter.read();

		let overhead_delta = intermediate.expect("couldn't read counter");
		let overhead_time = overhead_delta;
		let delta = end.expect("couldn't read counter") - overhead_delta - overhead_delta; // twice subtracted, once because it's the starting time and once because it's the overhead.
		let delta_cycles= delta;
		delta_cycles_avg = delta_cycles / (ITERATIONS*2) as u64; // *2 for one way IPC time

		printlninfo!("do_ipc_inner ({}/{}): : overhead {}, {} total_cycles -> {} avg_cycles", 
			th, nr, overhead_time, delta_cycles, delta_cycles_avg);
	}

	Ok(delta_cycles_avg)
}

fn do_ipc() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut overhead: u64 = 0;
	let mut vec = Vec::new();

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc: PerfCounter = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
        .for_all_pids()
        .finish()
        .expect("Could not create counter");
	
	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(&mut pmc);
	}
	
	for i in 0..TRIES {
		let lat = do_ipc_inner(overhead, i+1, TRIES, &mut pmc, core_id).expect("Error in IPC inner()");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	if MEASURE_CYCLE_COUNTS {

		print_stats(vec);
		let lat = tries / TRIES as u64;
		let err = (lat as f64* THRESHOLD_ERROR_RATIO) as u64;
		if 	max - lat > err || lat - min > err {
			printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
		}

		printlninfo!("IPC result: {} cycles", lat);
	}
}

/// Evaluation of linux memory management by creating, merging and deleting virtual memory areas
/// Similar to mm_eval in Theseus
fn mm_eval(num_mappings: usize, size_of_mapping_in_pages: usize) {
	let start_vaddr: usize = 0xFFFF_0000_0000_0000; // the start of the 500th P4 (PML4) entry
	const TRIES: usize = 100; 

	let mut create_times: Vec<u64> = Vec::with_capacity(TRIES);
    let mut remap_times: Vec<u64> = Vec::with_capacity(TRIES);
    let mut unmap_times: Vec<u64> = Vec::with_capacity(TRIES);

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);

	let mut pmc: PerfCounter = Builder::from_hardware_event(Hardware::RefCPUCycles)
		.on_cpu(core_id.id as isize)
        .for_all_pids()
        .finish()
        .expect("Could not create counter");


	for i in 0..TRIES {

		// (1) create mappings
		let result_create = create_mappings(size_of_mapping_in_pages, num_mappings, &mut pmc, "LINUX CREATE MAP");
		if result_create.is_err() {
			println!("Could not create mappings");
		}
		// (2) perform remappings
		let (mut create_mappings, create_time) = result_create.unwrap();
		create_times.push(create_time);
		
		let remap_time = remap(&create_mappings, size_of_mapping_in_pages, num_mappings, &mut pmc);
		remap_times.push(remap_time);
		
		// (3) perform unmappings
		let unmap_time = unmap(&mut create_mappings, &mut pmc);	
		unmap_times.push(unmap_time);
			
	}

	println!("Create Mappings");
    print_stats(create_times);

    println!("Remap Mappings");
    print_stats(remap_times);
    
    println!("Unmap Mappings");
    print_stats(unmap_times);
}

fn create_mappings(size_in_pages: usize, num_mappings: usize, counter: &mut PerfCounter, type_of_mapping: &str) -> Result<(Vec<MemoryMap>, u64), &'static str>{
	// unsafe { libc::mlockall(libc::MCL_FUTURE); }

	let mut mmap_options = [MapOption::MapWritable]; // MapOption::MapNonStandardFlags(libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_POPULATE)];
	let mut mapped_pages: Vec<MemoryMap> = Vec::with_capacity(num_mappings);
	let size_in_bytes = size_in_pages * page_size::get();
	let mut delta_cycles = 0;
	let mut overhead = 0;

	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(counter);
	}

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}
	
	for i in 0..num_mappings {

		match MemoryMap::new(size_in_bytes, &mmap_options) {
			Ok(mapping) => {
				mapped_pages.push(mapping);
			} 
			Err(_x) => {
				return Err("Could not map page");
			}
		}
	}
	
	if MEASURE_CYCLE_COUNTS {
		let end = counter.read();

		delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}
	}
	
	Ok((mapped_pages, delta_cycles))
}

fn remap(mem_mappings: &Vec<MemoryMap>, size_in_pages: usize, num_mappings: usize, counter: &mut PerfCounter) -> u64 {
	let size_in_bytes = size_in_pages * page_size::get();
	let mut delta_cycles = 0;
	let mut overhead = 0;

	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(counter);
	}

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}

	for i in 0..num_mappings {
		let vaddr = mem_mappings[i].data();
		let len = mem_mappings[i].len();

		unsafe { 
			let err = libc::mprotect(vaddr as *mut usize as *mut c_void, len, libc::PROT_READ);
			if err != 0 {
				println!("ERROR!: mprotect not working");
			}
		}
	}
	
	if MEASURE_CYCLE_COUNTS {
		let end = counter.read();

		delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}
	}


	delta_cycles
}

fn unmap(mappings: &mut Vec<MemoryMap>, counter: &mut PerfCounter) -> u64 {
	let num_mappings = mappings.len();
	let size = mappings[0].len();
	let mut delta_cycles = 0;
	let mut overhead = 0;

	if MEASURE_CYCLE_COUNTS {
		overhead = timing_overhead_cycles(counter);
	}

	if MEASURE_CYCLE_COUNTS {
		counter.reset();
		counter.start();
	}
	for i in 0..num_mappings {
		std::mem::drop(mappings.pop());
	}

	if MEASURE_CYCLE_COUNTS {
		let end = counter.read();

		delta_cycles = end.expect("couldn't read counter");
		if delta_cycles < overhead {
			printlnwarn!("Ignore overhead for null because overhead({}) > diff({})", 
				overhead, delta_cycles);
		} else {
			delta_cycles -= overhead;
		}
	}

	delta_cycles
}

fn print_header() {
	printlninfo!("========================================");
	printlninfo!("Time unit : nano sec");
	printlninfo!("Iterations: {}", ITERATIONS);
	printlninfo!("Tries     : {}", TRIES);
	printlninfo!("Core      : Don't care");
	printlninfo!("========================================");
}

fn main() {
	let prog = env::args().nth(0).unwrap();

    if env::args().count() != 2 && env::args().count() != 4  {
    	print_usage(&prog);
    	return;
    }

    // don't need to check rq
    let path = env::current_dir().unwrap();
    printlninfo!("The current directory is {}", path.display());

    print_header();

    match env::args().nth(1).unwrap().as_str() {
    	"null" => {
    		do_null();
    	}
    	"spawn" => {
    		do_spawn(true /*rust only*/);
    	}
		"exec" => {
    		do_spawn(false /*rust only*/);
    	}
    	"ctx" => {
    		do_ctx();
    	}
    	"ctx_yield" => {
    		do_ctx_yield();
    	}
		"fault" => {
    		do_page_fault();
    	}
		"ipc" => {
    		do_ipc();
    	}
		"mm_eval" => {
			if env::args().count() == 4 {
				let num_mappings = env::args().nth(2).unwrap().parse::<usize>().ok().unwrap_or(100);
				let size_in_pages = env::args().nth(3).unwrap().parse::<usize>().ok().unwrap_or(2);
				mm_eval(num_mappings, size_in_pages);
			}
			else {
				println!("Not enough arguments for mm_eval");
			}
		}

    	_ => {printlninfo!("Unknown command: {}", env::args().nth(1).unwrap());}
    }
}

fn print_stats(vec: Vec<u64>) {
	let avg;
  	let median;
  	let perf_75;
	let perf_25;
	let min;
	let max;
	let var;
	let std_dev;

  	{ // calculate average
		let mut sum: u64 = 0;
		for x in &vec {
			sum = sum + x;
		}

		avg = sum as u64 / vec.len() as u64;
  	}

	{ // calculate median
		let mut vec2 = vec.clone();
		vec2.sort();
		let mid = vec2.len() / 2;
		let p_75 = vec2.len() *3 / 4;
		let p_25 = vec2.len() *1 / 4;

		median = vec2[mid];
		perf_25 = vec2[p_25];
		perf_75 = vec2[p_75];
		min = vec2[0];
		max = vec2[vec.len() - 1];
  	}

	{ // calculate sample variance
		let mut diff_sum: u64 = 0;
      	for x in &vec {
			if x > &avg {
				diff_sum = diff_sum + ((x-avg)*(x-avg));
			}
			else {
				diff_sum = diff_sum + ((avg - x)*(avg -x));
			}
      	}

    	var = (diff_sum) / (vec.len() as u64 - 1);
	}

	{ // calculate the standard deviation
		std_dev = libm::sqrt(var as f64);		
	}

	printlninfo!("\n  mean : {}",avg);
	printlninfo!("\n  variance  : {}",var);
	printlninfo!("\n  standard deviation  : {}",std_dev);
	printlninfo!("\n  max  : {}",max);
	printlninfo!("\n  p_50 : {}",median);
	printlninfo!("\n  p_25 : {}",perf_25);
	printlninfo!("\n  p_75 : {}",perf_75);
	printlninfo!("\n  min  : {}",min);
	printlninfo!("\n");
}
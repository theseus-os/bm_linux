#![feature(asm)]
#![feature(duration_as_u128)]
extern crate libc;
extern crate hwloc;
extern crate core_affinity;
extern crate perfcnt;
extern crate libm;
extern crate os_pipe;
extern crate mmap;
extern crate page_size;
extern crate hashbrown;

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
use std::os::unix::net::UnixStream;
use std::ffi::{CStr, CString};
use hwloc::{Topology, ObjectType, CPUBIND_THREAD, CpuSet};
use mmap::{MemoryMap,MapOption};
use libc::{c_void, size_t, c_int, c_char};

use perfcnt::{PerfCounter, AbstractPerfCounter};
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;

use os_pipe::pipe;

use hashbrown::HashMap;

use libc::{open, close, mmap, munmap, fallocate};

#[macro_use]
mod timing;
use timing::*;

fn print_usage(prog: &String) {
	printlninfo!("\nUsage: {} cmd", prog);
	printlninfo!("\n  available cmds:");
	printlninfo!("\n    null             		: null syscall");
	printlninfo!("\n    ctx              		: context switch");
	printlninfo!("\n    ctx_yield            	: context switch where tasks yield to each other");
	printlninfo!("\n    spawn            		: process creation");
	printlninfo!("\n    memory_map		 		: memory mapping");
	printlninfo!("\n    memory_map_lmbench		: memory mapping matching the lmbench version");
	printlninfo!("\n    ipc_pipe	     		: ipc_pipe");
	printlninfo!("\n    ipc_socket	     		: ipc_socket");
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

fn do_null_inner(overhead_ns: u64, th: usize, nr: usize) -> u64 {
	let start;
	let end;
	let mut pid = 0;

	start = Instant::now();
	for _ in 0..ITERATIONS {
		pid = getpid();
	}
	end = Instant::now();

	let delta = end - start;
	let mut delta_time = delta.as_nanos() as u64;
	if delta_time < overhead_ns {
		printlnwarn!("Ignore overhead for null because overhead({:.2}) > diff({:.2})", 
			overhead_ns, delta_time);
	} else {
		delta_time -= overhead_ns;
	}

	let delta_time_avg = delta_time as u64 / ITERATIONS as u64;

	printlninfo!("null_test_inner ({}/{}): {} total_ns -> {} avg_ns (ignore: {})", 
		th, nr, delta_time, delta_time_avg, pid);

	delta_time_avg
}

fn do_null() {
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let mut vec = Vec::with_capacity(TRIES);
	let overhead = timing_overhead();

	for i in 0..TRIES {
		let lat = do_null_inner(overhead, i+1, TRIES);

		tries += lat;
		vec.push(lat);

		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("NULL test: {:.2} ns", lat);
	print_stats(vec);
}

fn do_spawn_inner(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let start;
	let end;

	start = Instant::now();
	for _ in 0..ITERATIONS {
		let mut child = Command::new("./hello")
			.stdout(Stdio::null())
	        .spawn()
	        .expect("Cannot run hello");

	    let exit_status = child.wait().expect("Cannot join child");
	    exit_status.code();
	}
    end = Instant::now();

    let delta = end - start;
	let delta_time = delta.as_nanos() as u64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as u64;

    printlninfo!("spawn_test_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

// because Rust version is too slow, I double check with libc version.
fn do_spawn_inner_libc(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let start;
	let end;

	start = Instant::now();
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
    end = Instant::now();

    let delta = end - start;
	let delta_time = delta.as_nanos() as u64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as u64;

    printlninfo!("spawn_test_inner (libc) ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

fn do_spawn(rust_only: bool) {
	let mut vec = Vec::with_capacity(TRIES);
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;

	let overhead_ns = timing_overhead();
	
	for i in 0..TRIES {
		let lat = if rust_only {
			do_spawn_inner(overhead_ns, i+1, TRIES).expect("Error in spawn inner()")
		} else {
			do_spawn_inner_libc(overhead_ns, i+1, TRIES).expect("Error in spawn inner()")
		};

		tries += lat;
		vec.push(lat);

		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("SPAWN result: {:.2} ns", lat);
	print_stats(vec);
}


fn do_ctx_inner(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let start;
    let intermediate;
	let end;

	let (tx1, rx1): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx3, rx3): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx4, rx4): (Sender<i32>, Receiver<i32>) = mpsc::channel();

    // println!("Found {} cores.", num_cores);
    let core_ids = core_affinity::get_core_ids().unwrap();
    let id3 = core_ids[2];
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

		start = Instant::now();


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

    	intermediate = Instant::now();

    	// println!("Hello");


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

    end = Instant::now();

    let overhead_delta = intermediate - start;
    let overhead_time = overhead_delta.as_nanos() as u64;
    let delta = end - intermediate - overhead_delta;
	let delta_time = delta.as_nanos() as u64;
	let delta_time_avg = delta_time / (ITERATIONS*2) as u64;

    printlninfo!("do_ctx_inner ({}/{}): : overhead {:.2}, {:.2} total_time -> {:.2} avg_ns", 
		th, nr, overhead_time, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}


fn do_ctx() {
	let mut vec = Vec::with_capacity(TRIES);
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;

	let overhead_ns = timing_overhead();
	
	for i in 0..TRIES {
		let lat = do_ctx_inner(overhead_ns, i+1, TRIES).expect("Error in spawn inner()");

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("CTX result: {:.2} ns", lat);
	print_stats(vec);
}

fn do_ctx_yield_inner(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let start;
    let intermediate;
	let end;

	let (tx1, rx1): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx3, rx3): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx4, rx4): (Sender<i32>, Receiver<i32>) = mpsc::channel();

    let core_ids = core_affinity::get_core_ids().unwrap();
    let id3 = core_ids[2];
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

		start = Instant::now();


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

    	intermediate = Instant::now();

    	// println!("Hello");


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

    end = Instant::now();

    let overhead_delta = intermediate - start;
    let overhead_time = overhead_delta.as_nanos() as u64;
    let delta = end - intermediate - overhead_delta;
	let delta_time = delta.as_nanos() as u64;
	let delta_time_avg = delta_time / (ITERATIONS*2) as u64;

    printlninfo!("do_ctx_inner ({}/{}): : overhead {:.2}, {:.2} total_time -> {:.2} avg_ns", 
		th, nr, overhead_time, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}


fn do_ctx_yield() {
	let mut vec = Vec::with_capacity(TRIES);
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;

	let overhead_ns = timing_overhead();
	
	for i in 0..TRIES {
		let lat = do_ctx_yield_inner(overhead_ns, i+1, TRIES).expect("Error in spawn inner()");

		tries += lat;
		vec.push(lat);

		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("CTX result: {:.2} ns", lat);
	print_stats(vec);
}

fn do_memory_map_inner_libc(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let start;
	let end;

	let len: libc::size_t = 4096;
	let prot: libc::c_int = libc::PROT_WRITE | libc::PROT_READ;
	let flags: libc::c_int = libc::MAP_SHARED;
	let offset: libc::off_t = 0;
	let file_name = CString::new("test_file").expect("CString::new failed");

	let size: isize = 4096;
	const PSIZE: isize = 16<<10;
	const N: isize = 10;
	let c: u8 = size as u8 & 0xff;


	let fd: libc::c_int = unsafe{open(file_name.as_c_str().as_ptr(), libc::O_RDWR | libc::O_CREAT)};

	if fd < 0 {
		return Err("Could not create file");
	}
	let ret = unsafe{ fallocate(fd, 0, 0, len as i64) };
	if ret < 0 {
		return Err("could not allocate file");
	}

	start = Instant::now();

	for _ in 0..ITERATIONS {
		let mut addr: *mut u8 = 0 as *mut u8;

		unsafe{ 
			let addr = libc::mmap(0 as *mut libc::c_void, len, prot, flags, fd, offset) as *mut u8; 
			if (addr as usize) < 0 {
				return Err("mmap failed");
			}

			// for a 4 KiB mapping, this ends up writing to only the starting address
			// so we remove the loop

			// let end = addr.offset(size / N);
			// let mut p = addr;
			// while p < end {
			// 	unsafe{ p.write(c) };
			// 	p = p.offset(PSIZE);
			// }

			unsafe{ *addr = 0xFF; }

			let ret = libc::munmap(addr as *mut libc::c_void, len);
			if ret < 0 {
				return Err("munmap failed");
			}
		}
	}

    end = Instant::now();

	let ret = unsafe{close(fd)};

	if ret < 0 {
		return Err("Could not close file");
	}

    let delta = end - start;
	let delta_time = delta.as_nanos() as u64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as u64;

    printlninfo!("memory_map_test_inner (libc) ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

fn do_memory_map_lmbench() {
	let mut vec = Vec::with_capacity(TRIES);
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let overhead = timing_overhead();

	for i in 0..TRIES {
		let lat = do_memory_map_inner_libc(overhead, i+1, TRIES).expect("memory map bm failed.");
		
		vec.push(lat);
		tries += lat;
		
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	print_stats(vec);
	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
	}

	printlninfo!("MEMORY MAP LMBENCH test: {:.2} ns", lat);
}


fn do_memory_map_inner(overhead_ns: u64, th: usize, nr: usize) -> Result<u64, &'static str> {
    let size_in_bytes = 4096;
	let start;
	let end;

	let mut mmap_options = [MapOption::MapWritable, MapOption::MapNonStandardFlags(libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_POPULATE)];

	start = Instant::now();

	for _ in 0..ITERATIONS {
		let mp = match MemoryMap::new(size_in_bytes, &mmap_options) {
			Ok(mapping) => {
				mapping
			} 
			Err(_x) => {
				return Err("Could not map page");
			}
		};
		// Write to the first byte like lmbench
		unsafe{ *(mp.data())= 0xFF; }

		drop(mp);
	}
    end = Instant::now();

    let delta = end - start;
	let delta_time = delta.as_nanos() as u64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as u64;

    printlninfo!("memory_map_test_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)

}


fn do_memory_map() {
	let mut vec = Vec::with_capacity(TRIES);
	let mut tries: u64 = 0;
	let mut max: u64 = core::u64::MIN;
	let mut min: u64 = core::u64::MAX;
	let overhead = timing_overhead();

	for i in 0..TRIES {
		let lat = do_memory_map_inner(overhead, i+1, TRIES).expect("Page Fault bm failed.");
		
		vec.push(lat);
		tries += lat;
		
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	print_stats(vec);
	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
	}

	printlninfo!("MEMORY MAP test: {:.2} ns", lat);
}


fn do_ipc_pipe_inner(th: usize, nr: usize, core_id: core_affinity::CoreId) -> Result<u64, &'static str> {
	let start;
	let end;
	let intermediate;

	let (mut reader1, mut writer1) = pipe().map_err(|_e| "Unable to create pipe")?;
	let (mut reader2, mut writer2) = pipe().map_err(|_e| "Unable to create pipe")?;

    let id3 = core_id.clone();
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

    start = Instant::now();

		let child3 = thread::spawn(move || {
            // core_affinity::set_for_current(id3);
        });

        child3.join().expect("oops! the child thread panicked");

        let child4 = thread::spawn(move || {
            // core_affinity::set_for_current(id4);
        });

        child4.join().expect("oops! the child thread panicked");

	intermediate = Instant::now();

        let child1 = thread::spawn(move || {
            // core_affinity::set_for_current(id1);
			let mut val = [0];

            for id in 0..ITERATIONS {
            	writer1.write(&val).expect("unable to write to pipe");
				reader2.read(&mut val).expect("unable to read from pipe");
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });


        let child2 = thread::spawn(move || {
            // core_affinity::set_for_current(id2);
			let mut val = [0];

            for id in 0..ITERATIONS {
            	reader1.read(&mut val).expect("unable to write to pipe");
				writer2.write(&val).expect("unable to read from pipe");
            }
            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
        });


    child1.join().expect("oops! the child thread panicked");
    child2.join().expect("oops! the child thread panicked");

	end = Instant::now();

    let overhead_delta = intermediate - start;
    let overhead_time = overhead_delta.as_nanos() as u64;
    let delta = end - intermediate - overhead_delta;
	let delta_time = delta.as_nanos() as u64;
	let delta_time_avg = delta_time / ITERATIONS as u64; //*2 for 1 way IPC time

    printlninfo!("do_ipc_pipe_inner ({}/{}): : overhead {:.2}, {:.2} total_time -> {:.2} avg_ns", 
		th, nr, overhead_time, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

fn do_ipc_pipe() {
	let mut tries = 0;
	let mut max = core::u64::MIN;
	let mut min = core::u64::MAX;
	let mut vec = Vec::with_capacity(TRIES);

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);
	
	for i in 0..TRIES {
		let lat = do_ipc_pipe_inner(i+1, TRIES, core_id).expect("Error in IPC inner()");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}


	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
	}

	printlninfo!("IPC PIPE Round Trip Time: {} ns", lat);
	print_stats(vec);
}


fn do_ipc_socket_inner(th: usize, nr: usize, core_id: core_affinity::CoreId) -> Result<u64, &'static str> {
	let start;
	let end;
	let intermediate;

	// Create socket pair
    let (mut sock1, mut sock2) = match UnixStream::pair() {
        Err(_) => return Err("Could not create socket"),
        Ok((s1, s2)) => (s1,s2),
    };


    let id3 = core_id.clone();
    let id4 = id3.clone();
    let id2 = id3.clone();
    let id1 = id3.clone();

    start = Instant::now();

		let child3 = thread::spawn(move || {
            // core_affinity::set_for_current(id3);
        });

        child3.join().expect("oops! the child thread panicked");

        let child4 = thread::spawn(move || {
            // core_affinity::set_for_current(id4);
        });

        child4.join().expect("oops! the child thread panicked");

	intermediate = Instant::now();

        let child1 = thread::spawn(move || {
            // core_affinity::set_for_current(id1);
			let mut val = [22];

            for id in 0..ITERATIONS {
            	sock1.write(&val).expect("unable to write to socket");
				sock1.read(&mut val).expect("unable to read from socket");
            }

            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
            
        });


        let child2 = thread::spawn(move || {
            // core_affinity::set_for_current(id2);
			let mut val = [32];

            for id in 0..ITERATIONS {
            	let res = sock2.read(&mut val).expect("unable to write to socket");
				let res = sock2.write(&val).expect("unable to read from socket");
            }
            // Sending is a non-blocking operation, the thread will continue
            // immediately after sending its message
        });


    child1.join().expect("oops! the child thread panicked");
    child2.join().expect("oops! the child thread panicked");

	end = Instant::now();

    let overhead_delta = intermediate - start;
    let overhead_time = overhead_delta.as_nanos() as u64;
    let delta = end - intermediate - overhead_delta;
	let delta_time = delta.as_nanos() as u64;
	let delta_time_avg = delta_time / ITERATIONS as u64; 

    printlninfo!("do_ipc_socket_inner ({}/{}): : overhead {:.2}, {:.2} total_time -> {:.2} avg_ns", 
		th, nr, overhead_time, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

fn do_ipc_socket() {
	let mut tries = 0;
	let mut max = core::u64::MIN;
	let mut min = core::u64::MAX;
	let mut vec = Vec::with_capacity(TRIES);

	let core_ids = core_affinity::get_core_ids().unwrap();
    let core_id = core_ids[2];
	core_affinity::set_for_current(core_id);
	
	for i in 0..TRIES {
		let lat = do_ipc_socket_inner(i+1, TRIES, core_id).expect("Error in IPC inner()");
		vec.push(lat);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}


	let lat = tries / TRIES as u64;
	// We expect the maximum and minimum to be within 10*THRESHOLD_ERROR_RATIO % of the mean value
	let err = (lat * 10 * THRESHOLD_ERROR_RATIO) / 100;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {}, max {},  min {})", lat, max, min);
	}

	printlninfo!("IPC SOCKET Round Trip Time: {} ns", lat);
	print_stats(vec);
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
		"memory_map" => {
    		do_memory_map();
    	}
		"memory_map_lmbench" => {
    		do_memory_map_lmbench();
    	}
		"ipc_pipe" => {
    		do_ipc_pipe();
    	}
		"ipc_socket" => {
    		do_ipc_socket();
    	}
    	_ => {printlninfo!("Unknown command: {}", env::args().nth(1).unwrap());}
    }
}

fn print_stats(vec: Vec<u64>) {
	let mean;
  	let median;
	let mode;
  	let p_75;
	let p_25;
	let min;
	let max;
	let var;
    let std_dev;

	if vec.is_empty() {
		return;
	}

	let len = vec.len();

  	{ // calculate average
		let sum: u64 = vec.iter().sum();
		mean = sum as f64 / len as f64;
  	}

	{ // calculate median
		let mut vec2 = vec.clone();
		vec2.sort();
		let mid = len / 2;
		let i_75 = len * 3 / 4;
		let i_25 = len * 1 / 4;

		median = vec2[mid];
		p_25 = vec2[i_25];
		p_75 = vec2[i_75];
		min = vec2[0];
		max = vec2[len - 1];
  	}

	{ // calculate sample variance
		let mut diff_sum: f64 = 0.0;
      	for val in &vec {
			let x = *val as f64; 
			if x > mean {
				diff_sum = diff_sum + ((x - mean)*(x - mean));
			}
			else {
				diff_sum = diff_sum + ((mean - x)*(mean - x));
			}
      	}

    	var = (diff_sum) / (len as f64);
        std_dev = libm::sqrt(var);
	}

	{ // calculate mode
		let mut values: HashMap<u64,usize> = HashMap::with_capacity(len);
		for val in &vec {
			values.entry(*val).and_modify(|v| {*v += 1}).or_insert(1);
		}
		mode = *values.iter().max_by(|(_k1,v1), (_k2,v2)| v1.cmp(v2)).unwrap().0; // safe to call unwrap since we've already checked if the vector is empty
	}

	printlninfo!("\n  min  		: {}",min);
	printlninfo!("\n  p_25 		: {}",p_25);
	printlninfo!("\n  median 	: {}",median);
	printlninfo!("\n  p_75 		: {}",p_75);
	printlninfo!("\n  max  		: {}",max);
	printlninfo!("\n  mode 		: {}",mode);
	printlninfo!("\n  mean 		: {}",mean);
	printlninfo!("\n  standard deviation  : {}",std_dev);

	printlninfo!("\n");
}
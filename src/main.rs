#![feature(duration_as_u128)]
extern crate libc;
extern crate hwloc;


use std::env;
use std::fs::{self, File};
use std::time::Instant;
use std::process::{self, Command, Stdio};
use std::io::{Read, Write, SeekFrom, Seek};
use std::path::Path;
use std::{thread, time};

use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc;
use hwloc::{Topology, ObjectType, CPUBIND_THREAD, CpuSet};
use std::sync::{Arc,Mutex};

// const ITERATIONS: usize = 1_000_000;
const ITERATIONS: usize = 1_000;
const TRIES: usize = 10;
const THRESHOLD_ERROR_RATIO: f64 = 0.1;
const MB_IN_KB: usize = 1024;
const MB: usize = 1024 * 1024;
const KB: usize = 1024;
const SEC_TO_NANO: f64 = 1_000_000_000.0;


// const T_UNIT: &str = "nano sec";

// don't change it.
const READ_BUF_SIZE: usize = 64*1024;
const WRITE_BUF_SIZE: usize = 1024*1024;
const WRITE_BUF: [u8; WRITE_BUF_SIZE] = [65; WRITE_BUF_SIZE];

macro_rules! printlninfo {
    ($fmt:expr) => (println!(concat!("BM-INFO: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-INFO: ", $fmt), $($arg)*));
}

macro_rules! printlnwarn {
    ($fmt:expr) => (println!(concat!("BM-WARN: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-WARN: ", $fmt), $($arg)*));
}


fn print_usage(prog: &String) {
	printlninfo!("\nUsage: {} cmd", prog);
	printlninfo!("\n  availavle cmds:");
	printlninfo!("\n    null             : null syscall");
	printlninfo!("\n    spawn            : process creation");
	printlninfo!("\n    fs_read_with_open: file read including open");
	printlninfo!("\n    fs_read_only     : file read");
	printlninfo!("\n    fs_create        : file create + del");
}

// overhead is TIME
fn timing_overhead_inner(th: usize, nr: usize) -> f64 {
	let mut temp;
	let start;
	let end;

	temp = Instant::now();

	start = Instant::now();
	for _ in 0..ITERATIONS {
		temp = Instant::now();
	}
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.as_nanos() as f64;
	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("t_overhead_inner ({}/{}): {} total -> {:.2} avg_ns (ignore: {})", 
		th, nr, delta_time, delta_time_avg, temp.elapsed().as_nanos());

	delta_time_avg
}

// overhead is TIME
fn timing_overhead() -> f64 {
	let mut tries: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;

	for i in 0..TRIES {
		let overhead = timing_overhead_inner(i+1, TRIES);
		tries += overhead;
		if overhead > max {max = overhead;}
		if overhead < min {min = overhead;}
	}

	let overhead = tries / TRIES as f64;
	let err = overhead * THRESHOLD_ERROR_RATIO;
	if 	max - overhead > err || overhead - min > err {
		printlnwarn!("timing_overhead diff is too big: {:.2} ({:.2} - {:.2}) ns", max-min, max, min);
	}

	printlninfo!("Timing overhead: {} ns\n\n", overhead);

	overhead
}


fn getpid() -> u32 { process::id() }

fn do_null_inner(overhead_ns: f64, th: usize, nr: usize) -> f64 {
	let start;
	let end;
	let mut pid = 0;

	start = Instant::now();
	for _ in 0..ITERATIONS {
		pid = getpid();
	}
	end = Instant::now();

	let delta = end - start;
	let mut delta_time = delta.as_nanos() as f64;
	if delta_time < overhead_ns {
		printlnwarn!("Ignore overhead for null because overhead({:.2}) > diff({:.2})", 
			overhead_ns, delta_time);
	} else {
		delta_time -= overhead_ns;
	}

	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("null_test_inner ({}/{}): {} total_ns -> {} avg_ns (ignore: {})", 
		th, nr, delta_time, delta_time_avg, pid);

	delta_time_avg
}

fn do_null() {
	let mut tries: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;
	let overhead = timing_overhead();

	for i in 0..TRIES {
		let lat = do_null_inner(overhead, i+1, TRIES);

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as f64;
	let err = lat * THRESHOLD_ERROR_RATIO;
	if max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("NULL test: {:.2} ns", lat);
}

fn do_spawn_inner(overhead_ns: f64, th: usize, nr: usize) -> Result<f64, &'static str> {
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
	let delta_time = delta.as_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;

    printlninfo!("spawn_test_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

// because Rust version is too slow, I double check with libc version.
fn do_spawn_inner_libc(overhead_ns: f64, th: usize, nr: usize) -> Result<f64, &'static str> {
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
	let delta_time = delta.as_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;

    printlninfo!("spawn_test_inner (libc) ({}/{}): : {:.2} total_time -> {:.2} avg_ns", 
		th, nr, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}

fn do_spawn(rust_only: bool) {
	let mut tries: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;

	let overhead_ns = timing_overhead();
	
	for i in 0..TRIES {
		let lat = if rust_only {
			do_spawn_inner(overhead_ns, i+1, TRIES).expect("Error in spawn inner()")
		} else {
			do_spawn_inner_libc(overhead_ns, i+1, TRIES).expect("Error in spawn inner()")
		};

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as f64;
	let err = lat * THRESHOLD_ERROR_RATIO;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("SPAWN result: {:.2} ns", lat);
}

fn cpuset_for_core(topology: &Topology, idx: usize) -> CpuSet {
    let cores = (*topology).objects_with_type(&ObjectType::Core).unwrap();
    match cores.get(idx) {
        Some(val) => val.cpuset().unwrap(),
        None => panic!("No Core found with id {}", idx)
    }
}


fn do_ctx_inner(overhead_ns: f64, th: usize, nr: usize) -> Result<f64, &'static str> {
    let start;
    let intermediate;
	let end;

	let (tx1, rx1): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx3, rx3): (Sender<i32>, Receiver<i32>) = mpsc::channel();
    let (tx4, rx4): (Sender<i32>, Receiver<i32>) = mpsc::channel();

    let topo = Arc::new(Mutex::new(Topology::new()));

    let num_cores = {
        let topo_rc = topo.clone();
        let topo_locked = topo_rc.lock().unwrap();
        (*topo_locked).objects_with_type(&ObjectType::Core).unwrap().len()
    };
    // println!("Found {} cores.", num_cores);

    	let child_topo3 = topo.clone();
    	let child_topo4 = topo.clone();
    	let child_topo1 = topo.clone();
    	let child_topo2 = topo.clone();


		start = Instant::now();


    		// Each thread will send its id via the channel
        let child3 = thread::spawn(move || {
            // The thread takes ownership over `thread_tx`
            // Each thread queues a message in the channel

            let tid = unsafe { libc::pthread_self() };
            let mut locked_topo = child_topo3.lock().unwrap();
            // let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
            let bind_to = cpuset_for_core(&*locked_topo, num_cores - 1);
            locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD).unwrap();
            // let after = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
            // println!("Thread {}: Before {:?}, After {:?}", 3 , before, after);


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

            let tid = unsafe { libc::pthread_self() };
            let mut locked_topo = child_topo4.lock().unwrap();
            // let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
            let bind_to = cpuset_for_core(&*locked_topo, num_cores - 1);
            locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD).unwrap();

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
            {
            	let tid = unsafe { libc::pthread_self() };
            	let mut locked_topo = child_topo1.lock().unwrap();
            	// let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
            	let bind_to = cpuset_for_core(&*locked_topo, num_cores - 1);
            	locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD).unwrap();
            }

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
            {
            	let tid = unsafe { libc::pthread_self() };
            	let mut locked_topo = child_topo2.lock().unwrap();
            	// let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
            	let bind_to = cpuset_for_core(&*locked_topo, num_cores - 1);
            	locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD).unwrap();
            }

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
    let overhead_time = overhead_delta.as_nanos() as f64;
    let delta = end - intermediate - overhead_delta;
	let delta_time = delta.as_nanos() as f64;
	let delta_time_avg = delta_time / (ITERATIONS*2) as f64;

    printlninfo!("do_ctx_inner ({}/{}): : overhead {:.2}, {:.2} total_time -> {:.2} avg_ns", 
		th, nr, overhead_time, delta_time, delta_time_avg);

	Ok(delta_time_avg)
}


fn do_ctx() {
	let mut tries: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;

	let overhead_ns = timing_overhead();
	
	for i in 0..TRIES {
		let lat = do_ctx_inner(overhead_ns, i+1, TRIES).expect("Error in spawn inner()");

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as f64;
	let err = lat * THRESHOLD_ERROR_RATIO;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("benchmark error is too big: (avg {:.2}, max {:.2},  min {:.2})", lat, max, min);
	}

	printlninfo!("SPAWN result: {:.2} ns", lat);
}

fn do_fs_read_with_open_inner(filename: &str, overhead_ns: f64, th: usize, nr: usize) -> Result<(f64, f64, f64), &'static str> {
	let start;
	let end;
	let mut dummy_sum: u64 = 0;
	let mut buf = vec![0; READ_BUF_SIZE];
	let size = fs::metadata(filename).expect("Cannot stat the file").len() as i64;
	let mut unread_size = size;

	if unread_size % READ_BUF_SIZE as i64 != 0 {
		return Err("File size is not alligned");
	}

	start = Instant::now();
	for _ in 0..ITERATIONS 	{
		let mut file = File::open(filename).expect("Cannot stat the file");
		unread_size = size;
    	while unread_size > 0 {	// now read()
        	file.read_exact(&mut buf).expect("Cannot read");
			unread_size -= READ_BUF_SIZE as i64;

			// LMbench based on C does the magic to cast a type from char to int
			// But, we dont' have the luxury with type-safe Rust, so we do...
			dummy_sum += buf.iter().fold(0 as u64, |acc, &x| acc + x as u64);
    	}
	}
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.as_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;
	// let mb_per_sec = size as f64 / MB as f64 / (delta_time_avg / SEC_TO_NANO);
	let mb_per_sec = (size as f64 * SEC_TO_NANO) / (MB as f64 * delta_time_avg);	// prefer this
	let kb_per_sec = (size as f64 * SEC_TO_NANO) / (KB as f64 * delta_time_avg);

	printlninfo!("read_with_open_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns || {:.3} MB/sec {:.3} KB/sec (ignore: {})",
		th, nr, delta_time, delta_time_avg, mb_per_sec, kb_per_sec, dummy_sum);

	Ok((delta_time_avg, mb_per_sec, kb_per_sec))
}

// return: (time, MB/sec)
fn do_fs_read_only_inner(filename: &str, overhead_ns: f64, th: usize, nr: usize) -> Result<(f64, f64, f64), &'static str> {
	let start;
	let end;
	let mut dummy_sum: u64 = 0;
	let mut buf = vec![0 as u8; READ_BUF_SIZE];
	let size = fs::metadata(filename).expect("Cannot stat the file").len() as i64;
	let mut unread_size = size;

	if unread_size % READ_BUF_SIZE as i64 != 0 {
		return Err("File size is not alligned");
	}

	let mut file = File::open(filename).expect("Cannot stat the file");

	start = Instant::now();
	for _ in 0..ITERATIONS 	{
		file.seek(SeekFrom::Start(0)).expect("Cannot seek");
		unread_size = size;
    	while unread_size > 0 {	// now read()
        	file.read_exact(&mut buf).expect("Cannot read");
			unread_size -= READ_BUF_SIZE as i64;
			
			// LMbench based on C does the magic to cast a type from char to int
			// But, we dont' have the luxury with type-safe Rust, so we do...
			dummy_sum += buf.iter().fold(0 as u64, |acc, &x| acc + x as u64);	
    	}
	}	// for
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.as_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;
	// let naive_mb_per_sec = (size as f64 / MB as f64) / (delta_time_avg / SEC_TO_NANO);
	let mb_per_sec = (size as f64 * SEC_TO_NANO) / (MB as f64 * delta_time_avg);	// prefer this
	let kb_per_sec = (size as f64 * SEC_TO_NANO) / (KB as f64 * delta_time_avg);

	printlninfo!("read_only_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns || {:.3} MB/sec {:.3} KB/sec (ignore: {}) ",
		th, nr, delta_time, delta_time_avg, mb_per_sec, kb_per_sec, dummy_sum);

	Ok((delta_time_avg, mb_per_sec, kb_per_sec))
}

fn mk_tmp_file(filename: &str, sz: usize) -> Result<(), &'static str> {
	if sz > WRITE_BUF_SIZE {
		return Err("Cannot test because the file size is too big");
	}

	let mut file = File::create(filename).expect("Cannot create the file");

	// let mut output = String::new();
	// for i in 0..sz-1 {
	// 	output.push((i as u8 % 10 + 48) as char);
	// }
	// output.push('!'); // my magic char for the last byte

	// file.write_all(output.as_bytes()).expect("File cannot be created.");
	file.write_all(&WRITE_BUF[0..sz]).expect("File cannot be created.");
	// printlninfo!("{} is created.", filename);
	Ok(())
}

fn do_fs_read_with_size(overhead_ns: f64, fsize_kb: usize, with_open: bool) {
	let mut tries: f64 = 0.0;
	let mut tries_mb: f64 = 0.0;
	let mut tries_kb: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;

	let filename = format!("./tmp_{}k.txt", fsize_kb);
	// printlninfo!("Creating {} KB or {} B", fsize_kb, fsize_kb*1024);
	mk_tmp_file(&filename, fsize_kb*1024).expect("Cannot create a file");

	for i in 0..TRIES {
		let (lat, tput_mb, tput_kb) = if with_open {
			do_fs_read_with_open_inner(&filename, overhead_ns, i+1, TRIES).expect("Error in read_open inner()")
		} else {
			do_fs_read_only_inner(&filename, overhead_ns, i+1, TRIES).expect("Error in read_only inner()")
		};

		tries += lat;
		tries_mb += tput_mb;
		tries_kb += tput_kb;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as f64;
	let tput_mb = tries_mb / TRIES as f64;
	let tput_kb = tries_kb / TRIES as f64;
	let err = lat * THRESHOLD_ERROR_RATIO;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("test diff is too big: {} ({} - {}) ns", max-min, max, min);
	}

	// printlninfo!("{} for {} KB: {} ns", if with_open {"READ WITH OPEN"} else {"READ ONLY"}, fsize_kb, lat);
	printlninfo!("{} for {} KB: {} ns, {} MB/sec, {} KB/sec", 
		if with_open {"READ WITH OPEN"} else {"READ ONLY"}, 
		fsize_kb, lat, tput_mb, tput_kb);
}

fn do_fs_read(with_open: bool) {
	printlninfo!("File size     : {:4} KB", MB_IN_KB);
	printlninfo!("Read buf size : {:4} KB", READ_BUF_SIZE / 1024);
	printlninfo!("========================================");

	let overhead_ns = timing_overhead();

	do_fs_read_with_size(overhead_ns, MB_IN_KB, with_open);
}

fn del_or_err(filename: &str) -> Result<(), &'static str> {
	let path = Path::new(filename);
	if path.exists() {
		fs::remove_file(path);
	}

	Ok(())
}

fn do_fs_create_del_inner(fsize_b: usize, overhead_ns: f64) -> Result<(), &'static str> {
	let mut filenames = vec!["".to_string(); ITERATIONS];
	let pid = getpid();
	let start_create;
	let end_create;
	let start_del;
	let end_del;


	// populate filenames
	for i in 0..ITERATIONS {
		filenames[i] = format!("tmp_{}_{}_{}.txt", pid, fsize_b, i);
	}

	// check if we have enough data to write. We use just const data to avoid unnecessary overhead
	if fsize_b > WRITE_BUF_SIZE {
		// don't put this into mk_tmp_file() or the loop below
		// mk_tmp_file() and the loop below must be minimal for create/del benchmark
		return Err("Cannot test because the file size is too big");
	}

	// delete existing files. To make sure that the file creation below succeeds.
	for filename in &filenames {
		del_or_err(filename).expect("Cannot continue the test. We need 'delete()'.");
	}

	// create
	start_create = Instant::now();
	for filename in &filenames {
		// checking if filename exists is done above
		// here, we only create files

		// we don't use mk_tmp_file() intentionally.
		File::create(filename).expect("Cannot create the file")
			.write_all(&WRITE_BUF[0..fsize_b]).expect("File cannot be created.");
	}
	end_create = Instant::now();

	// delete
	start_del = Instant::now();
	for filename in filenames {
		fs::remove_file(filename);
	}
	end_del = Instant::now();

	let delta_create = end_create - start_create;
	let delta_time_create = delta_create.as_nanos() as f64 - overhead_ns;
	let files_per_time_create = (ITERATIONS * ITERATIONS) as f64 * SEC_TO_NANO / delta_time_create;

	let delta_del = end_del - start_del;
	let delta_time_del = delta_del.as_nanos() as f64 - overhead_ns;
	let files_per_time_del = (ITERATIONS * ITERATIONS) as f64 * SEC_TO_NANO / delta_time_del;

	printlninfo!("{:8}    {:9}    {:16.2}    {:16.2}", 
		fsize_b/KB as usize, ITERATIONS, files_per_time_create, files_per_time_del);
	Ok(())
}

fn do_fs_create_del() {
	// let	fsizes_b = [0 as usize, 1024, 4096, 10*1024];	// Theseus thinks creating an empty file is stupid (for memfs)
	let	fsizes_b = [1024_usize, 4096, 10*1024];

	let overhead_ns = timing_overhead();

	printlninfo!("SIZE(KB)    Iteration    created(files/s)    deleted(files/s)");
	// printlninfo!("SIZE(KB)    Iteration    created(files/s)");
	for fsize_b in fsizes_b.iter() {
		do_fs_create_del_inner(*fsize_b, overhead_ns).expect("Cannot test File Create & Del");
	}
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

    if env::args().count() != 2 {
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
    	"fs_read_with_open" | "fs1" => {
    		do_fs_read(true /*with_open*/);
    	}
    	"fs_read_only" | "fs2" => {
    		do_fs_read(false /*with_open*/);
    	}
    	"fs_create" | "fs3" => {
    		do_fs_create_del();
    	}
    	"ctx" | "fs3" => {
    		do_ctx();
    	}
    	"exec" => {
    		do_spawn(false /*rust only*/);
    	}
    	_ => {printlninfo!("Unknown command: {}", env::args().nth(1).unwrap());}
    }
}

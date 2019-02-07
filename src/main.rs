extern crate libc;

use std::env;
use std::fs::{self, File};
use std::time::Instant;
use std::process::{self, Command, Stdio};
use std::io::{Read, Write, SeekFrom, Seek};

// const ITERATIONS: usize = 1_000_000;
const ITERATIONS: usize = 1_000;
const TRIES: usize = 10;
const THRESHOLD_ERROR_RATIO: f64 = 0.1;

// const T_UNIT: &str = "nano sec";

macro_rules! printlninfo {
    ($fmt:expr) => (println!(concat!("BM-INFO: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-INFO: ", $fmt), $($arg)*));
}

macro_rules! printlnwarn {
    ($fmt:expr) => (println!(concat!("BM-WARN: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-WARN: ", $fmt), $($arg)*));
}


// don't change it. 
const READ_BUF_SIZE: usize = 64*1024;

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
	let delta_time = delta.subsec_nanos() as f64;
	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("t_overhead_inner ({}/{}): {} total -> {:.2} avg_ns (ignore: {})", 
		th, nr, delta_time, delta_time_avg, temp.elapsed().subsec_nanos());

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
	let mut delta_time = delta.subsec_nanos() as f64;
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
	let delta_time = delta.subsec_nanos() as f64 - overhead_ns;
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
	let delta_time = delta.subsec_nanos() as f64 - overhead_ns;
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

fn do_fs_read_with_open_inner(filename: &str, overhead_ns: f64, th: usize, nr: usize) -> Result<f64, &'static str> {
	let start;
	let end;
	let mut dummy_sum: u64 = 0;
	let mut buf = vec![0; READ_BUF_SIZE];
	let mut unread_size = fs::metadata(filename).expect("Cannot stat the file").len() as i64;

	if unread_size % READ_BUF_SIZE as i64 != 0 {
		return Err("File size is not alligned");
	}

	start = Instant::now();
	for _ in 0..ITERATIONS 	{
		let mut file = File::open(filename).expect("Cannot stat the file");

    	while unread_size > 0 {	// now read()
        	let nr_read = file.read_to_end(&mut buf).expect("Cannot read");
			unread_size -= nr_read as i64;
			dummy_sum += buf.iter().fold(0 as u64, |acc, &x| acc + x as u64);
    	}
	}
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.subsec_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("read_with_open_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns (ignore: {})",
		th, nr, delta_time, delta_time_avg, dummy_sum);

	Ok(delta_time_avg)
}

fn do_fs_read_only_inner(filename: &str, overhead_ns: f64, th: usize, nr: usize) -> Result<f64, &'static str> {
	let start;
	let end;
	let mut dummy_sum: u64 = 0;
	let mut buf = vec![0; READ_BUF_SIZE];
	let mut unread_size = fs::metadata(filename).expect("Cannot stat the file").len() as i64;

	if unread_size % READ_BUF_SIZE as i64 != 0 {
		return Err("File size is not alligned");
	}

	let mut file = File::open(filename).expect("Cannot stat the file");

	start = Instant::now();
	for _ in 0..ITERATIONS 	{
		file.seek(SeekFrom::Start(0)).expect("Cannot seek");
    	while unread_size > 0 {	// now read()
        	let nr_read = file.read_to_end(&mut buf).expect("Cannot read");
			unread_size -= nr_read as i64;
			dummy_sum += buf.iter().fold(0 as u64, |acc, &x| acc + x as u64);
    	}
	}	// for
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.subsec_nanos() as f64 - overhead_ns;
	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("read_only_inner ({}/{}): : {:.2} total_time -> {:.2} avg_ns (ignore: {})",
		th, nr, delta_time, delta_time_avg, dummy_sum);

	Ok(delta_time_avg)
}

fn mk_tmp_file(filename: &str, sz: usize) -> Result<(), &'static str> {
	let mut file = File::create(filename).expect("Cannot create the file");

	let mut output = String::new();
	for i in 0..sz-1 {
		output.push((i as u8 % 10 + 48) as char);
	}
	output.push('!'); // my magic char for the last byte

	file.write_all(output.as_bytes()).expect("File cannot be created.");
	printlninfo!("{} is created.", filename);
	Ok(())
}

fn do_fs_read_with_size(overhead_ns: f64, fsize_kb: usize, with_open: bool) {
	let mut tries: f64 = 0.0;
	let mut max: f64 = core::f64::MIN;
	let mut min: f64 = core::f64::MAX;

	let filename = format!("./tmp_{}k.txt", fsize_kb);
	mk_tmp_file(&filename, fsize_kb*1024).expect("Cannot create a file");

	for i in 0..TRIES {
		let lat = if with_open {
			do_fs_read_with_open_inner(&filename, overhead_ns, i+1, TRIES).expect("Error in read_open inner()")
		} else {
			do_fs_read_only_inner(&filename, overhead_ns, i+1, TRIES).expect("Error in read_only inner()")
		};

		tries += lat;
		if lat > max {max = lat;}
		if lat < min {min = lat;}
	}

	let lat = tries / TRIES as f64;
	let err = lat * THRESHOLD_ERROR_RATIO;
	if 	max - lat > err || lat - min > err {
		printlnwarn!("test diff is too big: {} ({} - {}) ns", max-min, max, min);
	}

	printlninfo!("{} for {} KB: {} ns", if with_open {"READ WITH OPEN"} else {"READ ONLY"}, fsize_kb, lat);
}

fn do_fs_read(with_open: bool) {
	let overhead_ct = timing_overhead();

	// min: 64K
	for i in [64, 128, 256, 512, 1024].iter() {
        do_fs_read_with_size(overhead_ct, *i, with_open);
    }
}

fn do_fs_create_del() {
	printlninfo!("Cannot test without MemFile::Delete()...");
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
    	"exec" => {
    		do_spawn(false /*rust only*/);
    	}
    	_ => {printlninfo!("Unknown command: {}", env::args().nth(1).unwrap());}
    }
}

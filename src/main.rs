#![feature(libc)]
// #![feature(duration_as_u128)]

use std::env;
use std::time::Instant;
use std::process;

const ITERATIONS: usize = 1_000_000;
const TRIES: usize = 10;
const THRESHOLD_ERROR_RATIO: f64 = 0.1;

const T_UNIT: &str = "nano sec";

macro_rules! printlninfo {
    ($fmt:expr) => (println!(concat!("BM-INFO: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-INFO: ", $fmt), $($arg)*));
}

macro_rules! printlnwarn {
    ($fmt:expr) => (println!(concat!("BM-WARN: ", $fmt)));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("BM-WARN: ", $fmt), $($arg)*));
}

// fn gettimeofday(_,_) -> u64 {0}


struct timeval {
	tv_sec: i64,     /* seconds */
	tv_usec: i64    /* microseconds */
}

fn print_tv(tv: &timeval) {
	println!("{} sec {} usec", tv.tv_sec, tv.tv_usec);
}

fn tvsub(tdiff: &mut timeval, t1: &timeval, t0: &timeval) {
	tdiff.tv_sec = t1.tv_sec - t0.tv_sec;
	tdiff.tv_usec = t1.tv_usec - t0.tv_usec;
	if tdiff.tv_usec < 0 && tdiff.tv_sec > 0 {
		tdiff.tv_sec -= 1;
		tdiff.tv_usec += 1_000_000;
		assert!(tdiff.tv_usec >= 0);
	}

	/* time shouldn't go backwards!!! */
	if tdiff.tv_usec < 0 || t1.tv_sec < t0.tv_sec {
		tdiff.tv_sec = 0;
		tdiff.tv_usec = 0;
	}
}

fn tvdelta(start: &timeval, stop: &timeval) -> u64 {
	let mut td: timeval = timeval {tv_sec: 0, tv_usec: 0};
	let mut usecs: u64;

	tvsub(&mut td, stop, start);
	usecs = td.tv_sec as u64;
	usecs *= 1_000_000;
	usecs += td.tv_usec as u64;

	usecs
}


fn print_usage(prog: &String) {
	println!(	"\nUsage: {} cmd\n\
				\t availavle cmds: null
				", prog);
}

// overhead is TIME
fn timing_overhead_inner(th: usize, nr: usize) -> f64 {
	let mut temp;
	let mut start;
	let mut end;

	temp = Instant::now();

	start = Instant::now();
	for _ in 0..ITERATIONS {
		temp = Instant::now();
	}
	end = Instant::now();

	let delta = end - start;
	let delta_time = delta.subsec_nanos() as f64;
	let delta_time_avg = delta_time / ITERATIONS as f64;

	printlninfo!("t_overhead_inner ({}/{}): {} total -> {} avg_ns (ignore: {})", 
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

	let overhead = (tries / TRIES as f64);
	let err = (overhead * THRESHOLD_ERROR_RATIO);
	if 	max - overhead > err || overhead - min > err {
		printlnwarn!("timing_overhead diff is too big: {:.3} ({:.3} - {:.3}) ns", max-min, max, min);
	}

	overhead
}


fn getpid() -> u32 {
	process::id()
}

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
		printlnwarn!("Ignore overhead for null because overhead({:.3}) > diff({:.3})", 
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
		println!("WARNING! benchmark error is too big: (avg {:.3}, max {:.3},  min {:.3})", lat, max, min);
	}

	println!("null test: {:.3} us for {} iterations with {} tries", lat, ITERATIONS, TRIES);
}

fn print_header() {
	printlninfo!("========================================");
	printlninfo!("Time unit : {}", T_UNIT);
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

    print_header();

    match env::args().nth(1).unwrap().as_str() {
    	"null" => {
    		do_null();
    	}
    	_ => {println!("Unknown command: {}", env::args().nth(1).unwrap());}
    }
}

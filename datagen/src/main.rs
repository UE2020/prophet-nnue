use std::time::Instant;
use std::{path::PathBuf, fs::File};
use std::io::{prelude::*, BufWriter, BufReader, self};
use std::process::{Command, Stdio};
use regex::Regex;

use structopt::StructOpt;
use uciengine::{uciengine::*, analysis::*};

/// ProphetNNUE evaluation data generator.
#[derive(StructOpt, Debug)]
#[structopt(name = "prophet-datagen")]
struct Opt {
    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Set depth
    #[structopt(short, long)]
    depth: Option<u8>,

	/// Set nodes
	#[structopt(short, long)]
	nodes: Option<u8>,

    /// Engine path
    #[structopt(short, long, parse(from_os_str))]
    engine_path: PathBuf,

    /// Fens path
    #[structopt(short, long, parse(from_os_str))]
    fens_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    
    let mut child = Command::new(&opt.engine_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to execute child");

	let re = Regex::new(r"[-+]?[0-9]*\.?[0-9]+").unwrap();

	let mut new_data = BufWriter::new(File::create(&opt.output).expect("failed to create output file"));
	new_data.write_all(b"FEN,Evaluation\n").expect("failed to write to output file");

	let file = File::open(&opt.fens_path).expect("file not found");
    let mut rdr = csv::Reader::from_reader(file);

	println!("Using engine: {}", opt.engine_path.display());
	println!("Using FENs: {}", opt.fens_path.display());
	println!("Using depth: {:?}", opt.depth);
	println!("Using nodes: {:?}", opt.nodes);
	println!("Using output file: {}", opt.output.display());
	println!("Beginning data generation!");
	println!();

	let mut counter = 0usize;
	let now = Instant::now();
    'outer: for result in rdr.records() {
		let record = result.expect("failed to parse record");
		if &record[0] == "FEN" {
			continue;
		}

		let child_stdin = child.stdin.as_mut().unwrap();
        child_stdin
            .write_all(format!("position fen {}\neval\n", &record[0]).as_bytes())
            .expect("failed to write");
        drop(child_stdin);
        let child_stdout = child.stdout.as_mut().unwrap();
		let mut reader = BufReader::new(child_stdout);
        let eval = loop {
            let mut bytes = vec![];
            loop {
                // read a char
                let mut output = [0];
                reader
                    .read_exact(&mut output)
                    .expect("Failed to read output");
                if output[0] as char == '\n' {
                    break;
                }
                bytes.push(output[0]);
            }
            let output = String::from_utf8_lossy(&bytes).to_string();
			
			if output.contains("in check") {
				continue 'outer;
			}

			if output.contains("Total evaluation") {
				let matched = re.find(&output);
				if let Some(matched) = matched {
					//child_stdout.read_exact(&mut [0, 0]).expect("Failed to read output");
					break (matched.as_str().parse::<f32>().expect("failed to parse") * 100.0) as i32
				}
			}
        };

		new_data.write_all(format!("{},{}\n", &record[0], eval).as_bytes()).expect("failed to write to output file");

		if counter % 100 == 0 {
			let speed = counter as f32 / now.elapsed().as_secs_f32();
			print!("\x1B[2K\rWrote {} evals ({:.3} eval/s)", counter, speed);
			io::stdout().flush().ok();
		}

		counter += 1;
    }

	if counter > 0 {
		println!("\nDatagen finished, wrote {} evals", counter);
	}

	Ok(())
}

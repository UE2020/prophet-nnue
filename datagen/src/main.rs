use std::time::Instant;
use std::{path::PathBuf, fs::File};
use std::io::{prelude::*, BufWriter};

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
    
	let engine = UciEngine::new(opt.engine_path.display());
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
	println!();

	let mut counter = 0usize;
	let now = Instant::now();
    for result in rdr.records() {
		let record = result.expect("failed to parse record");
		if &record[0] == "FEN" {
			continue;
		}

		let mut job = GoJob::new()
        	.uci_opt("UCI_Variant", "chess")
        	.pos_fen(&record[0]);

		if let Some(depth) = opt.depth {
			job = job.go_opt("depth", depth);
		} else if let Some(nodes) = opt.nodes {
			job = job.go_opt("nodes", nodes);
		} else {
			eprintln!("Error: no depth or nodes specified.");
			break;
		}

		let result = engine.go(job).await.expect("engine failure");
		new_data.write_all(format!("{},{}\n", &record[0], match result.ai.score {
			Score::Cp(cp) => cp,
			Score::Mate(mate) => 1000 * mate.signum(),
		}).as_bytes()).expect("failed to write to output file");

		if counter % 100 == 0 {
			fn up() -> String {
				format!("{}[A", ESC)
			}
			
			fn erase() -> String {
				format!("{}[2K", ESC)
			}

			const ESC: char = 27u8 as char;
			const BACKSPACE: char = 8u8 as char;

			let speed = counter as f32 / now.elapsed().as_secs_f32();
			print!("{}{}", up(), erase());
			println!("Wrote {} evals ({:.3} eval/s)", counter, speed);
		}

		counter += 1;
    }

	if counter > 0 {
		println!("Datagen finished, wrote {} evals", counter);
	}

	Ok(())
}

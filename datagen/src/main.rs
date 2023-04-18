use std::path::PathBuf;
use structopt::StructOpt;

/// ProphetNNUE evaluation data generator.
#[derive(StructOpt, Debug)]
#[structopt(name = "prophet-datagen")]
struct Opt {
	/// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

	/// Set depth
    #[structopt(short, long, default_value = "6")]
    depth: f64,

	/// Engine path
    #[structopt(short, long, parse(from_os_str))]
    engine_path: PathBuf,

	/// Fens path
	#[structopt(short, long, parse(from_os_str))]
    fen_path: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    println!("{:#?}", opt);
}
# ProphetNNUE
Dense 768-bit (no HalfKP) double-layer NNUE implementation, using residual evals (TODO: explain residues).

## Usage

Compile (**REQUIRES NIGHTLY RUSTC**):
```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

For use as a shared library, see `nnue/include/prophet.h` and `target/release/libprophet.so`. Docs are included with headers.

For use as a UCI engine, run `./target/release/prophet-nnue`. The `learn` command can be used to create a new net, as long as chess evals are present at `chessData.csv` in this format:
```csv
FEN,Evaluation
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1,-10
rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2,+56
rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2,-9
rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3,+52
rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3,-26
rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/8/PPPN1PPP/R1BQKBNR w KQkq - 2 4,+50
```

Evaluations must be from the side-to-play PoV.

## Features

- Includes a very basic UCI-compliant α/β chess engine with transposition tables for testing new networks. ([search](/nnue/src/search.rs))
- Includes a C API for basic training & evaluation. ([headers](/nnue/include/prophet.h))
- Includes an evaluation data generator. ([here](/datagen))
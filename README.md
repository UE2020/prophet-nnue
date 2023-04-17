# ProphetNNUE

Dense 768-bit (no HalfKP) double-layer NNUE implementation, using residual evals (TODO: explain residues).

# Usage

Compile:
```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

For use as a shared library, see `include/prophet.h` and `target/release/libprophet.so".

For use as a UCI engine, run `./target/release/prophet_nnue`
# ProphetNNUE-datagen

Generate a compatible CSV using the engine of your choice, and a list of fens.

## Example usage

```
> ./target/release/datagen --depth 6 --engine-path stockfish --fens-path nnue/chessData.csv --output dataset.csv
Using engine: stockfish
Using fens: nnue/chessData.csv
Using depth: 6
Using output file: dataset.csv
Beginning data generation!

Wrote 12958000 evals (522.741 eval/s)
Datagen finished, wrote 12958035 evals.
```
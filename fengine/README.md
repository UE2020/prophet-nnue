# ProphetNNUE-fengine

Generate a list of realistic chess positions in FEN (Forsyth-Edwards Notation) format.

## Example usage

```
> ./fengine/fengine --games 250000 --max-plies 60 --noise-weight 0.95 --output nnue/chessData.csv
Using # games: 250000
Using max plies: 60
Using noise weight: 0.95
Using q-search: false
Using output file: nnue/chessData.csv

Played 249900 games, produced 14205785 positions (1177.799 games/s, 66953.008 positions/s)
Fengine finished, wrote 14211383 chess positions.
```
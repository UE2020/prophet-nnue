# ProphetNNUE-fengine

Generate a list of realistic chess positions in FEN (Forsyth-Edwards Notation) format.

## Example usage

```
> ./fengine/fengine --games 250000 --max-plies 60 --noise-weight 0.9 --output nnue/chessData.csv
Using # games: 250000
Using max plies: 60
Using noise weight: 0.9
Using output file: nnue/chessData.csv

Played 250000 games (1167.185 games/s)
Fengine finished, wrote 9635875 chess positions.
```
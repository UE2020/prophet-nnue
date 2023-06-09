#![allow(non_snake_case)]

use chess::*;
use vampirc_uci::UciPiece;

use std::str::FromStr;

use std::io::{self, BufRead};
use vampirc_uci::{parse_one, UciMessage};

//mod belief;
mod nn;
mod search;

use dfdx::prelude::*;

type Device = dfdx::tensor::Cpu;

fn main() {
    let dev = Device::default();

    let mut model = dev.build_module::<nn::Model<256>, f32>();
    model.load("./nnue/nnue.npz").unwrap();

    let mut nnue = nn::QuantizedNNUE::from_built_model(&model);

    //let mut inference = nn::QuantizedNNUE::from_built_model(&model);
    let mut board = Board::default();

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "learn" {
            nn::train(
                "./nnue/nnue.npz",
                "./nnue/dataset.csv",
                "./nnue/testset.csv",
                false,
                1e-3,
                0.0,
                50,
            );
        }
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::Uci => {
                println!("id name ProphetNNUE");
                println!("uciok")
            }
            UciMessage::Position {
                startpos,
                moves,
                fen,
            } => {
                if startpos {
                    board = Board::default();
                } else if let Some(fen) = fen {
                    board = Board::from_str(&fen.0).unwrap();
                }

                for mov in moves {
                    let from = mov.from;
                    let to = mov.to;

                    let from = chess::Square::make_square(
                        match from.rank {
                            1 => Rank::First,
                            2 => Rank::Second,
                            3 => Rank::Third,
                            4 => Rank::Fourth,
                            5 => Rank::Fifth,
                            6 => Rank::Sixth,
                            7 => Rank::Seventh,
                            8 => Rank::Eighth,
                            _ => unreachable!(),
                        },
                        match from.file {
                            'a' => File::A,
                            'b' => File::B,
                            'c' => File::C,
                            'd' => File::D,
                            'e' => File::E,
                            'f' => File::F,
                            'g' => File::G,
                            'h' => File::H,
                            _ => unreachable!(),
                        },
                    );

                    let to = chess::Square::make_square(
                        match to.rank {
                            1 => Rank::First,
                            2 => Rank::Second,
                            3 => Rank::Third,
                            4 => Rank::Fourth,
                            5 => Rank::Fifth,
                            6 => Rank::Sixth,
                            7 => Rank::Seventh,
                            8 => Rank::Eighth,
                            _ => unreachable!(),
                        },
                        match to.file {
                            'a' => File::A,
                            'b' => File::B,
                            'c' => File::C,
                            'd' => File::D,
                            'e' => File::E,
                            'f' => File::F,
                            'g' => File::G,
                            'h' => File::H,
                            _ => unreachable!(),
                        },
                    );
                    let mov = ChessMove::new(
                        from,
                        to,
                        mov.promotion.map(|piece| match piece {
                            UciPiece::Pawn => Piece::Pawn,
                            UciPiece::Knight => Piece::Knight,
                            UciPiece::Bishop => Piece::Bishop,
                            UciPiece::Rook => Piece::Rook,
                            UciPiece::Queen => Piece::Queen,
                            UciPiece::King => Piece::King,
                        }),
                    );
                    board = board.make_move_new(mov);
                }
            }
            UciMessage::Go { .. } => {
                let mut board_tensor = vec![0f32; 768];
                crate::nn::encode(&board, &mut board_tensor);
                let test_tensor = dev.tensor_from_vec(board_tensor, (Const::<768>,));
                let logits = model.forward(test_tensor);
                let eval = logits.array()[0];
                eprintln!("NN eval: {}", eval);

                nnue.reset();
                nnue.activate_all(&board);
                eprintln!("NNUE eval: {}", nnue.eval(board.side_to_move()));

                let result = search::iterative_deepening_search(board, &dev, &mut nnue);
                println!("bestmove {}", result.0);
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            _ => {}
        }
    }
}

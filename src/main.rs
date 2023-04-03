#![allow(non_snake_case)]
/*#![feature(portable_simd)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]*/
#![feature(generic_const_exprs)]

use chess::*;
use vampirc_uci::UciPiece;

use std::str::FromStr;

use std::io::{self, BufRead};
use vampirc_uci::{parse_one, UciMessage};

//mod belief;
mod nn;
mod search;

use dfdx::{
    data::*,
    losses::mse_loss,
    nn::SaveToNpz,
    optim::{Adam, AdamConfig, Momentum, Optimizer, Sgd, SgdConfig, WeightDecay},
    prelude::*,
    tensor::TensorFrom,
    tensor::TensorFromVec,
    tensor::{AsArray, DeviceStorage, Trace},
    tensor_ops::Backward,
};

type Device = dfdx::tensor::Cpu;
use rand::prelude::{SeedableRng, StdRng};

fn main() {
    nn::main();
    return;
    // let dev = Device::default();
    // let mut rng = StdRng::seed_from_u64(0);

    // let mut model = dev.build_module::<nn::Model, f32>();
    // model.load("dense_mlp.npz").unwrap();
    // let mut board_tensor = vec![0.0; 896];
    // let board = Board::from_str("4q3/1K1k2P1/3Q3b/3R4/8/8/8/5r1r b - - 0 1").expect("bad fen");
    // nn::encode(board, &mut board_tensor, false);
    // let logits = model.forward(dev.tensor_from_vec(board_tensor, (Const::<896>,)));
    // let logits_array = logits.array();

    // println!("Side to move: {:?}", board.side_to_move());
    // println!("Centipawn eval: {:.2}", logits_array[0] * 2000.0);
    // println!("Pawn eval: {:.2}", logits_array[0] * 20.0);
    //nn::main();
}

fn unused() {
    let dev = Device::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<nn::Model, f32>();
    model.load("dense_mlp.npz").unwrap();
    //nn::main();
    let mut board = Board::default();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap());
        match msg {
            UciMessage::Uci => {
                println!("id name DeepOrca");
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
                /*let mut best_move = None;
                let mut best_score = -1000.0;
                let movegen = MoveGen::new_legal(&board);
                for mov in movegen {
                    let mut board_tensor = vec![0f32; 896];
                    let board = board.make_move_new(mov);
                    nn::encode(board, &mut board_tensor, false);
                    let test_tensor = dev.tensor_from_vec(board_tensor, (Const::<896>,));
                    let logits = model.forward(test_tensor);
                    if (-logits.array()[0] * 20.0) > best_score {
                        best_score = -logits.array()[0] * 20.0;
                        best_move = Some(mov);
                    }
                }

                let mut board_tensor = vec![0f32; 896];
                nn::encode(board, &mut board_tensor, false);
                let test_tensor = dev.tensor_from_vec(board_tensor, (Const::<896>,));
                let logits = model.forward(test_tensor);
                let score = logits.array()[0] * 20.0;

                println!(
                    "info currmove {}  depth 1 score cp {} pv {}",
                    best_move.unwrap(),
                    (score * 100.0) as i32,
                    best_move.unwrap()
                );
                println!("bestmove {}", best_move.unwrap());*/

                let result = search::iterative_deepening_search(board, &dev, &model);
				println!(
                    "info currmove {} depth 1 score cp {} pv {}",
                    result.0,
                	result.1,
                    result.0
                );                println!("bestmove {}", result.0);
                eprintln!("Eval: {}", result.1);
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            _ => {}
        }
    }
}

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
mod search;
mod nn;

fn main() {
    nn::main();
    /*let mut board = Board::default();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap());
        match msg {
            UciMessage::Uci => {
                println!("id name DeepAkil");
                println!("uciok")
            }
            UciMessage::Position {
                startpos, moves, fen
            } => {
                if startpos {
                    board = Board::default();
                } else if let Some(fen) = fen {
                    board = Board::from_str(&fen.0).unwrap();
                }

                for mov in moves {
                    let from = mov.from;
                    let to = mov.to;

                    let from = Square::make_square(
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

                    let to = Square::make_square(
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
                //let result = search::search(board);
                //println!("bestmove {}", result.0);
                //eprintln!("Eval: {}", result.1);
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            _ => {}
        }
    }*/
}

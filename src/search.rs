use std::cmp::Ordering;
use std::hash::Hash;
use std::time::Duration;
use std::time::Instant;

mod eval;
pub use eval::*;

mod table;
pub use table::*;

use chess::*;

use crate::nn;

use dfdx::{
    data::*, losses::mse_loss, nn::modules, optim::*, prelude::*, tensor::*, tensor_ops::Backward,
};

const MAX_SEARCH_DEPTH: usize = 7;
const INF: i32 = 20000;

pub type HashTable = CacheTable<TranspositionTableEntry>;

pub fn iterative_deepening_search(board: Board, dev: &nn::Device, net: &nn::BuiltModel) -> (ChessMove, i32) {
    // initialize hash table, size: 256
    let mut table = CacheTable::new(256, TranspositionTableEntry::default());

    let mut curr_mov = None;
    let mut curr_value = 0;
    for depth in 1..MAX_SEARCH_DEPTH as u8 {
        let alpha = -2000;
        let beta = 2000;
        let (mov, value) = root_search(&mut table, board, alpha, beta, depth, dev, net);
        curr_mov = Some(mov);
        curr_value = value;
		println!(
			"info currmove {} depth {depth} score cp {} pv {}",
			curr_mov.expect("no move found"), curr_value, curr_mov.expect("no move found")
		);
        //first_guess = mtdf(&mut table, board, first_guess, depth as u8);
    }

    (curr_mov.expect("no move found"), curr_value)
}

pub fn root_search(
    table: &mut HashTable,
    board: Board,
    mut alpha: i32,
    mut beta: i32,
    depth: u8,
    dev: &nn::Device,
    net: &nn::BuiltModel,
) -> (ChessMove, i32) {
    let alpha_orig = alpha;
    if let Some(entry) = table.get(board.get_hash()) {
        if entry.depth >= depth {
            match entry.flag {
                TranspositionFlag::Exact => return (entry.mov, entry.value),
                TranspositionFlag::Lowerbound => alpha = alpha.max(entry.value),
                TranspositionFlag::Upperbound => beta = beta.min(entry.value),
            }

            if alpha >= beta {
                return (entry.mov, entry.value);
            }
        }
    }

    let movegen = MoveGen::new_legal(&board);
    let mut best_mov = None;
    for mov in movegen {
        let score = -negamax(
            table,
            board.make_move_new(mov),
            -beta,
            -alpha,
            depth - 1,
            dev,
            net,
        );

        if score >= beta {
            best_mov = Some(mov);
            alpha = beta;
            break;
        }

        if score > alpha {
            alpha = score;
            best_mov = Some(mov);
        }
    }

    if let Some(mov) = best_mov {
        let entry = TranspositionTableEntry {
            flag: if alpha <= alpha_orig {
                TranspositionFlag::Upperbound
            } else if alpha >= beta {
                TranspositionFlag::Lowerbound
            } else {
                TranspositionFlag::Exact
            },
            depth,
            value: alpha,
            mov,
        };

        table.add(board.get_hash(), entry);
    }

    (best_mov.unwrap(), alpha)
}

pub fn negamax(
    table: &mut HashTable,
    board: Board,
    mut alpha: i32,
    mut beta: i32,
    depth: u8,
    dev: &nn::Device,
    net: &nn::BuiltModel,
) -> i32 {
    let alpha_orig = alpha;
    if let Some(entry) = table.get(board.get_hash()) {
        if entry.depth >= depth {
            match entry.flag {
                TranspositionFlag::Exact => return entry.value,
                TranspositionFlag::Lowerbound => alpha = alpha.max(entry.value),
                TranspositionFlag::Upperbound => beta = beta.min(entry.value),
            }

            if alpha >= beta {
                return entry.value;
            }
        }
    }

    if depth == 0 {
        let mut board_tensor = vec![0f32; 768];
        crate::nn::encode(&board, &mut board_tensor);
        let test_tensor = dev.tensor_from_vec(board_tensor, (Const::<768>,));
        let logits = net.forward(test_tensor);
        let eval = (logits.array()[0] * 100.0) as i32 + (eval(board) * 100);
        return eval;
    }

    match board.status() {
        BoardStatus::Checkmate => return -20000,
        BoardStatus::Stalemate => return 0,
        _ => {}
    }

    let movegen = MoveGen::new_legal(&board);
    let mut best_mov = None;
    for mov in movegen {
        let score = -negamax(
            table,
            board.make_move_new(mov),
            -beta,
            -alpha,
            depth - 1,
            dev,
            net,
        );

        if score >= beta {
            best_mov = Some(mov);
            alpha = beta;
            break;
        }

        if score > alpha {
            alpha = score;
            best_mov = Some(mov);
        }
    }

    if let Some(mov) = best_mov {
        let entry = TranspositionTableEntry {
            flag: if alpha <= alpha_orig {
                TranspositionFlag::Upperbound
            } else if alpha >= beta {
                TranspositionFlag::Lowerbound
            } else {
                TranspositionFlag::Exact
            },
            depth,
            value: alpha,
            mov,
        };

        table.add(board.get_hash(), entry);
    }

    alpha
}

/*pub fn mtdf(table: &mut HashTable, board: Board, mut gamma: i32, depth: u8) -> (i32, ChessMove) {
    let mut upperbound = INF;
    let mut lowerbound = -INF;

    while upperbound > lowerbound {
        let beta = gamma.max(lowerbound + 1);
        gamma = alpha_beta_mem(table, board, beta - 1, beta, depth, true, board.side_to_move());
        if gamma < beta {
            upperbound = g;
        } else {
            lowerbound = g;
        }
    }

    todo!()
}

pub fn alpha_beta_mem(table: &mut HashTable, board: Board, mut alpha: i32, mut beta: i32, depth: u8, maximizing: bool, color: Color) -> (i32, ChessMove) {
    if let Some(entry) = table.get(board.get_hash()) {
        if entry.lowerbound >= beta {
            return (entry.lowerbound, entry.mov);
        }

        if entry.upperbound <= alpha {
            return (entry.upperbound, entry.mov);
        }

        alpha = alpha.max(entry.lowerbound);
        beta = beta.min(entry.upperbound);
    }

    let mut gamma;

    if depth == 0 {
        gamma = eval(board, color);
    } else if maximizing {
        gamma = -INF;
        let mut a = alpha;
        let mut movegen = MoveGen::new_legal(&board);
        for mov in movegen {
            if gamma >= beta {
                break;
            }
            let mut new_board = board.clone();
            board.make_move(mov, &mut new_board);
            gamma = gamma.max(alpha_beta_mem(table, new_board, a, beta, depth - 1, false, color).0);
            a = a.max(gamma);
        }
    } else {
        gamma = INF;
        let mut b = beta;
        let mut movegen = MoveGen::new_legal(&board);
        for mov in movegen {
            if gamma <= alpha {
                break;
            }
            let mut new_board = board.clone();
            board.make_move(mov, &mut new_board);
            gamma = gamma.min(alpha_beta_mem(table, new_board, alpha, b, depth - 1, true, color).0);
            b = b.min(gamma);
        }
    }

    let mut entry = TranspositionTableEntry::default();

    if gamma <= alpha {
        entry.upperbound = gamma;
        table.add(board.get_hash(), entry);
    }


}
*/

mod table;
pub use table::*;

use chess::*;

use crate::nn;

use dfdx::prelude::*;

const MAX_SEARCH_DEPTH: usize = 5;

pub type HashTable = CacheTable<TranspositionTableEntry>;

pub fn iterative_deepening_search(
    board: Board,
    dev: &nn::Device,
    net: &mut nn::QuantizedNNUE,
) -> (ChessMove, i32) {
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
            curr_mov.expect("no move found"),
            curr_value,
            curr_mov.expect("no move found")
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
    net: &mut nn::QuantizedNNUE,
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
    net: &mut nn::QuantizedNNUE,
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
        net.reset();
        net.activate_all(&board);
        return net.eval(board.side_to_move());
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

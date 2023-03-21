use std::cmp::Ordering;
use std::hash::Hash;
use std::time::Duration;
use std::time::Instant;

mod eval;
pub use eval::*;

mod table;
pub use table::*;

use chess::*;

const MAX_SEARCH_DEPTH: usize = 25;
const INF: i32 = 20000;

pub type HashTable = CacheTable<TranspositionTableEntry>;

pub fn iterative_deepening_search(board: Board) -> (ChessMove, i32) {
    // initialize hash table, size: 256
    //let mut table = CacheTable::new(256, TranspositionTableEntry::default());

    let mut first_guess = 0;

    for depth in 1..MAX_SEARCH_DEPTH {
        //first_guess = mtdf(&mut table, board, first_guess, depth as u8);
    }

    todo!()
}

pub fn negamax(
    table: &mut HashTable,
    board: Board,
    mut alpha: i32,
    mut beta: i32,
    depth: u8,
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
        return eval(board);
    }

    todo!()
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

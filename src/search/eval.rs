use chess::*;

pub fn eval(board: Board) -> i32 {
    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);

    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    let eval = ((pawns&white).popcnt() as i32 - (pawns&black).popcnt() as i32) + (((knights&white).popcnt() as i32*3) - ((knights&black).popcnt() as i32*3)) + (((bishops&white).popcnt() as i32*3) - ((bishops&black).popcnt() as i32*3)) + (((rooks&white).popcnt() as i32*5) - ((rooks&black).popcnt() as i32*5)) + (((queens&white).popcnt() as i32*9) - ((queens&black).popcnt() as i32*9));
    if board.side_to_move() == Color::White {
        eval 
    } else {
        -eval
    }
}

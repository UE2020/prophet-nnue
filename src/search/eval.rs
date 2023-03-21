use chess::*;

pub fn eval(board: Board, orig_color: Color) -> i32 {
    let mut eval = 0i32;

    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);

    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    #[rustfmt::skip]
    let pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ];

    #[rustfmt::skip]
    let knight_table = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ];

    #[rustfmt::skip]
    let bishop_table = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ];

    #[rustfmt::skip]
    let rook_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
       -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ];

    #[rustfmt::skip]
    let queen_table = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ];

    #[rustfmt::skip]
    let king_table = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ];

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval += 100 + pawn_table[sq.to_index()];

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval -= 100 + pawn_table[sq.to_index() ^ 0x38];

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval += 320 + knight_table[sq.to_index()];

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval -= 320 + knight_table[sq.to_index() ^ 0x38];

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval += 330 + bishop_table[sq.to_index()];

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval -= 330 + bishop_table[sq.to_index() ^ 0x38];

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval += 500 + rook_table[sq.to_index()];

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval -= 500 + rook_table[sq.to_index() ^ 0x38];

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval += 900 + queen_table[sq.to_index()];

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        eval -= 900 + queen_table[sq.to_index() ^ 0x38];

        remaining ^= BitBoard::from_square(sq);
    }

    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);

    eval += king_table[white_king.to_index()];
    eval -= king_table[black_king.to_index() ^ 0x38];

    if orig_color == Color::White {
        eval
    } else {
        -eval
    }
}

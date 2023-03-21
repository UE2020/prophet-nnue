use dfdx::{
    prelude::*,

    losses::mse_loss,
    nn::builders::*,
    nn::SaveToNpz,
    optim::{Momentum, Optimizer, Sgd, SgdConfig},
    shapes::Rank2,
    tensor::{AsArray, SampleTensor, Tensor, Trace, DeviceStorage},
    tensor_ops::Backward,
    nn::TensorCollection,
    tensor::TensorFrom,
    tensor::TensorFromVec
};

use chess::*;

use crate::search::eval;

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<768, 512>, ReLU),
    (Linear<512, 256>, ReLU),
    (Linear<256, 1>, Tanh),
);

pub fn main() {
    let dev = Device::default();

    let mut mlp = dev.build_module::<Mlp, f32>();
    let mut grads = mlp.alloc_grads();

    let mut sgd = Sgd::new(
        &mlp,
        SgdConfig {
            lr: 0.1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );

    let mut x_data = vec![vec![0f32; 768]; 40000];
    let mut y_data = vec![vec![0f32; 1]; 40000];
    for i in 0..10000 {
        let mut board = Board::default();
        for i in 0..40 {
            let mut moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
            if moves.len() == 0 {
                board = Board::default();
                moves = MoveGen::new_legal(&board).collect();
            }
            let rand = (dev.random_u64() as f64 / u64::MAX as f64) * moves.len() as f64;
            let mov = moves[rand as usize];
            board = board.make_move_new(mov);
            let eval = eval(board, board.side_to_move());
            y_data[i][0] = eval as f32 / 20.0;
            // encode
            encode(board, &mut x_data[i]);
        }
    }

    let x: Tensor<Rank2<40000, 768>, f32, _> = dev.tensor_from_vec(flatten(x_data), (Const::<40000>,Const::<768>));
    let y: Tensor<Rank2<40000, 1>, f32, _> = dev.tensor_from_vec(flatten(y_data), (Const::<40000>,Const::<1>));

    for i in 0..1000 {
        let prediction = mlp.forward_mut(x.trace(grads));
        let loss = mse_loss(prediction, y.clone());
        println!("{}\t{:?}", i, loss.array());
        grads = loss.backward();
        sgd.update(&mut mlp, &grads)
            .expect("Oops, there were some unused params");
        mlp.zero_grads(&mut grads);
    }
    
    mlp.save("model.npz").unwrap();
}

fn flatten<T>(nested: Vec<Vec<T>>) -> Vec<T> {
    nested.into_iter().flatten().collect()
}

pub fn encode(board: Board, out: &mut [f32]) {
    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);
    let kings = board.pieces(Piece::King);
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    let multiplier = if board.side_to_move() == Color::White {1.0} else {-1.0};

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index()] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index()] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*2] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*2] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*3] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*3] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*4] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*4] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*5] = multiplier * 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[sq.to_index() + 64*5] = multiplier * -1.0;

        remaining ^= BitBoard::from_square(sq);
    }
}
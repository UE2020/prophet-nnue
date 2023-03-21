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
    tensor::TensorFromVec,
    data::*,
};

use chess::*;

use std::{time::Instant, str::FromStr};
use rand::prelude::{SeedableRng, StdRng};
use crate::search::eval;
use indicatif::ProgressIterator;

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<384, 256>, ReLU),
    (Linear<256, 128>, ReLU),
    (Linear<128, 1>, Tanh),
);

pub struct Positions {
    input: Vec<Vec<f32>>,
    labels: Vec<f32>,
}


impl ExactSizeDataset for Positions {
    type Item<'a> = (Vec<f32>, f32) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        (self.input[index].clone(), self.labels[index])
    }
    fn len(&self) -> usize {
        self.input.len()
    }
}

pub fn main() {
    let dev = Device::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut mlp = dev.build_module::<Mlp, f32>();
    mlp.load("model.npz").unwrap();
    let mut test_tensor = [0f32; 384];
    let board = Board::from_str("4k3/8/8/8/8/8/RNBQKBNR/QQQQQQQQ w - - 0 1").unwrap();
    encode(board, &mut test_tensor);
    let test_tensor = dev.tensor(test_tensor);
    let logits = mlp.forward(test_tensor);
    dbg!(logits.array()[0] * 103.0);
    dbg!(eval(board));
    return;
    let mut grads = mlp.alloc_grads();

    let mut sgd = Sgd::new(
        &mlp,
        SgdConfig {
            lr: 0.1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );


    let mut x_data = vec![vec![0f32; 384]; 1000000];
    let mut y_data = vec![0f32; 1000000];
    let mut training_point = 0usize;
    for i in 0..25000 {
        let mut board = Board::default();
        for _ in 0..40 {
            let mut moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
            if moves.len() == 0 {
                board = Board::default();
                moves = MoveGen::new_legal(&board).collect();
            }
            let rand = (dev.random_u64() as f64 / u64::MAX as f64) * moves.len() as f64;
            let mov = moves[rand as usize];
            board = board.make_move_new(mov);
            let eval = eval(board);
            y_data[training_point] = eval as f32 / 103.0;
            // encode
            encode(board, &mut x_data[training_point]);
            training_point += 1;
        }
    }

    let positions = Positions {
        input: x_data,
        labels: y_data,
    };

    const BATCH_SIZE: usize = 32;

    let preprocess = |(input, lbl): <Positions as ExactSizeDataset>::Item<'_>| {
        (
            dev.tensor_from_vec(input, (Const::<384>,)),
            dev.tensor([lbl]),
        )
    };

    for i_epoch in 0..3 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (input, label) in positions
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = mlp.forward_mut(input.traced(grads));
            let loss = mse_loss(logits, label);

            total_epoch_loss += loss.array();
            //dbg!(loss.array());
            num_batches += 1;

            grads = loss.backward();
            sgd.update(&mut mlp, &grads).unwrap();
            mlp.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
        );
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
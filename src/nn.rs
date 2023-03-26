use dfdx::{
    data::*,
    losses::mse_loss,
    nn::SaveToNpz,
    optim::*,
    prelude::*,
    tensor::*,
    tensor_ops::Backward,
};

use chess::*;
use crate::search::eval;
use indicatif::ProgressIterator;
use rand::prelude::{SeedableRng, StdRng};
use std::{str::FromStr, time::Instant, vec};

type Device = dfdx::tensor::Cuda;

pub type BasicBlock<const C: usize> = Residual<(
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
    ReLU,
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
)>;

pub type Model = (
    ((Conv2D<12, 32, 3, 1, 1>, BatchNorm2D<32>), ReLU),
	(BasicBlock<32>, ReLU, BasicBlock<32>, ReLU),
	(BasicBlock<32>, ReLU, BasicBlock<32>, ReLU),
	(BasicBlock<32>, ReLU, BasicBlock<32>, ReLU),
    ((Conv2D<32, 1, 1>, BatchNorm2D<1>), ReLU),
    (Flatten2D, Linear<64, 256>, ReLU, Linear<256, 1>, Tanh),
    //(Linear<128, 1>, Tanh)
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

    let mut model = dev.build_module::<Model, f32>();
    let mut grads = model.alloc_grads();

    let mut opt = Adam::new(
        &model,
        AdamConfig {
			weight_decay: Some(WeightDecay::L2(0.00001)),
            ..Default::default()
        },
    );

    // read csv
    println!("Gathering data...");
    let file = std::fs::File::open("chessData.csv").expect("file not found");
    let mut rdr = csv::Reader::from_reader(file);
    let mut game = 0;
    let mut train_positions = Positions {
        input: vec![],
        labels: vec![],
    };

    let mut test_positions = Positions {
        input: vec![],
        labels: vec![],
    };

    for result in rdr.records() {
        if game > 501000 {
            break;
        }
        let record = result.unwrap();
        let board = Board::from_str(&record[0]).expect("bad fen");
        let eval = record[1].parse::<i32>();
        let eval = match eval {
            Ok(eval) => eval.clamp(-2000, 2000),
            Err(_) => continue,
        };

        if eval.abs() < 100 {
            continue;
        }

        let mut input = vec![0f32; 768];
        encode(board, &mut input);

		let eval = if board.side_to_move() == Color::White {
			eval
		} else {
			-eval
		};

        if game > 500000 {
            test_positions.input.push(input);
            test_positions.labels.push(eval as f32 / 2000.0);
        } else {
			train_positions.input.push(input);
            train_positions.labels.push(eval as f32 / 2000.0);
		}

        game += 1;
    }

    println!("Done!");

    const BATCH_SIZE: usize = 64;

    let preprocess = |(input, lbl): <Positions as ExactSizeDataset>::Item<'_>| {
        (
            dev.tensor_from_vec(input, (Const::<12>, Const::<8>, Const::<8>)),
            dev.tensor([lbl]),
        )
    };

    for i_epoch in 0..1000 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (input, label) in train_positions
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(input.traced(grads));
            let loss = mse_loss(logits, label);

            total_epoch_loss += loss.array();
            //dbg!(loss.array());
            num_batches += 1;

            grads = loss.backward();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

        model.save("conv_model.npz").unwrap();

        // test model
        let mut test_total_epoch_loss = 0.0;
        let mut test_num_batches = 0;
        for (input, label) in test_positions
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(Const::<BATCH_SIZE>)
            .collate()
            .stack()
        {
            let logits = model.forward(input);
            let loss = mse_loss(logits, label);

            test_total_epoch_loss += loss.array();
            test_num_batches += 1;
        }

		println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}; test loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
			BATCH_SIZE as f32 * test_total_epoch_loss / test_num_batches as f32,
        );

        if (BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32) <= 0.01 {
            break;
        }
    }

    //mlp.save("conv_model.npz").unwrap();
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
    let mut white = board.color_combined(Color::White);
    let mut black = board.color_combined(Color::Black);
	let flip = board.side_to_move() == Color::Black;

	if board.side_to_move() == Color::Black {
		std::mem::swap(&mut white, &mut black);
	}

	fn to_index(sq: chess::Square, flip: bool) -> usize {
		if flip { 63 - sq.to_index() } else { sq.to_index() }
	}

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 2] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 3] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 4] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 5] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 6] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 7] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 8] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 9] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 10] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, flip) + 64 * 11] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }
}

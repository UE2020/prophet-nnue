use dfdx::{
    data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor_ops::Backward,
};

use chess::*;
use indicatif::ProgressIterator;

use rand::prelude::{SeedableRng, StdRng};
use std::str::FromStr;

mod clipped_relu;
pub use clipped_relu::*;

pub type Device = dfdx::tensor::Cpu;

pub type FeatureTransformer<const TRANSFORMED_SIZE: usize> = (
    Linear<768, TRANSFORMED_SIZE>, ClippedReLU
);

pub type Model<const L1: usize = 32, const L2: usize = 32> = (
    // feature transformer
    FeatureTransformer<256>,
    (Linear<256, L1>, ClippedReLU),
    (Linear<L1, L2>, ClippedReLU),
    (Linear<L2, 1>, Tanh),
);

pub type BuiltModel = ((modules::Linear<768, 256, f32, Cpu>, ClippedReLU), (modules::Linear<256, 32, f32, Cpu>, ClippedReLU), (modules::Linear<32, 32, f32, Cpu>, ClippedReLU), (modules::Linear<32, 1, f32, Cpu>, Tanh));

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

/// Calculate material imbalance
pub fn eval<E: Encodable>(board: &E) -> i32 {
    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);

    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    let eval = ((pawns & white).popcnt() as i32 - (pawns & black).popcnt() as i32)
        + (((knights & white).popcnt() as i32 * 3) - ((knights & black).popcnt() as i32 * 3))
        + (((bishops & white).popcnt() as i32 * 3) - ((bishops & black).popcnt() as i32 * 3))
        + (((rooks & white).popcnt() as i32 * 5) - ((rooks & black).popcnt() as i32 * 5))
        + (((queens & white).popcnt() as i32 * 9) - ((queens & black).popcnt() as i32 * 9));
    if board.side_to_move() == Color::White {
        eval
    } else {
        -eval
    }
}

pub fn train() {
    let dev = Device::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<Model<32, 32>, f32>();

    println!(
        "Number of trainable parameters: {:.2}k",
        model.num_trainable_params() as f32 / 1000 as f32
    );
    let mut grads = model.alloc_grads();

    let mut opt = Adam::new(&model, AdamConfig {
        weight_decay: Some(WeightDecay::L2(0.0001)),
        ..Default::default()
    });

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
        if game > 2010000 {
            break;
        }
        let record = result.unwrap();
        let board = Board::from_str(&record[0]).expect("bad fen");

        let static_eval = eval(&board) * 100;

		if static_eval != 0 {
			continue;
		}

        let eval = if let Ok(eval) = record[1].parse::<i32>() {
            eval.clamp(-100, 100)
        } else {
            continue;
        };

        let eval = if board.side_to_move() == Color::Black {
            -eval
        } else {
            eval
        };

		if eval.abs() >= 100 {
			continue;
		}

        if game > 2000000 {
            let mut input = vec![0f32; 768];
            encode(&board, &mut input);
            test_positions.input.push(input);

            test_positions.labels.push(eval as f32 / 100.0);
        } else {
            let mut input = vec![0f32; 768];
            encode(&board, &mut input);
            train_positions.input.push(input);

            train_positions.labels.push(eval as f32 / 100.0);
        }

        game += 1;
    }

    println!("Done!");

    const BATCH_SIZE: usize = 64;

    let preprocess = |(input, lbl): <Positions as ExactSizeDataset>::Item<'_>| {
        (
            dev.tensor_from_vec(input, (Const::<768>,)),
            dev.tensor([lbl]),
        )
    };

    println!("Epoch\tTrain Loss\tTest Loss");
    for i_epoch in 0..1000 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
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
            num_batches += 1;

            grads = loss.backward();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }

        model.save("dense_mlp.npz").unwrap();

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
            "{i_epoch}\t{:.5}\t{:.5}",
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            BATCH_SIZE as f32 * test_total_epoch_loss / test_num_batches as f32,
        );

        if (BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32) <= 0.01 {
            break;
        }
    }
}

pub trait Encodable {
	fn pieces(&self, piece: Piece) -> BitBoard;
	fn color_combined(&self, color: Color) -> BitBoard;
	fn side_to_move(&self) -> Color;
}

impl Encodable for Board {
	fn pieces(&self, piece: Piece) -> BitBoard {
		*self.pieces(piece)
	}

	fn color_combined(&self, color: Color) -> BitBoard {
		*self.color_combined(color)
	}

	fn side_to_move(&self) -> Color {
		self.side_to_move()
	}
}

pub fn encode<E: Encodable>(board: &E, out: &mut [f32]) {
    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);
    let kings = board.pieces(Piece::King);
    let mut white = board.color_combined(Color::White);
    let mut black = board.color_combined(Color::Black);
    let is_black = board.side_to_move() == Color::Black;

    if is_black {
        std::mem::swap(&mut white, &mut black);
    }

    fn to_index(sq: chess::Square, is_black: bool) -> usize {
        let idx = sq.to_index();

        if is_black {
            idx ^ 56
        } else {
            idx
        }
    }

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 2] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 3] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 4] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 5] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 6] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 7] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 8] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 9] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 10] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[to_index(sq, is_black) + 64 * 11] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    // let mut movegen = MoveGen::new_legal(&board);
    // movegen.set_iterator_mask(*black); // calculate attacks
    // for mov in movegen {
    //     out[to_index(mov.get_dest(), flip) + 64 * 12] = 1.0;
    // }

    // let board = board.null_move();
    // if let Some(board) = board {
    //     let mut movegen = MoveGen::new_legal(&board);
    //     movegen.set_iterator_mask(*white); // calculate attacks
    //     for mov in movegen {
    //         out[to_index(mov.get_dest(), flip) + 64 * 13] = 1.0;
    //     }
    // }
}

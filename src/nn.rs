use dfdx::{
    data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor::*, tensor_ops::Backward,
};

use crate::search::eval;
use chess::*;
use indicatif::ProgressIterator;
use rand::prelude::{SeedableRng, StdRng};
use std::{str::FromStr, time::Instant, vec, default};

type Device = dfdx::tensor::Cpu;

pub type Model = (
    (Linear<896, 256>, ReLU, DropoutOneIn<4>),
    (Linear<256, 128>, ReLU, DropoutOneIn<4>),
    Linear<128, 3>
);

pub struct Positions {
    input: Vec<Vec<f32>>,
    labels: Vec<[f32; 3]>,
}

impl ExactSizeDataset for Positions {
    type Item<'a> = (Vec<f32>, [f32; 3]) where Self: 'a;
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
    //model.load("conv_model.npz");
    println!(
        "Number of trainable parameters: {:.2}k",
        model.num_trainable_params() as f32 / 1000 as f32
    );
    let mut grads = model.alloc_grads();

    let mut opt = Adam::new(
        &model,
        AdamConfig::default(),
    );

    // read csv
    println!("Gathering data...");
    let file = std::fs::File::open("tactic_evals.csv").expect("file not found");
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
        if game > 1010000 {
            break;
        }
        let record = result.unwrap();
        let board = Board::from_str(&record[0]).expect("bad fen");
        let eval = record[1].parse::<i32>();
        let eval = match eval {
            Ok(eval) => eval.clamp(-2000, 2000),
            Err(_) => continue,
        };

        let eval = if board.side_to_move() == Color::Black {
            -eval
        } else {
            eval
        };

        //let eval = (eval(board) * 100).clamp(-2000, 2000);

		let eval = if eval.abs() <= 100 {
			[0.0, 1.0, 0.0]
		} else {
			if eval.signum() == 1 {
				[0.0, 0.0, 1.0]
			} else {
				[1.0, 0.0, 0.0]
			}
		};

        if game > 1000000 {
			let mut input = vec![0f32; 896];
			encode(board, &mut input, false);
            test_positions.input.push(input);

			let mut input = vec![0f32; 896];
			encode(board, &mut input, true);
            test_positions.input.push(input);

            test_positions.labels.push(eval);
			test_positions.labels.push(eval);
        } else {
			let mut input = vec![0f32; 896];
			encode(board, &mut input, false);
            train_positions.input.push(input);

			let mut input = vec![0f32; 896];
			encode(board, &mut input, true);
            train_positions.input.push(input);

            train_positions.labels.push(eval);
			train_positions.labels.push(eval);
        }

        game += 1;
    }

    println!("Done!");

    const BATCH_SIZE: usize = 128;

    let preprocess = |(input, lbl): <Positions as ExactSizeDataset>::Item<'_>| {
        (
            dev.tensor_from_vec(input, (Const::<896>,)),
            dev.tensor(lbl),
        )
    };

	println!("Epoch\tTrain Loss\tTest Loss");
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
            let loss = cross_entropy_with_logits_loss(logits, label);

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
            let loss = cross_entropy_with_logits_loss(logits, label);

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

    //mlp.save("conv_model.npz").unwrap();
}

pub fn encode(board: Board, out: &mut [f32], flip: bool) {
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

    fn to_index(sq: chess::Square, flip: bool) -> usize {
        if flip {
			// https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
            sq.to_index() ^ 56
        } else {
            sq.to_index()
        }
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

    let mut movegen = MoveGen::new_legal(&board);
    movegen.set_iterator_mask(*black); // calculate attacks
    for mov in movegen {
        out[to_index(mov.get_dest(), flip) + 64 * 12] = 1.0;
    }

    let board = board.null_move();
    if let Some(board) = board {
        let mut movegen = MoveGen::new_legal(&board);
        movegen.set_iterator_mask(*white); // calculate attacks
        for mov in movegen {
            out[to_index(mov.get_dest(), flip) + 64 * 13] = 1.0;
        }
    }
}

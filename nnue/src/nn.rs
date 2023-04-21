use dfdx::{data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor_ops::Backward};

use chess::*;
use indicatif::ProgressIterator;

use rand::prelude::{SeedableRng, StdRng};
use std::str::FromStr;

mod clipped_relu;
pub use clipped_relu::*;

pub type Device = dfdx::tensor::AutoDevice;

pub type FeatureTransformer<const TRANSFORMED_SIZE: usize> =
    (Linear<768, TRANSFORMED_SIZE>, ClippedReLU, DropoutOneIn<4>);

pub type Model<const TRANSFORMED_SIZE: usize> = (
    // feature transformer
    FeatureTransformer<TRANSFORMED_SIZE>,
    (Linear<TRANSFORMED_SIZE, 1>, Tanh),
);

pub type BuiltModel = (
    (
        modules::Linear<768, 256, f32, Cpu>,
        ClippedReLU,
        DropoutOneIn<4>,
    ),
    (modules::Linear<256, 1, f32, Cpu>, Tanh),
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

/// Calculate material imbalance
pub fn eval<E: Encodable>(board: &E) -> i32 {
    let pawns = board.pieces(Piece::Pawn);
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    let rooks = board.pieces(Piece::Rook);
    let queens = board.pieces(Piece::Queen);

    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    let eval = (((pawns & white).popcnt() as i32 * 100) - ((pawns & black).popcnt() as i32 * 100))
        + (((knights & white).popcnt() as i32 * 300) - ((knights & black).popcnt() as i32 * 300))
        + (((bishops & white).popcnt() as i32 * 305) - ((bishops & black).popcnt() as i32 * 305))
        + (((rooks & white).popcnt() as i32 * 500) - ((rooks & black).popcnt() as i32 * 500))
        + (((queens & white).popcnt() as i32 * 900) - ((queens & black).popcnt() as i32 * 900));
    eval
}

/// Train a new or existing neural network, using the given model name, data path, test/train split, learning rate, and Nesterov momentum.
/// Enable the `cuda` feature flag to use a GPU.
pub fn train(
    model_name: &str,
    data: &str,
    test: &str,
    bootstrap: bool,
    lr: f32,
    l2_weight_decay: f32,
    epochs: usize,
) {
    let dev = Device::default();
    println!("[TRAINER] Using device: {:?}", dev);
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<Model<256>, f32>();
    if bootstrap {
        model
            .load(model_name)
            .expect("model corrupted or not found");
    }
    eprintln!(
        "[TRAINER]: Number of trainable parameters: {:.2}k",
        model.num_trainable_params() as f32 / 1000 as f32
    );
    let mut grads = model.alloc_grads();

    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: lr,
            weight_decay: if l2_weight_decay > 0.0 {
                Some(WeightDecay::L2(l2_weight_decay))
            } else {
                None
            },
            ..Default::default()
        },
    );

    // read csv
    eprintln!("[TRAINER] Loading & encoding data...");
    let file = std::fs::File::open(test).expect("file not found");
    let mut rdr = csv::Reader::from_reader(file);

    let mut test_positions = Positions {
        input: vec![],
        labels: vec![],
    };

    for result in rdr.records() {
        let record = result.unwrap();
        let board = Board::from_str(&record[0]).expect("bad fen");

		if board.null_move().is_none() {
			continue;
		}

        let eval = if let Ok(eval) = record[1].parse::<i32>() {
            eval.clamp(-1500, 1500)
        } else {
            continue;
        };

        let mut input = vec![0f32; 768];
        encode(&board, &mut input);
        test_positions.input.push(input);

        test_positions.labels.push(eval as f32 / 1500.0);
    }

    eprintln!("[TRAINER] Done! Uploading data...");
    eprintln!();
    eprintln!();

    const BATCH_SIZE: usize = 64;
    const BATCHES_IN_MEM: usize = BATCH_SIZE * 10000;

    let preprocess = |(input, lbl): <Positions as ExactSizeDataset>::Item<'_>| {
        (
            dev.tensor_from_vec(input, (Const::<768>,)),
            dev.tensor([lbl]),
        )
    };

    for i_epoch in 0..epochs {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;

        let file = std::fs::File::open(data).expect("file not found");
        let mut rdr = csv::Reader::from_reader(file);
        let mut records = rdr.records();
		let mut skipped = 0usize;
        loop {
            let mut last = false;
            let mut num_training_steps = 0;

            let mut train_positions = Positions {
                input: vec![],
                labels: vec![],
            };

            for _ in 0..BATCHES_IN_MEM {
                let record = records.next();
                if let Some(record) = record {
                    let record = record.expect("failed to read record");
                    let board = Board::from_str(&record[0]).expect("bad fen");

					if board.null_move().is_none() {
						skipped += 1;
						continue;
					}

                    let eval = if let Ok(eval) = record[1].parse::<i32>() {
                        eval.clamp(-1500, 1500)
                    } else {
                        continue;
                    };

                    let mut input = vec![0f32; 768];
                    encode(&board, &mut input);
                    train_positions.input.push(input);

                    train_positions.labels.push(eval as f32 / 1500.0);
                } else {
                    last = true;
                    break;
                }
            }

            for (input, label) in train_positions
                .shuffled(&mut rng)
                .map(preprocess)
                .batch_exact(Const::<BATCH_SIZE>)
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

            eprint!("{}{}", up(), erase());
            eprintln!(
                "[TRAINER] Running epoch {i_epoch} (training steps={num_batches}): Loss={:.5}",
                BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            );

            if last {
                break;
            }
        }

        model.save(model_name).unwrap();

        // test model
        let mut test_total_epoch_loss = 0.0;
        let mut test_num_batches = 0;
        for (input, label) in test_positions
            .shuffled(&mut rng)
            .map(preprocess)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
        {
            let logits = model.forward(input);
            let loss = mse_loss(logits, label);

            test_total_epoch_loss += loss.array();
            test_num_batches += 1;
        }

        eprint!("{}{}", up(), erase());
        eprintln!(
            "[TRAINER] Epoch {i_epoch}\tLoss: {:.5}\tTest Loss: {:.5}\tSkipped {} positions",
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            BATCH_SIZE as f32 * test_total_epoch_loss / test_num_batches as f32,
			skipped
        );
        eprintln!();

        if (BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32) <= 0.01 {
            break;
        }
    }
}

fn up() -> String {
    format!("{}[A", ESC)
}

fn erase() -> String {
    format!("{}[2K", ESC)
}

const ESC: char = 27u8 as char;

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

pub fn pair_to_index(piece: chess::Square, offset: usize) -> usize {
    piece.to_index() + (offset * 64)
    //piece.to_index() + (64 * king.to_index()) + (offset * 4096)
}

pub fn vertical_flip(x: BitBoard, is_black: bool) -> BitBoard {
    if is_black {
        let mut x = x.0;
        let k1 = 0x00FF00FF00FF00FF;
        let k2 = 0x0000FFFF0000FFFF;
        x = ((x >> 8) & k1) | ((x & k1) << 8);
        x = ((x >> 16) & k2) | ((x & k2) << 16);
        x = (x >> 32) | (x << 32);
        BitBoard(x)
    } else {
        x
    }
}

pub fn encode<E: Encodable>(board: &E, out: &mut [f32]) {
    let is_black = board.side_to_move() == Color::Black;
    let pawns = vertical_flip(board.pieces(Piece::Pawn), is_black);
    let knights = vertical_flip(board.pieces(Piece::Knight), is_black);
    let bishops = vertical_flip(board.pieces(Piece::Bishop), is_black);
    let rooks = vertical_flip(board.pieces(Piece::Rook), is_black);
    let queens = vertical_flip(board.pieces(Piece::Queen), is_black);
    let kings = vertical_flip(board.pieces(Piece::King), is_black);
    let mut white = vertical_flip(board.color_combined(Color::White), is_black);
    let mut black = vertical_flip(board.color_combined(Color::Black), is_black);

    if is_black {
        std::mem::swap(&mut white, &mut black);
    }

    //////////////////// pawns ////////////////////

    let mut remaining = white & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 0)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 1)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 2)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 3)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 4)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 5)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 6)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 7)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 8)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 9)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 10)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, 11)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }
}

#[derive(Clone)]
struct Layer {
    weights: Vec<i16>,
    biases: Vec<i16>,
    activations: Vec<i16>, // used for incremental layer
}

impl Layer {
    pub fn new(weights: Vec<f32>, biases: Vec<f32>) -> Self {
        Self {
            weights: weights
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            biases: biases
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            activations: biases
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct DoubleAccumulatorNNUE {
    input_weights: Vec<i16>,
    original_biases: Vec<i16>,
    // double accumulator architecture
    white_input_activations: Vec<i16>,
    black_input_activations: Vec<i16>,
    hidden_layer: Layer,
}

impl DoubleAccumulatorNNUE {
    pub fn from_built_model(net: &BuiltModel) -> Self {
        Self {
            input_weights: net
                .0
                 .0
                .weight
                .clone()
                .permute()
                .as_vec()
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            original_biases: net
                .0
                 .0
                .bias
                .as_vec()
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            white_input_activations: net
                .0
                 .0
                .bias
                .as_vec()
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            black_input_activations: net
                .0
                 .0
                .bias
                .as_vec()
                .iter()
                .map(|x| (x * SCALE as f32).round() as i16)
                .collect(),
            // no permutation is needed for 256x1 weight
            hidden_layer: Layer::new(net.1 .0.weight.as_vec(), net.1 .0.bias.as_vec()),
        }
    }

    pub fn reset(&mut self) {
        self.white_input_activations = self.original_biases.clone();
        self.black_input_activations = self.original_biases.clone();
    }

    #[inline(always)]
    pub fn activate(&mut self, piece: Piece, color: Color, sq: chess::Square) {
        // white accumulator
        let feature_idx = ((piece.to_index() * 2 + color.to_index()) * 64 + sq.to_index())
            * self.original_biases.len();
        let weights =
            self.input_weights[feature_idx..feature_idx + self.original_biases.len()].iter();

        self.white_input_activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation += weight);

        // black accumulator
        let feature_idx = ((piece.to_index() * 2 + (!color).to_index()) * 64
            + (sq.to_index() ^ 56))
            * self.original_biases.len();
        let weights =
            self.input_weights[feature_idx..feature_idx + self.original_biases.len()].iter();

        self.black_input_activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation += weight);
    }

    #[inline(always)]
    pub fn deactivate(&mut self, piece: Piece, color: Color, sq: chess::Square) {
        // white accumulator
        let feature_idx = ((piece.to_index() * 2 + color.to_index()) * 64 + sq.to_index())
            * self.original_biases.len();
        let weights =
            self.input_weights[feature_idx..feature_idx + self.original_biases.len()].iter();

        self.white_input_activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation -= weight);

        // black accumulator
        let feature_idx = ((piece.to_index() * 2 + (!color).to_index()) * 64
            + (sq.to_index() ^ 56))
            * self.original_biases.len();
        let weights =
            self.input_weights[feature_idx..feature_idx + self.original_biases.len()].iter();

        self.black_input_activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation -= weight);
    }

    pub fn eval(&self, side_to_play: Color) -> i32 {
        let mut output = self.hidden_layer.biases[0] as i32;

        let weights = self.hidden_layer.weights.iter();

        let activations = match side_to_play {
            Color::White => &self.white_input_activations,
            Color::Black => &self.black_input_activations,
        };

        activations
            .iter()
            .map(|x| Self::clipped_relu(*x))
            .zip(weights)
            .for_each(|(clipped_activation, weight)| {
                output += (clipped_activation as i32) * (*weight as i32)
            });
        ((output as f32 / (SCALE as f32 * SCALE as f32)).tanh() * 1500.0).round() as i32
    }

    #[inline(always)]
    fn clipped_relu(x: i16) -> i16 {
        x.clamp(0, SCALE)
    }

    pub fn activate_all<E: Encodable>(&mut self, board: &E) {
        let pawns = board.pieces(Piece::Pawn);
        let knights = board.pieces(Piece::Knight);
        let bishops = board.pieces(Piece::Bishop);
        let rooks = board.pieces(Piece::Rook);
        let queens = board.pieces(Piece::Queen);
        let kings = board.pieces(Piece::King);

        let white = board.color_combined(Color::White);
        let black = board.color_combined(Color::Black);

        //////////////////// pawns ////////////////////

        let mut remaining = white & pawns;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Pawn, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & pawns;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Pawn, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// knights ////////////////////

        let mut remaining = white & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Knight, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Knight, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// bishops ////////////////////

        let mut remaining = white & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Bishop, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Bishop, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// rooks ////////////////////

        let mut remaining = white & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Rook, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Rook, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// queens ////////////////////

        let mut remaining = white & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Queen, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::Queen, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// kings ////////////////////

        let mut remaining = white & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::King, Color::White, sq);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(Piece::King, Color::Black, sq);

            remaining ^= BitBoard::from_square(sq);
        }
    }
}

const SCALE: i16 = 1024;

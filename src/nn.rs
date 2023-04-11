use dfdx::{data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor_ops::Backward};

use chess::*;
use indicatif::ProgressIterator;

use rand::prelude::{SeedableRng, StdRng};
use std::str::FromStr;

mod clipped_relu;
pub use clipped_relu::*;

pub type Device = dfdx::tensor::Cpu;

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

    let mut model = dev.build_module::<Model<256>, f32>();

    println!(
        "Number of trainable parameters: {:.2}k",
        model.num_trainable_params() as f32 / 1000 as f32
    );
    let mut grads = model.alloc_grads();

    let mut opt = Sgd::new(
        &model,
        SgdConfig {
            lr: 0.001,
            momentum: Some(Momentum::Nesterov(0.9)),
            //weight_decay: Some(WeightDecay::L2(0.0001)),
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

    const BATCH_SIZE: usize = 256;

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

        model.save("sparse_mlp.npz").unwrap();

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

pub fn pair_to_index(piece: chess::Square, king: chess::Square, offset: usize) -> usize {
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

    let white_king = (kings | white).to_square();
    let black_king = (kings | black).to_square();

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

        out[pair_to_index(sq, white_king, 0)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & pawns;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 1)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// knights ////////////////////

    let mut remaining = white & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, white_king, 2)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & knights;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 3)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// bishops ////////////////////

    let mut remaining = white & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, white_king, 4)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & bishops;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 5)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// rooks ////////////////////

    let mut remaining = white & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, white_king, 6)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & rooks;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 7)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// queens ////////////////////

    let mut remaining = white & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, white_king, 8)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & queens;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 9)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    //////////////////// kings ////////////////////

    let mut remaining = white & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, white_king, 10)] = 1.0;

        remaining ^= BitBoard::from_square(sq);
    }

    let mut remaining = black & kings;
    while remaining != BitBoard(0) {
        let sq = remaining.to_square();

        out[pair_to_index(sq, black_king, 11)] = 1.0;

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
            weights: weights.iter().map(|x| (x * 256.0).round() as i16).collect(),
            biases: biases.iter().map(|x| (x * 256.0).round() as i16).collect(),
            activations: biases.iter().map(|x| (x * 256.0).round() as i16).collect(),
        }
    }
}

#[derive(Clone)]
pub struct ProphetNetwork {
    input_layer: Layer,
    hidden_layer: Layer,
}

impl ProphetNetwork {
    pub fn from_built_model(net: &BuiltModel) -> Self {
        Self {
            input_layer: Layer::new(net.0 .0.weight.clone().permute().as_vec(), net.0 .0.bias.as_vec()),
            hidden_layer: Layer::new(net.1 .0.weight.as_vec(), net.1 .0.bias.as_vec()),
        }
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

            self.activate(sq, 0);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & pawns;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 1);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// knights ////////////////////

        let mut remaining = white & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 2);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 3);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// bishops ////////////////////

        let mut remaining = white & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 4);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 5);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// rooks ////////////////////

        let mut remaining = white & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 6);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 7);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// queens ////////////////////

        let mut remaining = white & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 8);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 9);

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// kings ////////////////////

        let mut remaining = white & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 10);

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();

            self.activate(sq, 11);

            remaining ^= BitBoard::from_square(sq);
        }
    }

    // #[inline(always)]
    // pub fn move_piece(&mut self, piece: Piece, color: Color, from_sq: chess::Square, to_sq: chess::Square) {
    //     self.deactivate(piece, color, from_sq);
    //     self.activate(piece, color, to_sq);
    // }

    #[inline(always)]
    pub fn activate(&mut self, sq: chess::Square, offset: usize) {
        let feature_idx =
            pair_to_index(sq, sq, offset) * self.input_layer.activations.len();
        dbg!(feature_idx);
        let weights = self.input_layer.weights
            [feature_idx..feature_idx + self.input_layer.activations.len()]
            .iter();

        self.input_layer
            .activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation += weight);
    }

    #[inline(always)]
    pub fn deactivate(&mut self, piece: Piece, color: Color, sq: chess::Square) {
        let feature_idx =
        ((piece.to_index()*2 + color.to_index()) * 64 + sq.to_index()) * self.input_layer.activations.len();
        let weights = self.input_layer.weights
            [feature_idx..feature_idx + self.input_layer.activations.len()]
            .iter();

        self.input_layer
            .activations
            .iter_mut()
            .zip(weights)
            .for_each(|(activation, weight)| *activation -= weight);
    }

    pub fn eval(&self) -> f32 {
        //let bucket = (board.all_pieces().pop_count() as usize - 1) / 4;
        //let bucket_idx = bucket * self.input_layer.activations.len();
        //println!("{:?}", &self.hidden_layer.weights.len());
        let mut output = self.hidden_layer.biases[0] as i32;

        let weights = self.hidden_layer.weights.iter();

        self.input_layer
            .activations
            .iter()
            .map(|x| Self::clipped_relu(*x))
            .zip(weights)
            .for_each(|(clipped_activation, weight)| {
                output += (clipped_activation as i32) * (*weight as i32)
            });
        (output as f32 / (256.0*256.0)).tanh()
        //output / (Self::SCALE * Self::SCALE) as i32
    }

    #[inline(always)]
    fn clipped_relu(x: i16) -> i16 {
        x.clamp(0, 256)
    }
}

impl ProphetNetwork {
    const SCALE: i16 = 256;
}

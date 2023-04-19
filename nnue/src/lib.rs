use std::ffi::CStr;
use std::io::Cursor;
use std::os::raw::c_char;

use dfdx::prelude::*;

use chess::*;

mod nn;

#[repr(C)]
pub struct ProphetBoard {
    white: u64,
    black: u64,

    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,

    side_to_move: u8,
}

impl nn::Encodable for ProphetBoard {
    fn pieces(&self, piece: Piece) -> BitBoard {
        match piece {
            Piece::Pawn => BitBoard(self.pawns),
            Piece::Knight => BitBoard(self.knights),
            Piece::Bishop => BitBoard(self.bishops),
            Piece::Rook => BitBoard(self.rooks),
            Piece::Queen => BitBoard(self.queens),
            Piece::King => BitBoard(self.kings),
        }
    }

    fn color_combined(&self, color: Color) -> BitBoard {
        match color {
            Color::White => BitBoard(self.white),
            Color::Black => BitBoard(self.black),
        }
    }

    fn side_to_move(&self) -> Color {
        match self.side_to_move {
            0 => Color::White,
            1 => Color::Black,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

pub struct Prophet {
    net: nn::BuiltModel,
    dev: nn::Device,
    nnue: nn::DoubleAccumulatorNNUE,
}

/// Raise the Prophet. The Prophet shall not be freed.
///
/// If `net_path` is null, the default net will be used.
#[no_mangle]
pub unsafe extern "C" fn raise_prophet(net_path: *const c_char) -> *mut Prophet {
    let dev = nn::Device::default();
    let mut model = dev.build_module::<nn::Model<256>, f32>();
    if net_path.is_null() {
        let reader = Cursor::new(include_bytes!("../nnue.npz"));
        let mut zip = zip::ZipArchive::new(reader).expect("failed to parse archive");
        model
            .read(&mut zip)
            .expect("failed to read default model archive");
    } else {
        let path = CStr::from_ptr(net_path);
        let path = path.to_str().unwrap();
        model.load(path).expect("failed to load model");
    }
    let painter = Box::new(Prophet {
        dev,
        nnue: nn::DoubleAccumulatorNNUE::from_built_model(&model),
        net: model,
    });
    Box::into_raw(painter)
}

/// Let the Prophet die for our sins.
#[no_mangle]
pub unsafe extern "C" fn prophet_die_for_sins(prophet: *mut Prophet) {
    drop(Box::from_raw(prophet));
}

/// Evaluate a position in full accuracy (no NNUE)
#[no_mangle]
pub unsafe extern "C" fn prophet_sing_evaluation(prophet: &Prophet, board: &ProphetBoard) -> i32 {
    let mut board_tensor = vec![0f32; 768];
    nn::encode(board, &mut board_tensor);
    let test_tensor = prophet.dev.tensor_from_vec(board_tensor, (Const::<768>,));
    let logits = prophet.net.forward(test_tensor);
    let eval = (logits.array()[0] * 1500.0) as i32;
    eval
}

/// Activate a piece on the accumulators
#[no_mangle]
pub unsafe extern "C" fn prophet_activate(prophet: &mut Prophet, piece: i32, color: i32, sq: i32) {
    prophet.nnue.activate(
        match piece {
            0 => Piece::Pawn,
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            5 => Piece::King,
            _ => unsafe { std::hint::unreachable_unchecked() },
        },
        match color {
            0 => Color::White,
            1 => Color::Black,
            _ => unsafe { std::hint::unreachable_unchecked() },
        },
        chess::Square::new(sq as u8),
    );
}

/// Deactivate a piece on the accumulators
#[no_mangle]
pub unsafe extern "C" fn prophet_deactivate(
    prophet: &mut Prophet,
    piece: i32,
    color: i32,
    sq: i32,
) {
    prophet.nnue.deactivate(
        match piece {
            0 => Piece::Pawn,
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            5 => Piece::King,
            _ => unsafe { std::hint::unreachable_unchecked() },
        },
        match color {
            0 => Color::White,
            1 => Color::Black,
            _ => unsafe { std::hint::unreachable_unchecked() },
        },
        chess::Square::new(sq as u8),
    );
}

/// Activate all the pieces on a board
#[no_mangle]
pub unsafe extern "C" fn prophet_activate_all(prophet: &mut Prophet, board: ProphetBoard) {
    prophet.nnue.activate_all(&board);
}

/// Activate all the pieces on a board
#[no_mangle]
pub unsafe extern "C" fn prophet_reset(prophet: &mut Prophet) {
    prophet.nnue.reset();
}

/// Evaluate the NNUE network
#[no_mangle]
pub unsafe extern "C" fn prophet_utter_evaluation(prophet: &mut Prophet, side_to_play: u8) -> i32 {
    prophet.nnue.eval(match side_to_play {
        0 => Color::White,
        1 => Color::Black,
        _ => unsafe { std::hint::unreachable_unchecked() },
    })
}

/// Print board
// #[no_mangle]
// pub unsafe extern "C" fn prophet_sing_gospel(prophet: &mut Prophet, color: u8) {
//     prophet.nnue.print_fen(match color {
//         0 => Color::White,
//         1 => Color::Black,
//         _ => unsafe { std::hint::unreachable_unchecked() },
//     });
// }

/// Train a new or existing neural network, using the given model name, data path, test/train split, learning rate, and L2 regularization (weight decay).
/// Enable the `cuda` feature flag to use a GPU.
#[no_mangle]
pub unsafe extern "C" fn prophet_train(
    model_name: *const c_char,
    dataset: *const c_char,
    testset: *const c_char,
    bootstrap: bool,
    lr: f32,
    l2_weight_decay: f32,
    epochs: usize,
) {
    let model_name = CStr::from_ptr(model_name);
    let model_name = model_name.to_str().unwrap();

    let dataset = CStr::from_ptr(dataset);
    let dataset = dataset.to_str().unwrap();

    let testset = CStr::from_ptr(testset);
    let testset = testset.to_str().unwrap();

    nn::train(
        model_name,
        dataset,
        testset,
        bootstrap,
        lr,
        l2_weight_decay,
        epochs,
    );
}

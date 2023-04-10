use std::mem::MaybeUninit;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::c_char;

use dfdx::{
    data::*, losses::mse_loss, nn::SaveToNpz, optim::*, prelude::*, tensor_ops::Backward,
};

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
			_ => unsafe { std::hint::unreachable_unchecked() }
		}
	}
}

pub struct Prophet {
	net: nn::BuiltModel,
	dev: nn::Device,
}

/// Raise the Prophet. The Prophet shall not be freed.
#[no_mangle]
pub unsafe extern "C" fn RaiseProphet(net_path: *const c_char) -> *mut Prophet {
	let path = CStr::from_ptr(net_path);
    let path = path.to_str().unwrap();
	let dev = nn::Device::default();
    let mut model = dev.build_module::<nn::Model<256, 32, 32>, f32>();
    model.load(path).unwrap();
	let painter = Box::new(Prophet {
		dev,
		net: model,
	});
    Box::into_raw(painter)
}

#[no_mangle]
pub unsafe extern "C" fn ProphetSingEvaluation(
	prophet: &Prophet,
	board: *const ProphetBoard
) -> i32 {
	let board = &(*board);
	let mut board_tensor = vec![0f32; 768];
	nn::encode(board, &mut board_tensor);
	let test_tensor = prophet.dev.tensor_from_vec(board_tensor, (Const::<768>,));
	let logits = prophet.net.forward(test_tensor);
	let eval = (logits.array()[0] * 100.0) as i32 + (nn::eval(board) * 100);
	eval
}

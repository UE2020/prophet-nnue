use chess::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct TranspositionTableEntry {
    pub flag: TranspositionFlag,
    pub value: i32,
    pub depth: u8,
    pub mov: ChessMove,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub enum TranspositionFlag {
    #[default]
    Exact,
    Lowerbound,
    Upperbound,
}

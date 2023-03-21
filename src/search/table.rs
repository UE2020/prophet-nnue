use chess::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TranspositionTableEntry {
    pub flag: TranspositionFlag,
    pub value: i32,
    pub depth: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum TranspositionFlag {
    Exact,
    Lowerbound,
    Upperbound,
}

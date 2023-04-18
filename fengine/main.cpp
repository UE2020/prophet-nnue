#include "dirichlet.hpp"
#include "surge/src/position.h"
#include "surge/src/tables.h"
#include "surge/src/types.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>

namespace po = boost::program_options;

enum GameProgress {
    MIDGAME,
    ENDGAME
};

constexpr int piece_values[NPIECE_TYPES] = {
    100,    // PAWN
    300,    // KNIGHT
    305,    // BISHOP
    500,    // ROOK
    900,    // QUEEN
    2000000 // KING
};

GameProgress get_progress(int mv1, int mv2) {
    return (mv1 <= 1300 && mv2 <= 1300) ? ENDGAME : MIDGAME;
}

template <Color Us>
int evaluate(const Position& pos) {
    // Material value
    int mv = 0;
    int our_mv = 0;
    int their_mv = 0;
    for (PieceType i = PAWN; i < NPIECE_TYPES - 1; ++i) {
        our_mv += pop_count(pos.bitboard_of(Us, i)) * piece_values[i];
        their_mv += pop_count(pos.bitboard_of(~Us, i)) * piece_values[i];
    }
    mv += our_mv;
    mv -= their_mv;
    GameProgress progress = get_progress(our_mv, their_mv);

    // Color advantage
    int ca = 0;
    if (progress == MIDGAME) {
        ca = (Us == WHITE) ? 15 : -15;
    }

    // Center control
    int cc = 0;
    if (pos.at(d5) != NO_PIECE) cc += (color_of(pos.at(d5)) == Us) ? 25 : -25;
    if (pos.at(e5) != NO_PIECE) cc += (color_of(pos.at(e5)) == Us) ? 25 : -25;
    if (pos.at(d4) != NO_PIECE) cc += (color_of(pos.at(d4)) == Us) ? 25 : -25;
    if (pos.at(e4) != NO_PIECE) cc += (color_of(pos.at(e4)) == Us) ? 25 : -25;

    // Knight placement
    const static Bitboard edges_mask = MASK_FILE[AFILE] | MASK_RANK[RANK1] | MASK_FILE[HFILE] | MASK_RANK[RANK8];
    int np = 0;
    np -= pop_count(pos.bitboard_of(Us, KNIGHT) & edges_mask) * 50;
    np += pop_count(pos.bitboard_of(~Us, KNIGHT) & edges_mask) * 50;

    // Bishop placement
    static constexpr Bitboard black_squares = 0xAA55AA55AA55AA55;
    int bp = 0;
    for (Color color = WHITE; color < NCOLORS; ++color) {
        unsigned short white_square_count = 0;
        unsigned short black_square_count = 0;

        Bitboard bishops = pos.bitboard_of(color, BISHOP);
        while (bishops) {
            Square bishop = pop_lsb(&bishops);

            if ((black_squares >> bishop) & 1) {
                black_square_count++;
            } else {
                white_square_count++;
            }

            if (white_square_count && black_square_count) {
                bp += color == Us ? 50 : -50;
                break;
            }
        }
    }

    // Rook placement
    int rp = 0;
    rp += pop_count(pos.bitboard_of(Us, ROOK) & MASK_RANK[Us == WHITE ? RANK7 : RANK2]) * 30;
    rp -= pop_count(pos.bitboard_of(~Us, ROOK) & MASK_RANK[~Us == WHITE ? RANK7 : RANK2]) * 30;

    // King placement
    /*
    function distance(x1, y1, x2, y2) {
        return Math.hypot(x2 - x1, y2 - y1);
    }

    let table = [];
    for (let y = 7; y > -1; y--) {
        for (let x = 0; x < 8; x++) {
            table.push(Math.round(distance(x, y, 3.5, 3.5) * 20));
        }
    }

    table[0] = Math.round(distance(1, 6, 3.5, 3.5) * 20);
    table[7] = Math.round(distance(6, 6, 3.5, 3.5) * 20);
    table[56] = Math.round(distance(1, 1, 3.5, 3.5) * 20);
    table[63] = Math.round(distance(6, 1, 3.5, 3.5) * 20);
    */
    static constexpr int king_pcsq_table[64] = {71, 86, 76, 71, 71, 76, 86, 71, 86, 71, 58, 51, 51, 58, 71, 86, 76, 58, 42, 32, 32, 42, 58, 76, 71, 51, 32, 14, 14, 32, 51, 71, 71, 51, 32, 14, 14, 32, 51, 71, 76, 58, 42, 32, 32, 42, 58, 76, 86, 71, 58, 51, 51, 58, 71, 86, 71, 86, 76, 71, 71, 76, 86, 71};
    int kp = 0;
    if (progress == MIDGAME) {
        kp += king_pcsq_table[bsf(pos.bitboard_of(Us, KING))];
        kp -= king_pcsq_table[bsf(pos.bitboard_of(~Us, KING))];
    }

    // Doubled pawns
    int dp = 0;
    for (File file = AFILE; file < NFILES; ++file) {
        dp -= std::max(pop_count(pos.bitboard_of(Us, PAWN) & MASK_FILE[file]) - 1, 0) * 75;
        dp += std::max(pop_count(pos.bitboard_of(~Us, PAWN) & MASK_FILE[file]) - 1, 0) * 75;
    }

    // Passed pawns
    int pp = 0;
    for (Color color = WHITE; color < NCOLORS; ++color) {
        Bitboard pawns = pos.bitboard_of(color, PAWN);
        while (pawns) {
            Square sq = pop_lsb(&pawns);

            Bitboard pawns_ahead_mask = MASK_FILE[file_of(sq)];
            if (file_of(sq) > AFILE) {
                pawns_ahead_mask |= MASK_FILE[file_of(sq) - 1];
            }
            if (file_of(sq) < HFILE) {
                pawns_ahead_mask |= MASK_FILE[file_of(sq) + 1];
            }

            if (color == WHITE) {
                for (Rank rank = RANK1; rank <= rank_of(sq); ++rank) {
                    pawns_ahead_mask &= ~MASK_RANK[rank];
                }
            } else if (color == BLACK) {
                for (Rank rank = RANK8; rank >= rank_of(sq); --rank) {
                    pawns_ahead_mask &= ~MASK_RANK[rank];
                }
            } else {
                throw std::logic_error("Invalid color");
            }

            if (!(pos.bitboard_of(~color, PAWN) & pawns_ahead_mask)) {
                if (progress == MIDGAME) {
                    pp += color == Us ? 30 : -30;
                } else if (progress == ENDGAME) {
                    int score = 0;
                    if (color == WHITE) {
                        score = (rank_of(sq) - RANK1) * 50;
                    } else if (color == BLACK) {
                        score = (RANK8 - rank_of(sq)) * 50;
                    } else {
                        throw std::logic_error("Invalid color");
                    }
                    pp += color == Us ? score : -score;
                } else {
                    throw std::logic_error("Invalid game progress");
                }
            }
        }
    }

    // Isolated pawns
    int ip = 0;
    if (progress == MIDGAME) {
        for (Color color = WHITE; color < NCOLORS; ++color) {
            Bitboard pawns = pos.bitboard_of(color, PAWN);
            while (pawns) {
                Square sq = pop_lsb(&pawns);

                Bitboard buddies_mask = 0;
                if (file_of(sq) > AFILE) {
                    buddies_mask |= MASK_FILE[file_of(sq) - 1];
                }
                if (file_of(sq) < HFILE) {
                    buddies_mask |= MASK_FILE[file_of(sq) + 1];
                }

                if (!(pos.bitboard_of(color, PAWN) & buddies_mask)) {
                    ip += color == Us ? -15 : 15;
                }
            }
        }
    }

    // Open files
    int of = 0;
    for (Color color = WHITE; color < NCOLORS; ++color) {
        Bitboard rooks = pos.bitboard_of(color, ROOK);
        while (rooks) {
            Square sq = pop_lsb(&rooks);
            File file = file_of(sq);

            Bitboard pawns[NCOLORS] = {
                pos.bitboard_of(WHITE_PAWN) & MASK_FILE[file],
                pos.bitboard_of(BLACK_PAWN) & MASK_FILE[file],
            };

            if (pawns[~color]) {
                of += color == Us ? -5 : 5; // File is half-open
                if (pawns[color]) {
                    of += color == Us ? -5 : 5; // File is closed
                }
            }
        }
    }

    // Check status
    int cs = 0;
    if (pos.in_check<Us>()) {
        cs = -20;
    } else if (pos.in_check<~Us>()) {
        cs = 20;
    }

    // Sum up various scores
    return mv + ca + cc + np + bp + rp + kp + dp + pp + ip + of + cs;
}

template <Color Us, typename RNG>
void descend(Position& pos, RNG& generator, std::unordered_set<std::string>& fens, int max_plies = 60, double noise_weight = 1) {
    if (pos.game_ply > max_plies) {
        return;
    }

    Move moves[218];
    Move* last_move = pos.generate_legals<Us>(moves);
    size_t move_count = last_move - moves;

    if (!move_count) {
        return;
    } else if (move_count == 1) {
        pos.play<Us>(moves[0]);
        fens.insert(pos.fen());
        descend<~Us>(pos, generator, fens, max_plies, noise_weight);
        pos.undo<Us>(moves[0]);
        return;
    }

    int static_evaluations[move_count];
    for (Move* move = moves; move != last_move; move++) {
        pos.play<Us>(*move);
        static_evaluations[move - moves] = evaluate<Us>(pos);
        pos.undo<Us>(*move);
    }

    int max = *std::max_element(static_evaluations, static_evaluations + move_count);
    double sum = 0;
    for (int static_evaluation : static_evaluations) {
        sum += std::exp(static_evaluation - max);
    }

    double probabilities[move_count];
    std::transform(static_evaluations, static_evaluations + move_count, probabilities, [max, sum](int static_evaluation) {
        return std::exp(static_evaluation - max) / sum;
    });

    DirichletDistribution<RNG> dirichlet(std::vector<double>(move_count, 0.3));
    auto noise_probabilities = dirichlet(generator);
    for (size_t i = 0; i < move_count; i++) {
        probabilities[i] = probabilities[i] * (1 - noise_weight) + noise_weight * noise_probabilities[i];
    }

    double* max_probability = std::max_element(probabilities, probabilities + move_count);
    pos.play<Us>(moves[max_probability - probabilities]);
    fens.insert(pos.fen());
    descend<~Us>(pos, generator, fens, max_plies, noise_weight);
    pos.undo<Us>(moves[max_probability - probabilities]);
}

void print_help(po::options_description& desc, char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] [port]\n\n"
              << desc;
}

int main(int argc, char* argv[]) {
    int game_count;
    int max_plies;
    double noise_weight;
    std::string path;

    po::options_description desc("Options");
    po::variables_map vm;
    desc.add_options()("help,h", "Show this help message and exit")("games,n", po::value(&game_count)->default_value(250000), "Number of games to play out")("max-plies,m", po::value(&max_plies)->default_value(60), "Max amount of plies per game")("noise-weight", po::value(&noise_weight)->default_value(0.95), "Noise weight")("output,o", po::value(&path)->required(), "Output file path");
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            print_help(desc, argv[0]);
            return 0;
        }
    } catch (std::exception& e) {
        if (vm.count("help")) {
            print_help(desc, argv[0]);
            return 0;
        } else {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    std::cout << "Using # games: " << game_count << std::endl;
    std::cout << "Using max plies: " << max_plies << std::endl;
    std::cout << "Using noise weight: " << noise_weight << std::endl;
    std::cout << "Using output file: " << path << std::endl;
    std::cout << std::endl;

    initialise_all_databases();
    zobrist::initialise_zobrist_keys();

    std::unordered_set<std::string> fens;
    Position pos(DEFAULT_FEN);
    std::random_device rd;
    std::mt19937 mt(rd());

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < game_count; i++) {
        descend<WHITE>(pos, mt, fens, max_plies, noise_weight);
        auto current_time = std::chrono::high_resolution_clock::now();
        if ((i % 100) == 0) {
            std::cout << "\33[2KPlayed " << i << " games (" << ((float) i / std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time).count()) * 1'000'000 << " games/s)\r" << std::flush;
        }
    }

    size_t fen_count = fens.size();
    std::ostringstream ss;
    ss << "FEN" << std::endl;
    for (auto it = fens.begin(); it != fens.end(); it = fens.erase(it)) {
        ss << *it << std::endl;
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << path << std::endl;
        return 1;
    }
    file << ss.str();
    file.close();

    std::cout << "Fengine finished, wrote " << fen_count << " chess positions." << std::endl;
    return 0;
}

#pragma once

/* Warning, this file is autogenerated by cbindgen. Don't modify this manually. */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct Prophet Prophet;

typedef struct ProphetBoard {
    uint64_t white;
    uint64_t black;
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;
    uint8_t side_to_move;
} ProphetBoard;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Activate a piece on the accumulators
 */
void prophet_activate(struct Prophet *prophet, int32_t piece, int32_t color, int32_t sq);

/**
 * Activate all the pieces on a board
 */
void prophet_activate_all(struct Prophet *prophet, struct ProphetBoard board);

/**
 * Deactivate a piece on the accumulators
 */
void prophet_deactivate(struct Prophet *prophet, int32_t piece, int32_t color, int32_t sq);

/**
 * Let the Prophet die for our sins.
 */
void prophet_die_for_sins(struct Prophet *prophet);

/**
 * Activate all the pieces on a board
 */
void prophet_reset(struct Prophet *prophet);

/**
 * Evaluate a position in full accuracy (no NNUE)
 */
int32_t prophet_sing_evaluation(const struct Prophet *prophet, const struct ProphetBoard *board);

/**
 * Print board
 */
void prophet_sing_gospel(const struct Prophet *prophet);

/**
 * Train a new or existing neural network, using the given model name, data path, test/train split, learning rate, and L2 regularization (weight decay).
 * Enable the `cuda` feature flag to use a GPU.
 */
void prophet_train(const char *model_name,
                   const char *dataset,
                   const char *testset,
                   bool bootstrap,
                   float lr,
                   float l2_weight_decay,
                   size_t epochs);

/**
 * Evaluate the NNUE network
 */
int32_t prophet_utter_evaluation(struct Prophet *prophet, uint8_t side_to_play);

/**
 * Raise the Prophet. The Prophet shall not be freed.
 *
 * If `net_path` is null, the default net will be used.
 */
struct Prophet *raise_prophet(const char *net_path);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

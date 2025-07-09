#ifndef OBSERVATION_TOKEN_HPP_
#define OBSERVATION_TOKEN_HPP_
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "types.hpp"

// We want empty tokens to be 0xff, since 0s are very natural numbers to have in the observations, and we want
// empty to be obviously different.
const uint8_t EmptyTokenByte = 0xff;

using Layer = ObservationType;
using TypeId = ObservationType;
using ObservationCoord = ObservationType;

struct PartialObservationToken {
  ObservationType feature_id;
  ObservationType value;
};

static_assert(sizeof(PartialObservationToken) == 2 * sizeof(ObservationType), "PartialObservationToken size check");

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  ObservationType location;
  ObservationType feature_id;
  ObservationType value;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3, "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

/**
 * Utilities for packing/unpacking grid coordinates into compact byte representation.
 *
 * Provides space-efficient coordinate storage for contexts where memory is at a premium
 * (e.g., observation tokens). This is a compressed alternative to GridLocation when
 * the layer component is not needed and coordinates fit in 4 bits each.
 *
 * Packing scheme:
 * - Upper 4 bits: row (r / y-coordinate)
 * - Lower 4 bits: col (c / x-coordinate)
 * - Special value 0xFF represents empty/invalid coordinate
 */
namespace PackedCoordinate {

// Constants for bit packing
constexpr uint8_t ROW_SHIFT = 4;
constexpr uint8_t COL_MASK = 0x0F;
constexpr uint8_t ROW_MASK = 0xF0;

// Maximum coordinate value that can be packed (4 bits = 0-14)
constexpr uint8_t MAX_PACKABLE_COORD = 14;

/**
 * Packs grid coordinates (row, col) into a single byte.
 *
 * @param row Row coordinate (0–14). Must be ≤ MAX_PACKABLE_COORD.
 * @param col Column coordinate (0–14). Must be ≤ MAX_PACKABLE_COORD.
 * @return A packed byte representing the (row, col) coordinate.
 *
 * @note The value 0xFF is reserved to represent an empty or invalid coordinate.
 * @note In debug builds, this function uses assert() to check that row and col are within bounds.
 *       In release builds, these checks are omitted for performance.
 *
 * @warning Supplying out-of-range coordinates in release builds will result in undefined behavior.
 */
inline uint8_t pack(uint8_t row, uint8_t col) {
  assert(row <= MAX_PACKABLE_COORD && "row exceeds MAX_PACKABLE_COORD");
  assert(col <= MAX_PACKABLE_COORD && "col exceeds MAX_PACKABLE_COORD");
  return static_cast<uint8_t>((row << ROW_SHIFT) | (col & COL_MASK));
}

/**
 * Unpack byte into coordinates with empty handling.
 *
 * @param packed Packed coordinate byte
 * @return std::optional<std::pair<row, col>> or std::nullopt if empty
 */
inline std::optional<std::pair<uint8_t, uint8_t>> unpack(uint8_t packed) {
  if (packed == EmptyTokenByte) {
    return std::nullopt;
  }
  uint8_t row = (packed & ROW_MASK) >> ROW_SHIFT;
  uint8_t col = packed & COL_MASK;
  return {{row, col}};
}

/**
 * Check if a packed coordinate represents an empty/invalid position.
 */
inline bool is_empty(uint8_t packed_data) {
  return packed_data == EmptyTokenByte;
}
}  // namespace PackedCoordinate

#endif  // OBSERVATION_TOKEN_HPP_

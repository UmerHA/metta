#ifndef GRID_OBJECT_HPP_
#define GRID_OBJECT_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "observation_token.hpp"
#include "types.hpp"

class GridLocation {
public:
  GridCoord r;
  GridCoord c;
  Layer layer;

  inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
  inline GridLocation() : r(0), c(0), layer(0) {}
};

enum Orientation {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3
};

using GridObjectId = unsigned int;

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;

  GridObjectConfig(TypeId type_id, const std::string& type_name) : type_id(type_id), type_name(type_name) {}

  virtual ~GridObjectConfig() = default;
};

class GridObject {
public:
  GridObjectId id;
  GridLocation location;
  TypeId type_id;
  std::string type_name;

  virtual ~GridObject() = default;

  void init(TypeId object_type_id, const std::string& object_type_name, const GridLocation& object_location) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->location = object_location;
  }

  virtual std::vector<PartialObservationToken> obs_features() const = 0;
};

#endif  // GRID_OBJECT_HPP_

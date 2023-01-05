#pragma once

#include "types.h"

namespace Prop::Cell {
template <typename T>
struct BasicCell2D {
  virtual T getEz()      = 0;
  virtual T getHx()      = 0;
  virtual T getHy()      = 0;
  virtual ~BasicCell2D() = 0;
};

template <typename T>
struct HexagonalCell : BasicCell2D<T> {
  T Ez;
  T H1;
  T H2;
  T H3;
  void HexagonalCell() : Ez(0.0), H1(0.0), H2(0.0), H3(0.0){};
  T getEz() override { return Ez; };
  T getHx() override { return H1 - H3; };
  T getHy() override { return H2; };
};
};  // namespace Prop::Cell

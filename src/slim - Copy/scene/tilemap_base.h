#pragma once

#include "../math/vec2.h"


#define MAX_TILE_MAP_VIEW_DISTANCE 42
#define MAX_TILE_MAP_WIDTH 32
#define MAX_TILE_MAP_HEIGHT 32
#define MAX_TILE_MAP_SIZE (MAX_TILE_MAP_WIDTH * MAX_TILE_MAP_HEIGHT)
#define MAX_TILE_MAP_VERTICES ((MAX_TILE_MAP_WIDTH + 1) * (MAX_TILE_MAP_HEIGHT + 1))
#define MAX_TILE_MAP_EDGES (MAX_TILE_MAP_WIDTH * (MAX_TILE_MAP_HEIGHT + 1) + MAX_TILE_MAP_HEIGHT * (MAX_TILE_MAP_WIDTH + 1))

#define MAX_COLUMN_COUNT 16

#define FACING_UP      (1 << 0)
#define FACING_DOWN    (1 << 1)
#define FACING_LEFT    (1 << 2)
#define FACING_RIGHT   (1 << 3)
#define ABOVE          (1 << 4)
#define BELOW          (1 << 5)
#define ON_THE_LEFT    (1 << 5)
#define ON_THE_RIGHT   (1 << 7)


struct Circle {
    vec2 position;
    f32 radius;
};

struct LocalEdge {
    vec2 from;
    vec2 to;
    u8 is;
    u8 texture_id;
    // LocalEdge * portal_to;
};
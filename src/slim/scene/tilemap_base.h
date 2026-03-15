#pragma once

#include "tilemap.h"
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

#define FACING_MASK    (FACING_UP | FACING_DOWN | FACING_LEFT | FACING_RIGHT)


struct Is {
    bool facing_up : 1;
    bool facing_down : 1;
    bool facing_left : 1;
    bool facing_right : 1;
    bool above : 1;
    bool below : 1;
    bool on_the_right : 1;
    bool on_the_left : 1;
};

struct Circle {
    vec2 position;
    f32 radius;
};

struct TileEdge {
    vec2i from{};
    vec2i to{};
    // i32 length = 0;
    // i32 portal_ray_rotation = 0;
    // TileEdge* portal_to;
    // bool portal_edge_dir_flip;
    u8 texture_id = 0;
    u8 is = 0;

    INLINE_XPU u8 isVisible(const vec2& origin) const {
        u8 is_visible = 0;
        if (is & (FACING_LEFT | FACING_RIGHT)) {
            if (from.x > origin.x) is_visible |= ON_THE_RIGHT;
            if (to.x   < origin.x) is_visible |= ON_THE_LEFT;
        } else {
            if (to.y   < origin.y) is_visible |= ABOVE;
            if (from.y > origin.y) is_visible |= BELOW;
        }

        if (!(
            is & FACING_LEFT  && is_visible & ON_THE_RIGHT ||
            is & FACING_RIGHT && is_visible & ON_THE_LEFT ||
            is & FACING_DOWN  && is_visible & ABOVE ||
            is & FACING_UP    && is_visible & BELOW))
            is_visible = 0;

        return is_visible;
    }
};
#pragma once

#include "../math/vec2.h"


#define MAX_TILE_MAP_VIEW_DISTANCE 42
#define MAX_TILE_MAP_WIDTH 32
#define MAX_TILE_MAP_HEIGHT 32
#define MAX_TILE_MAP_SIZE (MAX_TILE_MAP_WIDTH * MAX_TILE_MAP_HEIGHT)
#define MAX_TILE_MAP_VERTICES ((MAX_TILE_MAP_WIDTH + 1) * (MAX_TILE_MAP_HEIGHT + 1))
#define MAX_TILE_MAP_EDGES (MAX_TILE_MAP_WIDTH * (MAX_TILE_MAP_HEIGHT + 1) + MAX_TILE_MAP_HEIGHT * (MAX_TILE_MAP_WIDTH + 1))

#define MAX_COLUMN_COUNT 16


enum struct Placed : u8 {
    NotVisible,
    Above,
    Below,
    OnTheLeft,
    OnTheRight
};


enum struct Rounding : u8 {
    None,
    Convex,
    Concave
};

enum struct Facing : u8 {
    NotApplicable,
    Up,
    Down,
    Left,
    Right
};


struct Corner {
    Rounding rounding{Rounding::None};
    Facing horizontal{Facing::NotApplicable};
    Facing vertical{Facing::NotApplicable};
};


INLINE_XPU bool isHorizontal(Facing facing) {
    return facing == Facing::Up || facing == Facing::Down;
}


INLINE_XPU bool isVertical(Facing facing) {
    return facing == Facing::Left || facing == Facing::Right;
}


struct Circle {
    vec2 position;
    f32 radius;
};

struct TileEdge {
    vec2i from{};
    vec2i to{};
    Facing facing{Facing::NotApplicable};
    Corner from_corner{};
    Corner to_corner{};
    u8 texture_id = 0;

    INLINE_XPU Placed isVisible(const vec2& origin) const {
        Placed placed = Placed::NotVisible;
        if (isVertical(facing)) {
            if (from.x > origin.x) placed = Placed::OnTheRight;
            if (to.x   < origin.x) placed = Placed::OnTheLeft;
        } else {
            if (to.y   < origin.y) placed = Placed::Above;
            if (from.y > origin.y) placed = Placed::Below;
        }

        if (!(
            (facing == Facing::Left  && placed == Placed::OnTheRight) ||
            (facing == Facing::Right && placed == Placed::OnTheLeft) ||
            (facing == Facing::Down  && placed == Placed::Above) ||
            (facing == Facing::Up    && placed == Placed::Below)))
            placed = Placed::NotVisible;

        return placed;
    }

    bool overlaps(const TileEdge& other) const {
        if (other.facing != facing)
            return false;

        if (isVertical(facing)) {
            if (from.x > other.to.x || other.from.x > to.x)
                return false;
        } else {
            if (from.y > other.to.y || other.from.y > to.y)
                return false;
        }
        return true;
    }
};

    //
    // INLINE_XPU u8 intersectsWithEdge(const TileEdge& edge, RayHit& hit, const f32 rounding_radius = 0.2f) {
    //     u8 is_visible = edge.isVisible(origin);
    //     if (is_visible == 0)
    //         return 0;
    //
    //     hit.edge_is = edge.is;
    //     if (edge.is & (FACING_LEFT | FACING_RIGHT)) {
    //         if (is_vertical ||
    //             (is_facing_right && (is_visible & ON_THE_LEFT)) ||
    //             (is_facing_left && (is_visible & ON_THE_RIGHT)))
    //             return 0;
    //
    //         hit.position = (f32)edge.to.x - origin.x;
    //         hit.position.y *= rise_over_run;
    //         hit.position += origin;
    //
    //         if (inRange((f32)edge.from.y, hit.position.y, (f32)edge.to.y)) {
    //             hit.distance = hit.position.y - edge.from.y;
    //             if (hit.distance < rounding_radius) {
    //                 hit.edge_is |= edge.from_corner & CONCAVE ? CONCAVE : CONVEX;
    //                 hit.center = edge.from;
    //                 hit.center.y += rounding_radius;
    //                 hit.center.x +=
    //                     ((edge.is & FACING_RIGHT) && (edge.from_corner & FACING_DOWN)) ||
    //                     ((edge.is & FACING_LEFT)  && (edge.from_corner & FACING_UP)) ?
    //                     rounding_radius :
    //                     -rounding_radius;
    //             } else {
    //                 hit.distance = edge.to.y - hit.position.y;
    //                 if (hit.distance < rounding_radius) {
    //                     hit.edge_is |= edge.to_corner & CONCAVE ? CONCAVE : CONVEX;
    //                     hit.center = edge.to;
    //                     hit.center.y -= rounding_radius;
    //                     hit.center.x +=
    //                         ((edge.is & FACING_RIGHT) && (edge.to_corner & FACING_UP)) ||
    //                         ((edge.is & FACING_LEFT)  && (edge.to_corner & FACING_DOWN)) ?
    //                         rounding_radius :
    //                         -rounding_radius;
    //                 }
    //             }
    //
    //             if (hit.edge_is & (CONVEX | CONCAVE)) {
    //                 if (intersectsWithCircle(hit.center, rounding_radius, hit.distance, hit.edge_is & CONCAVE))
    //                     hit.position = origin + direction * hit.distance;
    //                 else
    //                     return 0;
    //             }
    //
    //             return is_visible;
    //         }
    //
    //         return 0;
    //     }
    //
    //     // Edge is horizontal:
    //     if (is_horizontal ||
    //         (is_facing_up && (is_visible & BELOW)) ||
    //         (is_facing_down && (is_visible & ABOVE)))
    //         return 0;
    //
    //     hit.position = (f32)edge.to.y - origin.y;
    //     hit.position.x *= run_over_rise;
    //     hit.position += origin;
    //
    //     if (inRange((f32)edge.from.x, hit.position.x, (f32)edge.to.x)) {
    //         hit.distance = hit.position.x - edge.from.x;
    //         if (hit.distance < rounding_radius) {
    //             hit.edge_is |= edge.from_corner & CONCAVE ? CONCAVE : CONVEX;
    //             hit.center = edge.from;
    //             hit.center.x += rounding_radius;
    //             hit.center.y +=
    //                 ((edge.is & FACING_DOWN) && (edge.from_corner & FACING_RIGHT)) ||
    //                 ((edge.is & FACING_UP)   && (edge.from_corner & FACING_LEFT)) ?
    //                 rounding_radius :
    //                 -rounding_radius;
    //         } else {
    //             hit.distance = edge.to.x - hit.position.x;
    //             if (hit.distance < rounding_radius) {
    //                 hit.edge_is |= edge.to_corner & CONCAVE ? CONCAVE : CONVEX;
    //                 hit.center = edge.to;
    //                 hit.center.x -= rounding_radius;
    //                 hit.center.y +=
    //                     ((edge.is & FACING_DOWN) && (edge.to_corner & FACING_LEFT)) ||
    //                     ((edge.is & FACING_UP)   && (edge.to_corner & FACING_RIGHT)) ?
    //                     rounding_radius :
    //                     -rounding_radius;
    //             }
    //         }
    //
    //         if (hit.edge_is & (CONVEX | CONCAVE)) {
    //             if (intersectsWithCircle(hit.center, rounding_radius, hit.distance, hit.edge_is & CONCAVE))
    //                 hit.position = origin + direction * hit.distance;
    //             else
    //                 return 0;
    //         }
    //
    //         return is_visible;
    //     }
    //     return 0;
    // }
#pragma once

#include "./render_data.h"


INLINE_XPU bool inRange(i32 start, i32 value, i32 end) { return value >= start && value <= end; }
INLINE_XPU bool inRange(f32 start, f32 value, f32 end) { return value >= start && value <= end; }
INLINE_XPU u8 min(u8 a, u8 b) { return a < b ? a : b; }
INLINE_XPU u8 max(u8 a, u8 b) { return a > b ? a : b; }
INLINE_XPU u8 clamp(u8 v, u8 min_v, u8 max_v) { return min(max(v, min_v), max_v); }
INLINE_XPU f32 clamp(f32 v, f32 min_v, f32 max_v) { return fminf(fmaxf(v, min_v), max_v); }
INLINE_XPU u8 closestLog2(u32 v) {
    u8 r = 0;
    while (v) {r++; v >>= 1;}
    return r;
}
INLINE_XPU u8 computeMip(f32 pixel_coverage, f32 texel_size, u8 last_mip) {
    return pixel_coverage < texel_size ? 0 : min(last_mip, closestLog2((u32)(pixel_coverage / texel_size) - 1));
}
INLINE_XPU bool inRange(vec2 start, vec2 value, vec2 end) {
    return inRange(start.x, value.x, end.x) &&
           inRange(start.y, value.y, end.y);
}
INLINE_XPU f32 getU(vec2 v) {
    f32 u = v.y / v.x;
    if (u > 1.0f || u < -1.0f) u = -1.0f / u;
    return u + 1.0f;
}

struct RayHit {
    vec2i tile_coords;
    vec2 position;

    f32 distance;
    f32 perp_distance;
    f32 texture_u;

    u16 edge_id;
    u8 column_id;
    u8 texture_id;
    u8 edge_is;

    INLINE_XPU void init() {
        column_id = INVALID_COLUMN_ID;
        edge_id = INVALID_EDGE_ID;
    }

    INLINE_XPU bool isValid() {
        return column_id != INVALID_COLUMN_ID ||
               edge_id != INVALID_EDGE_ID;
    }

    INLINE_XPU void finalizeFromEdge(const TileEdge& edge, const vec2 ray_origin, const vec2 forward) {
        texture_id = edge.texture_id;

        perp_distance = fmaxf(0.001f, forward.dot(position - ray_origin));

        tile_coords.x = edge_is & FACING_RIGHT ? (i32)position.x - 1 : (i32)position.x;
        tile_coords.y = edge_is & FACING_DOWN  ? (i32)position.y - 1 : (i32)position.y;

        texture_u = edge_is & (FACING_LEFT | FACING_RIGHT) ?
            position.y - (f32)edge.from.y :
            position.x - (f32)edge.from.x;

        texture_u -= (f32)(i32)texture_u;

        if (edge_is & (FACING_RIGHT | FACING_UP))
            texture_u = 1.0f - texture_u;
    }

    INLINE_XPU void finalizeFromColumn(const Circle& column, const vec2 ray_origin, const vec2 ray_direction, const vec2 forward) {
        position = ray_direction * distance;
        perp_distance = fmaxf(0.001f, forward.dot(position));
        position += ray_origin;
        tile_coords.x = (i32)position.x;
        tile_coords.y = (i32)position.y;
        texture_u = getU(position - column.position);
        texture_u *= column.radius;
        texture_id = 12;
        edge_is = 0;
    }
};

struct GroundHit {
    f32 z;
    u8 mip;
    u8 flags;
};

struct WallHit {
    vec2 ray_direction, hit_position, hit_normal;
    f32 u, v, texel_step;
    u16 top, bot, edge_id;
    u8 texture_id;
    u8 mip;
    u8 edge_is;
    u8 column_id;

    INLINE_XPU void init() {
        v = -1.0f;
    }

    INLINE_XPU bool isValid() const {
        return v >= 0.0f;
    }

    INLINE_XPU void update(u16 screen_height, f32 texel_size, f32 pixel_coverage_factor, f32 column_height_factor, u8 last_mip, vec2 new_ray_direction, i32 mid_point, const Circle* columns, const RayHit &ray_hit) {
        ray_direction = new_ray_direction;
        texture_id = ray_hit.texture_id;

        u = ray_hit.texture_u;
        v = 0.0f;

        f32 height = column_height_factor / ray_hit.perp_distance;
        f32 half_height = height * 0.5f;
        mip = computeMip(ray_hit.perp_distance * pixel_coverage_factor, texel_size, last_mip);
        texel_step = 1.0f / height;
        i32 ibot = mid_point + (i32)half_height;
        bot = ibot >= (i32)screen_height ? (screen_height - 1) : (u16)ibot;

        if (mid_point < half_height) {
            v = (half_height - mid_point) / height;
            top    = 0;
        }
        else
            top = (u16)(mid_point - half_height);

        edge_is = ray_hit.edge_is;
        edge_id = ray_hit.edge_id;
        column_id = ray_hit.column_id;
        hit_position = ray_hit.position;
        if (column_id != INVALID_COLUMN_ID) {
            hit_normal = (hit_position - columns[column_id].position).normalized();
        }
    }
};

struct WallHitGroup {
    WallHit main_hit;
    WallHit portal_hits[MAX_PORTAL_DEPTH];
    vec2 portal_origins[MAX_PORTAL_DEPTH];
};

struct Ray {
    RayHit hit;
    vec2 origin;
    vec2 direction;
    vec2 forward;

    f32 rise_over_run;
    f32 run_over_rise;

    bool is_vertical;
    bool is_horizontal;
    bool is_facing_up;
    bool is_facing_down;
    bool is_facing_left;
    bool is_facing_right;

    INLINE_XPU void update(vec2 new_origin, vec2 new_direction, vec2 new_forward) {
        origin = new_origin;
        direction = new_direction.normalized();
        forward = new_forward;
        is_vertical     = direction.x == 0;
        is_horizontal   = direction.y == 0;
        is_facing_left  = direction.x < 0;
        is_facing_up    = direction.y < 0;
        is_facing_right = direction.x > 0;
        is_facing_down  = direction.y > 0;
        rise_over_run = direction.y / direction.x;
        run_over_rise = 1 / rise_over_run;
        hit.init();
    }

    INLINE_XPU void finalizeHit(const TileEdge *edges, const Circle* columns, const f32 offset = 0.0f) {
        hit.distance = sqrtf(hit.distance) + offset;
        if (hit.column_id != INVALID_COLUMN_ID)
            hit.finalizeFromColumn(columns[hit.column_id], origin, direction, forward);
        else
            hit.finalizeFromEdge(edges[hit.edge_id], origin, forward);
    }

    INLINE_XPU bool intersectsWithCircle(const Circle& circle) {
        vec2 C = circle.position - origin;
        f32 t = C.dot(direction);
        if (t > 0.0f) {
            f32 dt = circle.radius * circle.radius - (direction * t - C).squaredLength();
            if (dt > 0.0f && t*t > dt) { // Inside the sphere
                t -= sqrt(dt);
                t *= t;
                if (t < hit.distance) {
                    hit.distance = t;
                    return true;
                }
            }
        }

        return false;
    }

    INLINE_XPU void intersectWithEdgePlane(const TileEdge& edge) {
        if (edge.is & (FACING_LEFT | FACING_RIGHT)) {
            hit.position = (f32)edge.to.x - origin.x;
            hit.position.y *= rise_over_run;
            hit.position += origin;
        } else {
            hit.position = (f32)edge.to.y - origin.y;
            hit.position.x *= run_over_rise;
            hit.position += origin;
        }
    }

    INLINE_XPU u8 intersectsWithEdge(const TileEdge& edge) {
        u8 is_visible = edge.isVisible(origin);
        if (is_visible == 0)
            return 0;

        if (edge.is & (FACING_LEFT | FACING_RIGHT)) {
            if (is_vertical ||
                (is_facing_right && (is_visible & ON_THE_LEFT)) ||
                (is_facing_left && (is_visible & ON_THE_RIGHT)))
                return 0;

            hit.position = (f32)edge.to.x - origin.x;
            hit.position.y *= rise_over_run;
            hit.position += origin;

            if (inRange((f32)edge.from.y, hit.position.y, (f32)edge.to.y)) {
                hit.edge_is = edge.is | is_visible;
                return is_visible;
            }

            return 0;
        }

        // Edge is horizontal:
        if (is_horizontal ||
            (is_facing_up && (is_visible & BELOW)) ||
            (is_facing_down && (is_visible & ABOVE)))
            return 0;

        hit.position = (f32)edge.to.y - origin.y;
        hit.position.x *= run_over_rise;
        hit.position += origin;

        if (inRange((f32)edge.from.x, hit.position.x, (f32)edge.to.x)) {
            hit.edge_is = edge.is | is_visible;
            return is_visible;
        }
        return 0;
    }
};


struct Portal {
    Color color;
    vec3 position;
    f32 radius;
    f32 spawned_time;
    u16 projectile_index;
    u16 edge_id;
    u8 edge_is;

    INLINE_XPU void init() {
        spawned_time = 0.0f;
        position = vec3{0.0f};
        radius = 0.0f;
        projectile_index = INVALID_PROJECTILE_INDEX;
        edge_id = INVALID_EDGE_ID;
        edge_is = 0;
    }

    INLINE_XPU bool isActive() const {
        return edge_id != INVALID_EDGE_ID;
    }

    INLINE_XPU bool isHit(const RayHit& ray_hit) const {
        if (ray_hit.edge_id != edge_id)
            return false;

        if (edge_is & (FACING_DOWN | FACING_UP))
            return fabsf(position.x - ray_hit.position.x) < PORTAL_FINAL_RADIUS;

        return fabsf(position.z - ray_hit.position.y) < PORTAL_FINAL_RADIUS;
    }

    INLINE_XPU bool containsWallPosition2D(const vec2& wall_position_2d, u16 wall_edge_id, f32 y_factor = 0.5f) const {
        return wall_edge_id == edge_id && (
            wall_position_2d - vec2{position.x, position.z}
        ).squaredLength() < (radius * radius);
    }

    INLINE_XPU bool containsWallPosition3D(const vec3& wall_position_3d, u16 wall_edge_id, f32 y_factor = 0.5f) const {
        return wall_edge_id == edge_id && (
            vec3{wall_position_3d.x, wall_position_3d.y * y_factor, wall_position_3d.z} -
            vec3{position.x, position.y * y_factor, position.z}
        ).squaredLength() < (radius * radius);
    }

    INLINE_XPU i32 getRotation(u8 to_edge_is) const {
        i32 ray_rotation = 0;
        if (edge_is & (FACING_LEFT | FACING_RIGHT)) {
            if (to_edge_is & (FACING_DOWN | FACING_UP)) {
                if (edge_is & FACING_RIGHT)
                    ray_rotation = (to_edge_is & FACING_UP) ? 90 : -90;
                else
                    ray_rotation = (to_edge_is & FACING_DOWN) ? 90 : -90;
            } else if ((edge_is & FACING_RIGHT) == (to_edge_is & FACING_RIGHT))
                ray_rotation = 180;
        } else
            if (to_edge_is & (FACING_LEFT | FACING_RIGHT)) {
                if (edge_is & FACING_UP)
                    ray_rotation = (to_edge_is & FACING_LEFT) ? 90 : -90;
                else
                    ray_rotation = (to_edge_is & FACING_RIGHT) ? 90 : -90;
            } else
                if ((edge_is & FACING_UP) == (to_edge_is & FACING_UP))
                    ray_rotation = 180;

        return ray_rotation;
    }

    void update(const f32 time, const TileEdge* edges) {
        const f32 elapsed_time = time - spawned_time;
        if (elapsed_time <= PORTAL_GROW_TIME) {
            radius = PORTAL_GROW_RADIUS +
                            PORTAL_GROW_RANGE * smoothStep(0.0f, 1.0f, PORTAL_GROW_RATE * elapsed_time);
            if ((fabsf(position.y) + (radius * 2.0f) + PORTAL_BREATHING_RANGE) > 1.0f)
                position.y = (1.0f - (radius * 2.0f) - PORTAL_BREATHING_RANGE) * (position.y > 0.0f ? 1.0f : -1.0f);

            const TileEdge& edge{edges[edge_id]};
            if (edge.is & (FACING_DOWN | FACING_UP)) {
                if ((position.x + radius + PORTAL_BREATHING_RANGE) > edge.to.x)
                    position.x = edge.to.x - radius - PORTAL_BREATHING_RANGE;
                else if ((position.x - radius - PORTAL_BREATHING_RANGE) < edge.from.x)
                    position.x = edge.from.x + radius + PORTAL_BREATHING_RANGE;
            } else {
                if ((position.z + radius + PORTAL_BREATHING_RANGE) > edge.to.y)
                    position.z = edge.to.y - radius - PORTAL_BREATHING_RANGE;
                else if ((position.z - radius - PORTAL_BREATHING_RANGE) < edge.from.y)
                    position.z = edge.from.y + radius + PORTAL_BREATHING_RANGE;
            }
        } else
            radius = PORTAL_FINAL_RADIUS + PORTAL_BREATHING_RANGE * cos((elapsed_time - PORTAL_GROW_TIME) * 2.0f);
    }

    INLINE_XPU vec2 teleportTo(const Portal& other_portal, const vec2& origin, i32& rotation) const {
        rotation = getRotation(other_portal.edge_is);
        vec2 origin_to_portal = vec2{position.x, position.z} - origin;
        origin_to_portal.rotateBy(rotation);
        return vec2{other_portal.position.x, other_portal.position.z} - origin_to_portal;
    }
};


struct Portals {
    Portal from{Cyan};
    Portal to{Magenta};

    INLINE_XPU bool areBothActive() const {
        return from.isActive() && to.isActive();
    }

    INLINE_XPU const Portal* getPortalsFromRayHit(const RayHit& ray_hit, const Portal** other_portal) const {
        if (ray_hit.edge_id != INVALID_EDGE_ID) {
            if (from.isHit(ray_hit)) {
                *other_portal = &to;
                return &from;
            }

            if (to.isHit(ray_hit)) {
                *other_portal = &from;
                return &to;
            }
        }

        *other_portal = nullptr;
        return nullptr;
    }

    INLINE_XPU const Portal* getPortalsFromWallPosition2D(const vec2& wall_position_2d, u16 wall_edge_id, const Portal** other_portal) const {
        if (wall_edge_id != INVALID_EDGE_ID) {
            if (from.containsWallPosition2D(wall_position_2d, wall_edge_id)) {
                *other_portal = &to;
                return &from;
            }

            if (to.containsWallPosition2D(wall_position_2d, wall_edge_id)) {
                *other_portal = &from;
                return &to;
            }
        }

        *other_portal = nullptr;
        return nullptr;
    }

    INLINE_XPU const Portal* getPortalsFromWallPosition3D(const vec3& wall_position, u16 wall_edge_id, const Portal** other_portal, f32 y_factor = 0.5f) const {
        if (wall_edge_id != INVALID_EDGE_ID) {
            if (from.containsWallPosition3D(wall_position, wall_edge_id, y_factor)) {
                *other_portal = &to;
                return &from;
            }

            if (to.containsWallPosition3D(wall_position, wall_edge_id, y_factor)) {
                *other_portal = &from;
                return &to;
            }
        }

        *other_portal = nullptr;
        return nullptr;
    }

    Portal* getPortalsFromProjectileIndex(u8 projectile_index, const Portal** other_portal) {
        if (projectile_index != INVALID_PROJECTILE_INDEX) {
            if (from.projectile_index == projectile_index) {
                *other_portal = &to;
                return &from;
            }

            if (to.projectile_index == projectile_index) {
                *other_portal = &from;
                return &to;
            }
        }

        *other_portal = nullptr;
        return nullptr;
    }

    void update(const f32 time, const TileEdge* edges) {
        if (from.isActive()) from.update(time, edges);
        if (to.isActive()) to.update(time, edges);
    }
};

struct RayCast {
    Portals portals;
    RayHit closest_hit;
    Ray ray;
    vec2 position;
    vec2 forward;
    vec2 right_step;
    vec2 first_ray_direction;
    i32 mid_point;
    u16 screen_width;
    u16 screen_height;
    f32 texel_size;
    f32 pixel_coverage_factor;
    f32 column_height_factor;
    u8 last_mip;

    INLINE_XPU bool castRay(const vec2& ray_origin, const vec2& ray_direction, const vec2& ray_forward, const Slice<TileEdge> &edges, const Slice<Circle> &columns, const bool check_columns = true) {
        closest_hit.init();
        closest_hit.distance = 10000000;

        ray.update(ray_origin, ray_direction, ray_forward);

        for (u16 i = 0; i < (u16)edges.size; i++) {
            if (ray.intersectsWithEdge(edges.data[i])) {
                ray.hit.distance = (ray.hit.position - ray.origin).squaredLength();
                if (ray.hit.distance < closest_hit.distance) {
                    closest_hit = ray.hit;
                    closest_hit.edge_id = i;
                }
            }
        }

        ray.hit = closest_hit;

        if (check_columns) {
            for (u8 i = 0; i < (u8)columns.size; i++) {
                if (ray.intersectsWithCircle(columns[i])) {
                    ray.hit.column_id = i;
                    ray.hit.edge_id = INVALID_EDGE_ID;
                    ray.hit.edge_is = 0;
                }
            }
        }

        return ray.hit.isValid();
    }

    INLINE_XPU void generateWallHit(WallHitGroup &wall_hit_group, vec2 ray_direction, const Slice<TileEdge> &edges, const Slice<Circle> &columns) {
        wall_hit_group.main_hit.init();
        if (castRay(position, ray_direction, forward, edges, columns))
            ray.finalizeHit(edges.data, columns.data);
        else
            return;

        wall_hit_group.main_hit.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray_direction, mid_point, columns.data, ray.hit);

        if (!portals.areBothActive())
            return;

        vec2 prior_forward = forward;
        vec2 prior_position = position;
        for (u8 p = 0; p < MAX_PORTAL_DEPTH; p++) {
            WallHit& portal_hit{wall_hit_group.portal_hits[p]};
            vec2& portal_origin{wall_hit_group.portal_origins[p]};

            const Portal* other_portal = nullptr;
            const Portal* portal = portals.getPortalsFromRayHit(ray.hit, &other_portal);

            if (portal == nullptr)
                return;

            i32 rotation = 0;
            portal_origin = portal->teleportTo(*other_portal, prior_position, rotation);
            vec2 ray_forward = prior_forward.rotatedBy(rotation);
            vec2 origin_to_hit_position = (ray.hit.position - prior_position).rotatedBy(rotation);
            ray_direction.rotateBy(rotation);

            f32 prior_distance = ray.hit.distance;
            portal_hit.init();
            if (castRay(portal_origin + origin_to_hit_position, ray_direction, ray_forward, edges, columns)) {
                ray.origin = portal_origin;
                ray.finalizeHit(edges.data, columns.data, prior_distance);
                portal_hit.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray_direction, mid_point, columns.data, ray.hit);
            }
            prior_position = portal_origin;
            prior_forward = ray_forward;
        }
    }
};
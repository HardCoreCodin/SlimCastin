#pragma once

#include "./render_data.h"


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

    INLINE_XPU void finalizeHit(const TileEdge *edges, const Circle* columns) {
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


struct RayCaster {
    Portal portal_from;
    Portal portal_to;
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
    f32 up_aim;
    f32 up_aim_over_focal_length;
    u8 last_mip;

    void onScreenChanged(const f32 focal_length, vec2 new_forward, vec2 right, f32 new_up_aim) {
        right = right.normalized() * ((f32)screen_width / (f32)screen_height);
        forward = new_forward.normalized();
        right_step = right / (f32)screen_width;
        column_height_factor = 2.0f * focal_length * (f32)screen_height;
        pixel_coverage_factor = 2.0f * focal_length / (f32)screen_height;
        first_ray_direction = focal_length * forward + right_step * (0.5f - 0.5f * (f32)screen_width);
        up_aim = new_up_aim;
        up_aim_over_focal_length = up_aim / focal_length;
        mid_point = (i32)((1.0f + up_aim) * (f32)(screen_height >> 1));
    }

    INLINE_XPU void generateWallHit(WallHit &wall_hit, const Slice<TileEdge> &edges, const Slice<Circle> &columns) {
        closest_hit.init();
        closest_hit.distance = 10000000;
        wall_hit.init();

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

        for (u8 i = 0; i < (u8)columns.size; i++)
            if (ray.intersectsWithCircle(columns[i]))
                ray.hit.column_id = i;
    }

    INLINE_XPU void generateWallHitWithPortals(WallHitGroup &wall_hit_group, vec2 ray_direction, const Slice<TileEdge> &edges, const Slice<Circle> &columns) {
        ray.update(position, ray_direction, forward);
        generateWallHit(wall_hit_group.main, edges, columns);
        if (!ray.hit.isValid())
            return;

        ray.hit.distance  = sqrt(ray.hit.distance);
        ray.finalizeHit(edges.data, columns.data);
        wall_hit_group.main.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray_direction, mid_point, columns.data, ray.hit);

        if (portal_from.edge_id == INVALID_EDGE_ID ||
            portal_to.edge_id == INVALID_EDGE_ID)
            return;

        const Portal* portal = nullptr;
        if (ray.hit.edge_id == portal_from.edge_id && fabsf(
            (portal_from.edge_is & (FACING_DOWN | FACING_UP)) ?
            (portal_from.position.x - ray.hit.position.x) :
            (portal_from.position.z - ray.hit.position.y)) < PORTAL_FINAL_RADIUS)
            portal = &portal_from;
        if (portal == nullptr && ray.hit.edge_id == portal_to.edge_id && fabsf(
            (portal_to.edge_is & (FACING_DOWN | FACING_UP)) ?
            (portal_to.position.x - ray.hit.position.x) :
            (portal_to.position.z - ray.hit.position.y)) < PORTAL_FINAL_RADIUS)
            portal = &portal_to;

        if (portal == nullptr)
            return;

        const Portal& other_portal{portal == &portal_from ? portal_to : portal_from};
        const vec2 other_portal_position{other_portal.position.x, other_portal.position.z};

        i32 ray_rotation = portal->getRotation(other_portal.edge_is);

        vec2 origin_to_portal = vec2{portal->position.x, portal->position.z} - position;
        vec2 origin_to_hit_position = ray.hit.position - position;
        vec2 ray_forward = forward;
        if (ray_rotation == 90) {
            ray_direction = ray_direction.ccw90();
            ray_forward = forward.ccw90();
            origin_to_hit_position = origin_to_hit_position.ccw90();
            origin_to_portal = origin_to_portal.ccw90();
        } else if (ray_rotation == -90) {
            ray_direction = ray_direction.cw90();
            ray_forward = forward.cw90();
            origin_to_hit_position = origin_to_hit_position.cw90();
            origin_to_portal = origin_to_portal.cw90();
        } else if (ray_rotation == 180) {
            ray_direction = -ray_direction;
            ray_forward = -forward;
            origin_to_hit_position = -origin_to_hit_position;
            origin_to_portal = -origin_to_portal;
        }

        f32 prior_distance = ray.hit.distance;
        wall_hit_group.portal_origin = other_portal_position - origin_to_portal;
        ray.update( wall_hit_group.portal_origin + origin_to_hit_position * (1.0001f), ray_direction, ray_forward);
        generateWallHit(wall_hit_group.portal, edges, columns);
        if (ray.hit.isValid()) {
            ray.hit.distance  = sqrt(ray.hit.distance) + prior_distance;
            ray.origin = wall_hit_group.portal_origin;
            ray.finalizeHit(edges.data, columns.data);
            wall_hit_group.portal.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray_direction, mid_point, columns.data, ray.hit);
        }
    }
};
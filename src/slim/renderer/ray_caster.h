#pragma once

#include "../serialization/texture.h"
#include "../scene/tilemap.h"
// #include "../viewport/viewport.h"
#include "../math/vec2.h"
// #include "../math/vec3.h"

#define MIN_DIM_FACTOR 0.0f
#define MAX_DIM_FACTOR 1
#define DIM_FACTOR_RANGE (MAX_DIM_FACTOR - MIN_DIM_FACTOR)

#define MAX_COLOR_VALUE 0xFF

// #define EPS 0.000001f

// struct Plane {
//     vec2 position, normal;
// };

f32 getU(vec2 v) {
    f32 u = v.y / v.x;
    if (u > 1.0f || u < -1.0f) u = -1.0f / u;
    return (u + 1.0f) * 0.5f;
}

struct RayHit {
    LocalEdge local_edge{};
    vec2i tile_coords = {};
    vec2 position = {};

    f32 distance = 0.0f;
    f32 perp_distance = 0.0f;
    f32 tile_fraction = 0.0f;

    // TileEdge* edge = nullptr;
    const Circle* column = nullptr;
    u8 texture_id = 0;
};

vec2 ccw90(vec2 v) { return vec2{-v.y, v.x}; }
vec2 cw90(vec2 v) { return vec2{v.y, -v.x}; }


struct Ray {
    vec2 origin;
    vec2 direction;
    vec2 forward;
    // vec2 primary_origin;
    // vec2 primary_direction;
    // vec2 primary_forward;

    f32 rise_over_run;
    f32 run_over_rise;
    f32 base_distance = 0.0f;

    bool is_vertical;
    bool is_horizontal;
    bool is_facing_up;
    bool is_facing_down;
    bool is_facing_left;
    bool is_facing_right;

    RayHit hit;

    void update(vec2 new_origin, vec2 new_direction, vec2 new_forward) {
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
        //     direction = norm2(new_direction);
        //     origin = new_origin;
        //     is_vertical = direction.x == 0;
        //     is_horizontal = direction.y == 0;
        //     is_facing_left = direction.x < 0;
        //     is_facing_right = direction.x > 0;
        //     is_facing_up = direction.y < 0;
        //     is_facing_down = direction.y > 0;

        //     if is_vertical {
        //         if is_facing_down {
        //             rise_over_run = inf;
        //         } else {
        //             rise_over_run = -inf;
        //         }
        //     } else {
        //         rise_over_run = direction.y / direction.x;
        //     }

        //     if is_horizontal {
        //         if is_facing_right {
        //             rise_over_run = inf;
        //         } else {
        //             rise_over_run = -inf;
        //         }
        //     } else {
        //         rise_over_run = direction.x / direction.y;
        //     }
    }

    // bool intersectsWithPlane(const Plane& plane, const RayHit& hit) {
    //     f32 RD_dot_N = direction.dot(plane.normal);                if (RD_dot_N > 0.0f || -RD_dot_N < EPS) return false;
    //     f32 RP_dot_N = plane.normal.dot(plane.position - origin);  if (RP_dot_N > 0.0f || -RP_dot_N < EPS) return false;
    //     f32 t = RP_dot_N / RD_dot_N;
    //     hit.position = origin + t*direction;
    //     return true;
    // }

    void intersectsWithCircle(const Circle& circle) {
        vec2 C = circle.position - origin;
        f32 t = C.dot(direction);
        if (t > 0.0f) {
            f32 dt = circle.radius * circle.radius - (direction * t - C).squaredLength();
            if (dt > 0.0f && t*t > dt) { // Inside the sphere
                t -= sqrt(dt);
                if (t < hit.distance) {
                    hit.distance = t;
                    hit.column = &circle;
                }
            }
        }
    }

    bool intersectsWithEdge(const LocalEdge& local_edge) {
        hit.local_edge = local_edge;

        if (local_edge.is & (FACING_LEFT | FACING_RIGHT)) {
            if (is_vertical ||
                (is_facing_right && (local_edge.is & ON_THE_LEFT)) ||
                (is_facing_left && (local_edge.is & ON_THE_RIGHT)))
                return false;

            hit.position = local_edge.to.x;
            hit.position.y *= rise_over_run;

            return inRange(local_edge.from.y, hit.position.y, local_edge.to.y);
        }

        // Edge is horizontal:
        if (is_horizontal ||
            (is_facing_up && (local_edge.is & BELOW)) ||
            (is_facing_down && (local_edge.is & ABOVE)))
            return false;

        hit.position = local_edge.to.y;
        hit.position.x *= run_over_rise;

        return inRange(local_edge.from.x, hit.position.x, local_edge.to.x);
    }
    //
    // void updateToHitPortalEdge() {
    //     const TileEdge& edge = *hit.edge;
    //     vec2 from, to, new_origin, new_direction, new_forward;
    //
    //     from.x = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->to->x : edge.portal_to->from->x);
    //     from.y = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->to->y : edge.portal_to->from->y);
    //     to.x   = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->from->x : edge.portal_to->to->x);
    //     to.y   = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->from->y : edge.portal_to->to->y);
    //
    //     new_origin = lerp(from, to, hit.tile_fraction);
    //
    //     if (edge.portal_ray_rotation == 180) {
    //         new_direction = -direction;
    //         new_forward = -forward;
    //     } else if (edge.portal_ray_rotation == 90) {
    //         new_direction = ccw90(direction);
    //         new_forward = ccw90(forward);
    //     } else if (edge.portal_ray_rotation == -90) {
    //         new_direction = cw90(direction);
    //         new_forward = cw90(forward);
    //     } else {
    //         new_direction = direction;
    //         new_forward = forward;
    //     }
    //     // new_origin.x += new_direction.x * EPS;
    //     // new_origin.y += new_direction.y * EPS;
    //
    //     update(new_origin, new_direction, new_forward);
    // }
    //

    bool cast(const TileMap &tm, TileEdge* skip_edge = nullptr, f32 prior_hit_distance = 0.0f, bool primary_ray = true) {
        RayHit closest_hit;
        closest_hit.distance = 10000000;

        LocalEdge *local_edge_ptr = nullptr;
        LocalEdge local_edge;
        iterSlice(tm.local_edges, local_edge_ptr, i) {
            local_edge = *local_edge_ptr;
            // if (edge != skip_edge && edge->is_facing_forward && intersectsWithEdge(*edge)) {
            if (intersectsWithEdge(local_edge)) {
                hit.distance = hit.position.squaredLength();
                if (hit.distance < closest_hit.distance) {
                    closest_hit = hit;
                    closest_hit.local_edge = local_edge;
                }
            }
        }

        hit = closest_hit;
        hit.distance  = sqrt(hit.distance) + prior_hit_distance;
        hit.column = nullptr;

        for (i32 column_id = 0; column_id < tm.column_count; column_id++)
            intersectsWithCircle(tm.columns[column_id]);

        vec2 local_hit_position = hit.position;

        if (hit.column != nullptr) {
            hit.local_edge = {};
            hit.position = origin + direction * hit.distance;
            hit.tile_coords.x = (i32)hit.position.x;
            hit.tile_coords.y = (i32)hit.position.y;
            hit.tile_fraction = getU(hit.position - hit.column->position);
            hit.tile_fraction *= hit.column->radius;
            hit.tile_fraction -= (f32)(i32)hit.tile_fraction;
            hit.texture_id = 0;
        } else {
            // if ray.hit.edge.is_vertical {
            //     ray.hit.tile_fraction = ray.hit.position.y - f32(ray.hit.edge.local.from.y);
            // } else {
            //     ray.hit.tile_fraction = ray.hit.position.x - f32(ray.hit.edge.local.from.x);
            // }
            // ray.hit.tile_fraction -= f32(i32(ray.hit.tile_fraction));
            hit.position += origin;

            hit.tile_coords.x = (i32)hit.position.x;
            hit.tile_coords.y = (i32)hit.position.y;

            float edge_fraction = 0.0f;
            if (hit.local_edge.is & (FACING_LEFT | FACING_RIGHT)) {
                edge_fraction = local_hit_position.y - hit.local_edge.from.y;
                if (hit.local_edge.is & FACING_RIGHT) hit.tile_coords.x -= 1;
            } else {
                edge_fraction = local_hit_position.x - hit.local_edge.from.x;
                if (hit.local_edge.is & FACING_DOWN) hit.tile_coords.y -= 1;
            }
            hit.tile_fraction = edge_fraction - (f32)(i32)edge_fraction;
            hit.texture_id = hit.local_edge.texture_id;
        }

        hit.perp_distance =
            // primary_ray ?
            forward.dot(local_hit_position);// :
            // primary_forward.dot(primary_direction * hit.distance);

        return true; //hit.edge == nullptr || hit.edge->portal_to == nullptr;
    }
};



// Ray* portal_rays[MAX_WIDTH];
// int portal_ray_indices[MAX_WIDTH];


// u32 castRays(TileMap& tm, vec2 position) {
//     u32 portal_ray_count = 0;
//     u32 next_portal_ray_count = 0;
//
//     // RayHit closest_hit;
//     Ray* ray = nullptr;
//     iterSlice(rays, ray, ray_index){
//         if (!ray->cast(tm)) {
//             portal_rays[portal_ray_count] = ray;
//             portal_ray_indices[portal_ray_count] = (i32)ray_index;
//             portal_ray_count += 1;
//         }
//     }
//
//     u32 original_portal_rays_count = portal_ray_count;
//
//     while (portal_ray_count != 0) {
//         next_portal_ray_count = 0;
//         for (u32 ray_index = 0; ray_index < portal_ray_count; ray_index++) {
//             ray = portal_rays[ray_index];
//             ray->updateToHitPortalEdge();
//             moveTileMap(tm, ray->origin);
//             if (ray->cast(tm, ray->hit.edge->portal_to, ray->hit.distance, false)) {
//                 portal_rays[next_portal_ray_count] = ray;
//                 next_portal_ray_count += 1;
//             }
//         }
//
//         swap(&portal_ray_count, &next_portal_ray_count);
//     }
//
//     vec2 pos;
//     Slice<VerticalHit>* vertical_hit_row = nullptr;
//     VerticalHit* vertical_hit = nullptr;
//     iterSlice(vertical_hits.cells, vertical_hit_row, y) {
//         if (y == 0) continue;
//         iterSlice((*vertical_hit_row), vertical_hit, x) {
//             pos = position + vertical_hit->direction;
//             vertical_hit->found =
//                 inRange(0.0f, pos.x, (f32)(tm.width - 1)) &&
//                 inRange(0.0f, pos.y, (f32)(tm.height-1));
//
//             if (vertical_hit->found) {
//                 vertical_hit->tile_coords.x = (i32)pos.x;
//                 vertical_hit->tile_coords.y = (i32)pos.y;
//                 vertical_hit->u = pos.x - (f32)vertical_hit->tile_coords.x;
//                 vertical_hit->v = pos.y - (f32)vertical_hit->tile_coords.y;
//
//                 // if tile_coords.x != last_tile_coords.x ||
//                 //    tile_coords.y != last_tile_coords.y {
//
//                 //     last_tile_texture_id = cells[tile_coords.y][tile_coords.x].texture_id;
//                 //     last_tile_coords = tile_coords;
//                 // }
//             }
//         }
//     }
//
//     return original_portal_rays_count;
// }

/*

INLINE_XPU void renderPixelBeauty(
    const Viewport &viewport,
    const RayCasterSettings &settings,
    const CameraRayProjection &projection,
    const TileMap &tile_map,
    Ray &ray,
    RayHit &hit,

    const vec3 &direction,
    const f32 half_max_distance,

    Color &color,
    f32 &depth
) {
    color = Black;
    depth = INFINITY;

    color.applyToneMapping();
}

INLINE_XPU void renderPixelDebugMode(
    const RayCasterSettings &settings,
    const CameraRayProjection &projection,
    TileMap &tile_map,
    Camera &camera,
    Ray &ray,
    RayHit &hit,

    const vec3 &direction,
    const f32 half_max_distance,

    Color &color,
    f32 &depth
) {
    color = Black;
    depth = INFINITY;

    // ray.reset(projection.camera_position, direction.normalized());

    // surface.geometry = scene_tracer.trace(ray, hit, scene);
    // if (surface.geometry) {
    //     // surface.prepareForShading(ray, hit, scene.materials, scene.textures);
    //     depth = projection.getDepthAt(hit.position);
    //     switch (settings.render_mode) {
    //         case RenderMode_UVs      : color = getColorByUV(hit.uv); break;
    //         case RenderMode_Depth    : color = getColorByDistance(hit.distance); break;
    //         case RenderMode_Normals  : color = directionToColor(hit.normal);  break;
    //         case RenderMode_NormalMap: color = sampleNormal(*surface.material, hit, scene.textures);  break;
    //         case RenderMode_MipLevel : color = scene.counts.textures ? settings.mip_level_colors[scene.textures[0].mipLevel(hit.uv_coverage)] : Grey;
    //         default: break;
    //     }
    // }
}

INLINE_XPU void renderPixel(
    const RayCasterSettings &settings,
    const CameraRayProjection &projection,
    TileMap &tile_map,
    Camera &camera,
    Ray &ray,
    RayHit &hit,

    const vec3 &direction,

    Color &color,
    f32 &depth
) {
    // if (settings.render_mode == RenderMode_Beauty)
    //     renderPixelBeauty(settings, projection, tile_map, camera, ray, hit, direction, half_max_distance, color, depth);
    // else
    //     renderPixelDebugMode(settings, projection, tile_map, camera, ray, hit, direction, half_max_distance, color, depth);
}
*/
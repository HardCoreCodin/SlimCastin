#pragma once

#include "../serialization/texture.h"
#include "../scene/tilemap.h"
#include "../viewport/viewport.h"
#include "../math/vec2.h"
#include "../math/vec3.h"

#define MIN_DIM_FACTOR 0.0f
#define MAX_DIM_FACTOR 1
#define DIM_FACTOR_RANGE (MAX_DIM_FACTOR - MIN_DIM_FACTOR)

#define MAX_COLOR_VALUE 0xFF

// #define EPS 0.000001f

struct Plane {
    vec2 position, normal;
};

f32 getU(vec2 v) {
    f32 u = v.y / v.x;
    if (u > 1.0f || u < -1.0f) u = -1.0f / u;
    return (u + 1.0f) * 0.5f;
}

struct RayHit {
    vec2i tile_coords = {};
    vec2 position = {};

    f32 distance = 0.0f;
    f32 perp_distance = 0.0f;
    f32 edge_fraction = 0.0f;
    f32 tile_fraction = 0.0f;

    TileEdge* edge = nullptr;
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

    bool intersectsWithEdge(TileEdge& edge) {
        hit.edge = &edge;

        if (edge.is_vertical) {
            if (is_vertical || (edge.local.is_left && is_facing_right) || (edge.local.is_right && is_facing_left))
                return false;

            hit.position = edge.local.to->x;
            hit.position.y *= rise_over_run;

            return inRange(edge.local.from->y, hit.position.y, edge.local.to->y);
        }

        // Edge is horizontal:
        if (is_horizontal || (edge.local.is_below && is_facing_up) || (edge.local.is_above && is_facing_down))
            return false;

        hit.position = edge.local.to->y;
        hit.position.x *= run_over_rise;

        return inRange(edge.local.from->x, hit.position.x, edge.local.to->x);
    }

    void updateToHitPortalEdge() {
        const TileEdge& edge = *hit.edge;
        vec2 from, to, new_origin, new_direction, new_forward;

        from.x = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->to->x : edge.portal_to->from->x);
        from.y = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->to->y : edge.portal_to->from->y);
        to.x   = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->from->x : edge.portal_to->to->x);
        to.y   = (f32)(edge.portal_edge_dir_flip ? edge.portal_to->from->y : edge.portal_to->to->y);

        new_origin = lerp(from, to, hit.tile_fraction);

        if (edge.portal_ray_rotation == 180) {
            new_direction = -direction;
            new_forward = -forward;
        } else if (edge.portal_ray_rotation == 90) {
            new_direction = ccw90(direction);
            new_forward = ccw90(forward);
        } else if (edge.portal_ray_rotation == -90) {
            new_direction = cw90(direction);
            new_forward = cw90(forward);
        } else {
            new_direction = direction;
            new_forward = forward;
        }
        // new_origin.x += new_direction.x * EPS;
        // new_origin.y += new_direction.y * EPS;

        update(new_origin, new_direction, new_forward);
    }


    bool cast(const TileMap &tm, TileEdge* skip_edge = nullptr, f32 prior_hit_distance = 0.0f, bool primary_ray = true) {
        RayHit closest_hit;
        closest_hit.distance = 10000000;

        TileEdge *edge = nullptr;
        iterSlice(tm.edges, edge, i) {
            if (edge != skip_edge && edge->is_facing_forward && intersectsWithEdge(*edge)) {
                hit.distance = hit.position.squaredLength();
                if (hit.distance < closest_hit.distance) {
                    closest_hit = hit;
                    closest_hit.edge = edge;
                }
            }
        }

        if (closest_hit.edge == nullptr) {
            return true;
        }

        hit = closest_hit;
        hit.distance  = sqrt(hit.distance) + prior_hit_distance;
        hit.column = nullptr;

        for (i32 column_id = 0; column_id < tm.column_count; column_id++)
            intersectsWithCircle(tm.columns[column_id]);

        if (hit.column != nullptr) {
            hit.edge = nullptr;
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

            if (hit.edge->is_vertical) {
                hit.edge_fraction = hit.position.y - (f32)hit.edge->from->y;
                if (hit.edge->is_facing_right) hit.tile_coords.x -= 1;
            } else {
                hit.edge_fraction = hit.position.x - (f32)hit.edge->from->x;
                if (hit.edge->is_facing_down) hit.tile_coords.y -= 1;
            }
            hit.tile_fraction = hit.edge_fraction - (f32)(i32)hit.edge_fraction;
            hit.texture_id = hit.edge->texture_id;
        }

        hit.perp_distance =
            // primary_ray ?
            forward.dot(hit.position - origin);// :
            // primary_forward.dot(primary_direction * hit.distance);

        return hit.edge == nullptr || hit.edge->portal_to == nullptr;
    }
};


enum FilterMode {
    FilterMode_None,
    FilterMode_BiLinear,
    FilterMode_TriLinear
};

#define RAY_CASTER_DEFAULT_SETTINGS_MAX_DEPTH 3
#define RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE RenderMode_Beauty
#define RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE FilterMode_BiLinear

struct RayCasterSettings {
    const Slice<Texture>& wall_textures;
    const Texture* floor_texture;
    const Texture* ceiling_texture;
    const u32 max_depth;
    const u32 mip_count;
    const u32 texture_width;
    const u32 texture_height;

    FilterMode filter_mode = FilterMode_BiLinear;
    RenderMode render_mode = RenderMode_Beauty;

    const ColorID mip_level_colors[9] = {
        BrightRed,
        BrightYellow,
        BrightGreen,
        BrightMagenta,
        BrightCyan,
        BrightBlue,
        BrightGrey,
        Grey,
        DarkGrey,
    };

    RayCasterSettings(
        const Slice<Texture>& wall_textures,
        const Texture* floor_texture,
        const Texture* ceiling_texture,
        const u32 max_depth = RAY_CASTER_DEFAULT_SETTINGS_MAX_DEPTH,
        RenderMode render_mode = RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE,
        FilterMode filter_mode = RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE) :
    wall_textures{wall_textures},
    floor_texture{floor_texture},
    ceiling_texture{ceiling_texture},
    max_depth{max_depth},
    mip_count{wall_textures[0].mip_count},
    texture_width{wall_textures[0].width},
    texture_height{wall_textures[0].height},
    render_mode{render_mode},
    filter_mode{filter_mode}
    {}
};

Ray all_rays[MAX_WIDTH];
Slice<Ray> rays;

Ray* portal_rays[MAX_WIDTH];
int portal_ray_indices[MAX_WIDTH];


struct VerticalHitLevel {
    f32 distance = 0.0f;
    f32 dim_factor = 1.0f;
    f32 mip_factor = 1.0f;
    u8 mip_level = 0;
};


struct VerticalHit {
    f32 dim_factor = 1.0f;
    vec2 direction = {};
    vec2i tile_coords = {};
    f32 u = 0.0f;
    f32 v = 0.0f;

    u8 floor_texture_id = 0;
    u8 ceiling_texture_id = 0;
    bool found = false;
};

VerticalHitLevel vertical_hit_levels[MAX_HEIGHT / 2];
VerticalHit vertical_hits_buffer[MAX_WIDTH * (MAX_HEIGHT / 2)];
Grid<VerticalHit> vertical_hits;

f32 horizontal_distance_factor = 20 * DIM_FACTOR_RANGE / MAX_TILE_MAP_VIEW_DISTANCE;



u32 castRays(TileMap& tm, vec2 position) {
    u32 portal_ray_count = 0;
    u32 next_portal_ray_count = 0;

    // RayHit closest_hit;
    Ray* ray = nullptr;
    iterSlice(rays, ray, ray_index){
        if (!ray->cast(tm)) {
            portal_rays[portal_ray_count] = ray;
            portal_ray_indices[portal_ray_count] = (i32)ray_index;
            portal_ray_count += 1;
        }
    }

    u32 original_portal_rays_count = portal_ray_count;

    while (portal_ray_count != 0) {
        next_portal_ray_count = 0;
        for (u32 ray_index = 0; ray_index < portal_ray_count; ray_index++) {
            ray = portal_rays[ray_index];
            ray->updateToHitPortalEdge();
            moveTileMap(tm, ray->origin);
            if (ray->cast(tm, ray->hit.edge->portal_to, ray->hit.distance, false)) {
                portal_rays[next_portal_ray_count] = ray;
                next_portal_ray_count += 1;
            }
        }

        swap(&portal_ray_count, &next_portal_ray_count);
    }

    vec2 pos;
    Slice<VerticalHit>* vertical_hit_row = nullptr;
    VerticalHit* vertical_hit = nullptr;
    iterSlice(vertical_hits.cells, vertical_hit_row, y) {
        if (y == 0) continue;
        iterSlice((*vertical_hit_row), vertical_hit, x) {
            pos = position + vertical_hit->direction;
            vertical_hit->found =
                inRange(0.0f, pos.x, (f32)(tm.width - 1)) &&
                inRange(0.0f, pos.y, (f32)(tm.height-1));

            if (vertical_hit->found) {
                vertical_hit->tile_coords.x = (i32)pos.x;
                vertical_hit->tile_coords.y = (i32)pos.y;
                vertical_hit->u = pos.x - (f32)vertical_hit->tile_coords.x;
                vertical_hit->v = pos.y - (f32)vertical_hit->tile_coords.y;

                // if tile_coords.x != last_tile_coords.x ||
                //    tile_coords.y != last_tile_coords.y {

                //     last_tile_texture_id = cells[tile_coords.y][tile_coords.x].texture_id;
                //     last_tile_coords = tile_coords;
                // }
            }
        }
    }

    return original_portal_rays_count;
}


void generateRays(const Viewport& viewport) {
    vec2 position = {viewport.camera->position.x, viewport.camera->position.z};
    vec2 forward_direction = vec2(viewport.camera->orientation.forward.x, viewport.camera->orientation.forward.z).normalized();
    vec2 right_direction = vec2(viewport.camera->orientation.right.x, viewport.camera->orientation.right.z).normalized();
    vec2 ray_direction = viewport.camera->focal_length * forward_direction;
    ray_direction *= viewport.dimensions.h_width;
    ray_direction += right_direction / 2.0f;

    Ray* ray = nullptr;
    iterSlice(rays, ray, x) {
        ray->update(position, ray_direction, forward_direction);
        // ray->primary_origin = ray->origin;
        // ray->primary_direction = ray->direction;
        // ray->primary_forward = forward_direction;
        // ray->is_vertical     = ray->direction.x == 0;
        // ray->is_horizontal   = ray->direction.y == 0;
        // ray->is_facing_left  = ray->direction.x < 0;
        // ray->is_facing_up    = ray->direction.y < 0;
        // ray->is_facing_right = ray->direction.x > 0;
        // ray->is_facing_down  = ray->direction.y > 0;
        ray->rise_over_run = ray->direction.y / ray->direction.x;
        ray->run_over_rise = 1 / ray->rise_over_run;
        ray->base_distance = ray_direction.length();

        Slice<VerticalHit>* vertical_hit_row = nullptr;
        iterSlice(vertical_hits.cells, vertical_hit_row, y) {
            const VerticalHitLevel& vertical_hit_level = vertical_hit_levels[y];
            VerticalHit& vertical_hit = (*vertical_hit_row)[x];
            vertical_hit.direction = ray->direction;
            vertical_hit.direction *= ray->base_distance * vertical_hit_level.distance * 0.5f;
            vertical_hit.dim_factor = 1.5f / std::max(1.0f, 0.25f + vertical_hit.direction.squaredLength());
        }

        ray_direction += right_direction;
    }
}



void onResize(const Viewport& viewport, u32 mip_count) {
    u16 half_height_int = viewport.dimensions.height / 2;

    setSliceToRangeOfStaticArray(rays, all_rays, 0, viewport.dimensions.width);
    initGrid<VerticalHit>(vertical_hits, viewport.dimensions.width, half_height_int, SliceFromStaticArray(VerticalHit, vertical_hits_buffer));

    f32 current_mip_level = (f32)mip_count;
    u8 current_mip_levelI = 0;

    VerticalHitLevel* vertical_hit_level = nullptr;

    for (u16 y = 0; y < half_height_int; y++) {
        vertical_hit_level = &vertical_hit_levels[half_height_int - y];

        vertical_hit_level->distance = viewport.camera->focal_length / (f32)(2 * y);
        vertical_hit_level->dim_factor = 1.1f / (1 + vertical_hit_level->distance * viewport.dimensions.h_width);// + MIN_DIM_FACTOR;

        current_mip_level *= 0.975f;
        current_mip_levelI = (u8)current_mip_level;

        vertical_hit_level->mip_level = current_mip_levelI;
        vertical_hit_level->mip_factor = current_mip_level - (f32)current_mip_levelI;
    }
    generateRays(viewport);
}


void onFocalLengthChanged(const Viewport& viewport) {
    generateRays(viewport);
}

void drawWalls(const Viewport &viewport, const RayCasterSettings& settings) {
    i32 top, bottom, pixel_offset, column_height; top = bottom = pixel_offset = column_height = 0;
    f32 texel_height, distance, u, v; texel_height = distance = u = v = 0.0f;

    f32 half_max_distance = viewport.dimensions.h_width * viewport.camera->focal_length * 0.5f;

    VerticalHit* vertical_hit;

    u32 floor_pixel_offset = viewport.dimensions.width_times_height - viewport.dimensions.width;
    i32 ceiling_pixel_offset = 0;

    u8 mip_level = 0;
    u8 last_mip = (u8)settings.mip_count - 1;
    f32 initial_mip = (f32)settings.mip_count * 0.9f;

    u8 other_mip_level = 1;

    f32 current_mip_factor = 0.0f;
    f32 next_mip_factor = 0.0f;

    Texture* wall_texture;

    // TextureMip* wall_bitmap;
    // TextureMip* floor_bitmap;
    // TextureMip* ceiling_bitmap;
    TextureMip* wall_samples;
    TextureMip* floor_samples;
    TextureMip* ceiling_samples;
    TextureMip* other_wall_samples;
    TextureMip* other_floor_samples;
    TextureMip* other_ceiling_samples;

    Pixel
        other_wall_pixel, wall_pixel,
        other_floor_pixel, floor_pixel,
        other_ceiling_pixel, ceiling_pixel,
        black = {};

    // u8 texture_id = 0;
    u8 current_mip_levelI = 0;
    f32 current_mip_levelf = 0.0f;
    f32 texture_height_ratio = 0.0f;

    VerticalHitLevel* vertical_hit_level;

    Slice<VerticalHit>* vertical_hit_row = nullptr;
    iterSlice(vertical_hits.cells, vertical_hit_row, y) {
        vertical_hit_level = &vertical_hit_levels[y];
        mip_level = std::min(vertical_hit_level->mip_level, last_mip);
        current_mip_factor = vertical_hit_level->mip_factor;

        // if (filter_mode == FilterMode::None) {
        //     floor_bitmap = &floor_texture->mips[mip_level];
        //     ceiling_bitmap = &ceiling_texture->mips[mip_level];
        // } else {
            floor_samples = &settings.floor_texture->mips[mip_level];
            ceiling_samples = &settings.ceiling_texture->mips[mip_level];

            if (settings.filter_mode == FilterMode_TriLinear) {
                other_mip_level = mip_level == 0 ? 0 : std::max(mip_level - 1, 0);
                next_mip_factor = 1 - current_mip_factor;

                other_floor_samples = &settings.floor_texture->mips[other_mip_level];
                other_ceiling_samples = &settings.ceiling_texture->mips[other_mip_level];
            }
        // }
        iterSlice((*vertical_hit_row), vertical_hit, x) {
            if (!vertical_hit->found) {
                viewport.canvas.pixels[floor_pixel_offset + i32(x)] = black;
                viewport.canvas.pixels[ceiling_pixel_offset + i32(x)] = black;
                continue;
            }
            // if filter_mode == FilterMode.None {
            //     sampleGrid(floor_bitmap, vertical_hit.u, vertical_hit.v, &floor_pixel);
            //     sampleGrid(ceiling_bitmap, vertical_hit.u, vertical_hit.v, &ceiling_pixel);
            // } else {
            floor_pixel = floor_samples->sample(vertical_hit->u, vertical_hit->v);
            ceiling_pixel = ceiling_samples->sample(vertical_hit->u, vertical_hit->v);

            if (settings.filter_mode == FilterMode_TriLinear) {
                other_floor_pixel = other_floor_samples->sample(vertical_hit->u, vertical_hit->v);
                other_ceiling_pixel = other_ceiling_samples->sample(vertical_hit->u, vertical_hit->v);

                floor_pixel = floor_pixel * current_mip_factor + other_floor_pixel * next_mip_factor;
                ceiling_pixel = ceiling_pixel * current_mip_factor + other_ceiling_pixel * next_mip_factor;
            }
            // }

            floor_pixel   *= vertical_hit->dim_factor;
            ceiling_pixel *= vertical_hit->dim_factor;
            viewport.canvas.pixels[floor_pixel_offset + i32(x)] = floor_pixel;
            viewport.canvas.pixels[ceiling_pixel_offset + i32(x)] = ceiling_pixel;
        }

        floor_pixel_offset   -= viewport.dimensions.width;
        ceiling_pixel_offset += viewport.dimensions.width;
    }

    f32 half_column_height, distance_squared, height_squared; half_column_height = distance_squared = height_squared = 0.0f;

    Ray* ray = nullptr;
    iterSlice(rays, ray, x) {
        distance = ray->hit.perp_distance;
        if (distance < 0) distance = -distance;

        distance_squared = distance * distance;

        half_column_height = half_max_distance / distance;
        column_height = (i32)(half_column_height + half_column_height);

    	top    = column_height < viewport.dimensions.height ? (viewport.dimensions.height - column_height) / 2 : 0;
        bottom = column_height < viewport.dimensions.height ? (viewport.dimensions.height + column_height) / 2 : viewport.dimensions.height;

        wall_texture = &settings.wall_textures.data[ray->hit.texture_id];

        texel_height = 1.0f / (f32)column_height;
        v = column_height > viewport.dimensions.height ? (f32)((column_height - (i32)viewport.dimensions.height) / 2) * texel_height : 0;
        u = ray->hit.tile_fraction;

        // if filter_mode == FilterMode.None do wall_bitmap = wall_texture.bitmaps[0]; else
        {
            texture_height_ratio = texel_height * 256 / 18;

            current_mip_levelf = initial_mip;
            while ((current_mip_levelf / (f32)(settings.mip_count)) > texture_height_ratio) current_mip_levelf *= 0.9f;

            current_mip_levelI = (u8)current_mip_levelf;
            current_mip_factor = current_mip_levelf - (f32)current_mip_levelI;

            mip_level = std::min(u8(current_mip_levelI), last_mip);
            wall_samples = &wall_texture->mips[mip_level];

            if (settings.filter_mode == FilterMode_TriLinear) {
                other_mip_level = mip_level == 0 ? 0 : std::max(mip_level - 1, 0);
                other_wall_samples = &wall_texture->mips[other_mip_level];
                next_mip_factor = 1 - current_mip_factor;
            }
        }

		pixel_offset = top * viewport.dimensions.width + (i32)x;
        for (i32 y = top; y < bottom; y++) {
            // if (filter_mode == FilterMode.None {
                // sampleGrid(wall_bitmap, u, v, &wall_pixel);
            // } else {
            wall_pixel = wall_samples->sample(u, v);

            if (settings.filter_mode == FilterMode_TriLinear) {
    	        other_wall_pixel = other_wall_samples->sample(u, v);
    	        wall_pixel = wall_pixel * current_mip_factor + other_wall_pixel * next_mip_factor;
            }
            // }
            height_squared = ((f32)y - viewport.dimensions.h_height) / half_column_height;
            height_squared *= height_squared;
            wall_pixel *= 1.5f / std::max(1.0f, 0.25f + distance_squared);
            viewport.canvas.pixels[pixel_offset] = wall_pixel;

            pixel_offset += viewport.dimensions.width;
            v += texel_height;
        }
    }
}


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
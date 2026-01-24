#pragma once

#include "../viewport/viewport.h"
#include "ray_caster.h"
#include "pixel_shader.h"

#ifdef __CUDACC__
#include "./renderer_GPU.h"
#else
#define USE_GPU_BY_DEFAULT false
void initDataOnGPU(const RayCasterSettings& settings) {}
void uploadWallHits(WallHit* wall_hits, u16 wall_hits_count)  {}
void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {}
#endif

WallHit wall_hits[MAX_WALL_HITS_COUNT];
GroundHit ground_hits[MAX_GROUND_HITS_COUNT];


namespace ray_cast_renderer {
    const RayCasterSettings* settings;

    vec2 position, forward, right;
    u16 half_screen_height, screen_height, screen_width;
    f32 texel_size;
    u8 last_mip;

    void generateFloorAndCeilingHits(f32 focal_length) {
        focal_length *= 0.5f;
        f32 screen_pixel_height = 1.0f / (f32)half_screen_height;

        f32 Y, Z, priorZ = 0.0f;
        ground_hits[0].mip = 0;
        ground_hits[0].z = focal_length;
        for (u16 y = 1; y < half_screen_height; y++) {
            Y = (f32)y * screen_pixel_height;
            Z = Y * focal_length / (1.0f - Y);

            ground_hits[y].mip = computeMip((Z - priorZ) * 0.5f, texel_size, last_mip);
            ground_hits[y].z = Z + focal_length;

            priorZ = Z;
        }
        uploadGroundHits(ground_hits, half_screen_height);
    }

    void generateWallHits(f32 focal_length, const TileMap& tile_map) {
        vec2 right_step = right / (f32)screen_width;
        vec2 ray_direction = focal_length * forward + right_step * (0.5f - 0.5f * (f32)screen_width);

        f32 column_height_factor = 0.5f * focal_length * (f32)screen_height;
        f32 pixel_coverage_factor = focal_length / (f32)screen_height;

        WallHit wall_hit;
        Ray ray;
        for (u16 x = 0; x < screen_width; x++, ray_direction += right_step) {
            ray.update(position, ray_direction, forward);
            ray.cast(tile_map);
            wall_hit.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray_direction, ray.hit);
            wall_hits[x] = wall_hit;
        }
        uploadWallHits(wall_hits, screen_width);
    }

    void onMove(const Camera& camera, TileMap& tile_map) {
        position = vec2(camera.position.x, camera.position.z);

        moveTileMap(tile_map, position);
    }

    void onMoveOrTurn(const Camera& camera, const TileMap& tile_map) {
        forward = vec2(camera.orientation.Z.x, camera.orientation.Z.z).normalized();
        right = vec2(camera.orientation.X.x, camera.orientation.X.z).normalized() * ((f32)screen_width / (f32)screen_height);

        generateWallHits(camera.focal_length, tile_map);
    }

    void onResize(u16 width, u16 height, const Camera& camera, const TileMap& tile_map) {
        screen_width = width;
        half_screen_height = height >> 1;
        screen_height = half_screen_height << 1;

        generateFloorAndCeilingHits(camera.focal_length);
        onMoveOrTurn(camera, tile_map);
    }

    void renderOnCPU(Canvas &canvas) {
        // canvas.clear(1.0f, 0.0f, 1.0f, 1.0f);
        const vec2 tile_map_end = vec2((f32)(settings->tile_map_width - 1), (f32)(settings->tile_map_height - 1));

        for (u16 x = 0; x < screen_width; x++) {
            WallHit wall_hit = wall_hits[x];
            for (u16 y = 0; y < half_screen_height; y++) {
                GroundHit ground_hit = ground_hits[y];

                renderPixel(x, y, position, tile_map_end,
                    canvas.pixels, screen_width, screen_height,
                    wall_hit, ground_hit,
                    settings->textures,
                    settings->ceiling_texture_id,
                    settings->floor_texture_id);
            }
        }
    }

    void init(const RayCasterSettings* render_settings, const Dimensions& dim, const Camera& camera, TileMap& tile_map)
    {
        settings = render_settings;

        initDataOnGPU(*settings);

        Texture &texture{settings->textures[0]};
        texel_size = 1.0f / (f32)texture.width;
        last_mip = (u8)(texture.mip_count - 1);

        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(Canvas &canvas, bool use_GPU = false) {
        #ifdef __CUDACC__
        if (use_GPU) renderOnGPU(canvas, position);
        else         renderOnCPU(canvas);
        #else
        renderOnCPU(canvas);
        #endif
    }
};
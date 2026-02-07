#pragma once

#include "../scene/camera.h"
#include "../scene/tilemap.h"
#include "ray_caster.h"
#include "pixel_shader.h"

#ifdef __CUDACC__
#include "./renderer_GPU.h"
#else
#define USE_GPU_BY_DEFAULT false
void initDataOnGPU(const RayCasterSettings& settings) {}
void uploadLocalEdges(const Slice<LocalEdge>& local_edges) {}
void uploadColumns(const Slice<Circle>& columns) {}
void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {}
void generateWallHitsOnGPU(const RayCaster &ray_caster) {}
void uploadWallHits(WallHit* wall_hits, u16 wall_hits_count)  {}
void uploadSettings(const RayCasterSettings* settings)  {}
#endif

WallHit wall_hits[MAX_WALL_HITS_COUNT];
GroundHit ground_hits[MAX_GROUND_HITS_COUNT];


namespace ray_cast_renderer {
    const RayCasterSettings* settings;
    RayCaster ray_caster;
    f32 prior_up_aim;
    u16 prior_screen_height;
    bool useGPU = false;

    void toggleUseOfGPU() {
#ifdef __CUDACC__
        if (useGPU) {
            downloadWallHits(wall_hits, ray_caster.screen_width);
            useGPU = false;
        } else {
            uploadWallHits(wall_hits, ray_caster.screen_width);
            uploadSettings(settings);
            useGPU = true;
        }
#endif
    }

    void generateFloorAndCeilingHits() {
        f32 Y = 1.0f + ray_caster.up_aim;

        f32 screen_pixel_height = 2.0f / (f32)ray_caster.screen_height;

        f32 Z, priorZ = 1.0f / (Y + screen_pixel_height);
        i32 y = 0;

        for (; y < ray_caster.mid_point; y++, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z;
            ground_hits[y].dim_factor = getDimFactor(Z);
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        Y = 1.0f - ray_caster.up_aim;
        priorZ = 1.0f / (Y + screen_pixel_height);
        y = ray_caster.screen_height - 1;

        for (; y > ray_caster.mid_point; y--, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z;
            ground_hits[y].dim_factor = getDimFactor(Z);
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        uploadGroundHits(ground_hits, ray_caster.screen_height);
    }

    void generateWallHits(const TileMap& tile_map) {
        if (useGPU) {
            generateWallHitsOnGPU(ray_caster);
        } else {
            WallHit wall_hit;
            RayHit closest_hit;
            Ray ray;
            vec2 ray_direction = ray_caster.first_ray_direction;
            for (u16 x = 0; x < ray_caster.screen_width; x++, ray_direction += ray_caster.right_step) {
                ray_caster.generateWallHit(wall_hit, ray_direction, ray, closest_hit, tile_map.local_edges, tile_map.columns);
                wall_hits[x] = wall_hit;
            }
        }
    }

    void onMove(const Camera& camera, TileMap& tile_map) {
        ray_caster.position = vec2(camera.position.x, camera.position.z);

        moveTileMap(tile_map, ray_caster.position);

        uploadLocalEdges(tile_map.local_edges);
    }

    void onScreenChanged(const Camera& camera, const TileMap& tile_map) {
        vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z);
        vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z);
        ray_caster.onScreenChanged(camera.focal_length, forward, right, camera.orientation.Z.y);
        generateWallHits(tile_map);
        if (prior_screen_height != ray_caster.screen_height ||
            prior_up_aim != ray_caster.up_aim)
            generateFloorAndCeilingHits();

        prior_up_aim = ray_caster.up_aim;
    }

    void onResize(u16 width, u16 height, const Camera& camera, const TileMap& tile_map) {
        ray_caster.screen_height = (height >> 1) << 1;
        ray_caster.screen_width = width;
        onScreenChanged(camera, tile_map);

        prior_screen_height = ray_caster.screen_height;
    }

    void onSettingsChanged() {
        if (useGPU)
            uploadSettings(settings);
    }

    void renderOnCPU(u32* window_content) {
        Color pixel;

        u32 offset = 0;
        for (u16 y = 0; y < ray_caster.screen_height; y++) {
            const bool is_ceiling = y < ray_caster.mid_point;
            GroundHit ground_hit = ground_hits[y];
            for (u16 x = 0; x < ray_caster.screen_width; x++, offset++) {
                WallHit wall_hit = wall_hits[x];
                if (y < wall_hit.top ||
                    y > wall_hit.bot)
                    renderGroundPixel(ground_hit, ray_caster.position, wall_hit.ray_direction, is_ceiling, *settings, pixel);
                else
                    renderWallPixel(wall_hit, y, *settings, pixel);
                window_content[offset] = pixel.asContent();
            }
        }
    }

    void init(const RayCasterSettings* render_settings, const Dimensions& dim, const Camera& camera, TileMap& tile_map)
    {
        settings = render_settings;

        Texture &texture{settings->textures[0]};
        ray_caster.texel_size = 1.0f / (f32)texture.width;
        ray_caster.last_mip = (u8)(texture.mip_count - 1);

        initDataOnGPU(*settings);

        // uploadColumns(const Slice<Circle>& columns);

        prior_screen_height = 0;
        prior_up_aim = 0.0f;

        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(u32* window_content) {
        #ifdef __CUDACC__
        if (useGPU) renderOnGPU(ray_caster, window_content);
        else        renderOnCPU(window_content);
        #else
        renderOnCPU(window_content);
        #endif
    }
};
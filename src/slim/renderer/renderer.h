#pragma once

#include "../draw/canvas.h"
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
#endif

WallHit wall_hits[MAX_WALL_HITS_COUNT];
GroundHit ground_hits[MAX_GROUND_HITS_COUNT];


namespace ray_cast_renderer {
    const RayCasterSettings* settings;
    RayCaster ray_caster;
    u16 half_screen_height;
    bool useGPU = false;

    void toggleUseOfGPU() {
#ifdef __CUDACC__
        if (useGPU) {
            downloadWallHits(wall_hits, ray_caster.screen_width);
            useGPU = false;
        } else {
            uploadWallHits(wall_hits, ray_caster.screen_width);
            useGPU = true;
        }
#endif
    }

    void generateFloorAndCeilingHits() {
        f32 screen_pixel_height = 1.0f / ((f32)half_screen_height);
        f32 Y, Z, priorZ = 0.0f;
        ground_hits[0].mip = 0;
        ground_hits[0].z = 1.0f;
        for (u16 y = 1; y < half_screen_height; y++) {
            Y = (f32)y * screen_pixel_height;
            Z = Y / (1.0f - Y);
            ground_hits[y].z = Z + 1.0f;
            ground_hits[y].mip = computeMip((Z - priorZ) * 0.5f, ray_caster.texel_size, ray_caster.last_mip);

            priorZ = Z;
        }
        uploadGroundHits(ground_hits, half_screen_height);
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
            // uploadWallHits(wall_hits, ray_caster.screen_width);
        }
    }

    void onMove(const Camera& camera, TileMap& tile_map) {
        ray_caster.position = vec2(camera.position.x, camera.position.z);

        moveTileMap(tile_map, ray_caster.position);

        uploadLocalEdges(tile_map.local_edges);
    }

    void onCameraChange(const Camera& camera, const TileMap& tile_map) {
        vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z);
        vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z);
        ray_caster.onCameraChanged(camera.focal_length, forward, right);

        generateWallHits(tile_map);
    }

    void onResize(u16 width, u16 height, const Camera& camera, const TileMap& tile_map) {
        half_screen_height = height >> 1;
        if (height != ray_caster.screen_height)
            generateFloorAndCeilingHits();
        ray_caster.screen_height = half_screen_height << 1;
        ray_caster.screen_width = width;
        onCameraChange(camera, tile_map);
    }

    void renderOnCPU(Canvas &canvas) {
        const vec2 tile_map_end = vec2((f32)(settings->tile_map_width - 1), (f32)(settings->tile_map_height - 1));

        for (u16 x = 0; x < ray_caster.screen_width; x++) {
            WallHit wall_hit = wall_hits[x];
            for (u16 y = 0; y < half_screen_height; y++) {
                GroundHit ground_hit = ground_hits[y];

                renderPixel(x, y, ray_caster.position, tile_map_end,
                    canvas.pixels, ray_caster.screen_width, ray_caster.screen_height,
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

        Texture &texture{settings->textures[0]};
        ray_caster.texel_size = 1.0f / (f32)texture.width;
        ray_caster.last_mip = (u8)(texture.mip_count - 1);

        initDataOnGPU(*settings);

        // uploadColumns(const Slice<Circle>& columns);

        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(Canvas &canvas) {
        #ifdef __CUDACC__
        if (useGPU) renderOnGPU(canvas, ray_caster.position);
        else        renderOnCPU(canvas);
        #else
        renderOnCPU(canvas);
        #endif
    }
};
#pragma once

#include "../viewport/viewport.h"
#include "ray_caster.h"
#include "pixel_shader.h"

#ifdef __CUDACC__
#include "./renderer_GPU.h"
#else
#define USE_GPU_BY_DEFAULT false
void initDataOnGPU(const TileMap &tile_map) {}
void uploadTileMap(const TileMap &tile_map) {}
#endif

struct RayCastingRenderer {
    RayCasterSettings& settings;
    const Viewport& viewport;
    TileMap &tile_map;
    // const Camera &camera;
    // CameraRayProjection &projection;
    // Ray ray;
    // RayHit hit;
    // Color color;
    // f32 depth;

    explicit RayCastingRenderer(
        RayCasterSettings& settings,
        const Viewport& viewport,
        TileMap &tile_map) :
        settings{settings},
        viewport{viewport},
        tile_map{tile_map}
    {

        // initDataOnGPU(tile_map);
    }

    void render(bool update_scene = true, bool use_GPU = false) {
        // const Camera &camera = *viewport.camera;
        // const Canvas &canvas = viewport.canvas;

        // ray.origin = camera.position;

        //         if (update_scene) {
        //             //
        //             if (use_GPU) {
        //                 uploadTileMap(tileMap);
        //             }
        //         }
        // #ifdef __CUDACC__
        //         if (use_GPU) renderOnGPU(canvas, projection, settings);
        //         else         renderOnCPU(canvas);
        // #else
        renderOnCPU();
        // #endif
    }




    WallHit wall_hits[1024*5];
    GroundHit ground_hits[1024*2];


    void generateFloorAndCeilingHits(const u16 screen_height, f32 focal_length, const f32 texel_size, const u8 last_mip) {
        focal_length *= 0.5f;
        u16 half_screen_height = screen_height >> 1;
        f32 screen_pixel_height = 1.0f / (f32)half_screen_height;

        f32 Y, Z, priorZ = 0.0f;
        ground_hits[0].mip = 0;
        ground_hits[0].z = focal_length;
        for (u16 y = 1; y < half_screen_height; y++) {
            Y = (f32)y * screen_pixel_height;
            Z = Y * focal_length / (1.0f - Y);

            ground_hits[y].mip = computeMip(Z - priorZ, texel_size, last_mip);
            ground_hits[y].z = Z + focal_length;

            priorZ = Z;
        }
    }

    void generateWallHits(u16 screen_width, u16 screen_height, f32 focal_length, u8 last_mip, f32 texel_size, vec2 position, vec2 forward, vec2 right) {
        vec2 right_step = right / (f32)screen_width;
        vec2 ray_direction = focal_length * forward + right_step * (0.5f - 0.5f * (f32)screen_width);

        f32 column_height_factor = 0.5f * focal_length * (f32)screen_height;
        f32 pixel_coverage_factor = focal_length / (f32)screen_height;

        WallHit wall_hit;
        Ray ray;
        for (u16 x = 0; x < screen_width; x++, ray_direction += right_step) {
            ray.update(position, ray_direction, forward);
            ray.cast(tile_map);
            wall_hit.update(screen_height, texel_size, pixel_coverage_factor, column_height_factor, last_mip, ray.direction, ray.hit);
            wall_hits[x] = wall_hit;
        }
    }

    void draw(u16 screen_width, u16 screen_height, vec2 position) {
        TextureMip* wall_texture;

        u32 last_line = screen_width * (screen_height - 1);
        f32 u, v, dim_factor;
        vec2 pos;
        GroundHit ground_hit;
        WallHit wall_hit;
        for (u16 x = 0; x < screen_width; x++) {
            wall_hit = wall_hits[x];
            wall_texture = &settings.wall_textures.data[wall_hit.texture_id].mips[wall_hit.mip];

            for (u16 y = 0; y <= wall_hit.top; y++) {
                ground_hit = ground_hits[y];

                pos = position + wall_hit.ray_direction * ground_hit.z;
                if (!(inRange(0.0f, pos.x, (f32)(tile_map.width - 1)) &&
                      inRange(0.0f, pos.y, (f32)(tile_map.height - 1))))
                    continue;

                u = pos.x - (f32)(i32)pos.x;
                v = pos.y - (f32)(i32)pos.y;

                dim_factor = 0.25f + ground_hit.z * ground_hit.z;
                dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
                dim_factor = 1.5f / dim_factor;
                viewport.canvas.pixels[            screen_width * y + x] = settings.ceiling_texture->mips[ground_hit.mip].sample(u, v) * dim_factor;
                viewport.canvas.pixels[last_line - screen_width * y + x] = settings.floor_texture->mips[ground_hit.mip].sample(u, v) * dim_factor;
            }

            u = wall_hit.u;
            for (u16 y = 0; y < wall_hit.height; y++) {
                dim_factor = 0.25f + wall_hit.z * wall_hit.z;
                dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
                dim_factor = 1.5f / dim_factor;
                v = wall_hit.v + (f32)y * wall_hit.texel_step;
                u32 offset = (u32)screen_width * (u32)(wall_hit.top + y) + (u32)x;
                viewport.canvas.pixels[offset] = wall_texture->sample(u, v) * dim_factor;
            }
        }
    }

    void renderOnCPU() {
        vec2 position = {viewport.camera->position.x, viewport.camera->position.z};
        if (viewport.navigation.moved) moveTileMap(tile_map, position);


        const u16 screen_width = viewport.dimensions.width;
        const u16 screen_height = viewport.dimensions.height;
        const f32 focal_length = viewport.camera->focal_length;
        const u8 last_mip = settings.mip_count - 1;
        const f32 texel_size = 1.0f / (f32)settings.ceiling_texture->width;
        const vec2 forward = vec2(viewport.camera->orientation.forward.x, viewport.camera->orientation.forward.z).normalized();
        const vec2 right = vec2(viewport.camera->orientation.right.x, viewport.camera->orientation.right.z).normalized();

        generateFloorAndCeilingHits(screen_height, focal_length, texel_size, last_mip);
        generateWallHits(screen_width, screen_height, focal_length, last_mip, texel_size, position, forward, right);

        viewport.canvas.clear(1.0f, 0.0f, 1.0f, 1.0f);
        // draw(screen_width, screen_height, position);
// return;
        for (u16 x = 0; x < screen_width; x++)
            for (u16 y = 0; y < screen_height; y++)
                renderPixel(x, y, position,
                    viewport.canvas.pixels, screen_width, screen_height, tile_map.width, tile_map.height,
                    wall_hits, ground_hits, settings.wall_textures.data, settings.floor_texture, settings.ceiling_texture);
    }
};
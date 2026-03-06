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
    RayCasterSettings* settings;
    RayCaster ray_caster;
    f32 prior_up_aim;
    u16 prior_screen_height;
    bool useGPU = false;
    bool adding_column = false;
    bool adding_tiles = false;
    bool removing_tiles = false;

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
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        Y = 1.0f - ray_caster.up_aim;
        priorZ = 1.0f / (Y + screen_pixel_height);
        y = ray_caster.screen_height - 1;

        for (; y > ray_caster.mid_point; y--, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z;
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        if (useGPU) uploadGroundHits(ground_hits, ray_caster.screen_height);
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

        if (useGPU) uploadLocalEdges(tile_map.local_edges);
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

    void onStopEditing() {
        settings->hovered_pos_x = 0.0f;
        settings->hovered_pos_y = 0.0f;
        adding_column = false;
        adding_tiles = false;
        removing_tiles = false;
        onSettingsChanged();
    }

    void onEditHover(TileMap& tile_map, vec2i mouse_pos, bool create_new_column = false) {
        if (settings->flags & (EDITING_WALLS | EDITING_COLUMNS) == 0 ||
            mouse_pos.x < 0 ||
            mouse_pos.y < 0 ||
            mouse_pos.x >= ray_caster.screen_width ||
            mouse_pos.y >= ray_caster.screen_height) {
            onSettingsChanged();
            return;
        }

        const vec2 position = ray_caster.position + wall_hits[mouse_pos.x].ray_direction * ground_hits[mouse_pos.y].z;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(settings->tile_map_width - 1),
            (f32)(settings->tile_map_height - 1)
        };
        if (!inRange(start, position, end)) {
            onSettingsChanged();
            return;
        }

        if (create_new_column) {
            Circle& column{tile_map.columns[tile_map.columns.size++]};
            column.position = position;
            column.radius = 0.1f;
            if (useGPU) uploadColumns(tile_map.columns);
            generateWallHits(tile_map);
        } else if (adding_column) {
            Circle& column{tile_map.columns[tile_map.columns.size - 1]};
            column.radius = fmaxf(0.1f, (position - column.position).length());
            if (useGPU) uploadColumns(tile_map.columns);
            generateWallHits(tile_map);
        } else if ((i32)settings->hovered_pos_x != (i32)position.x ||
                   (i32)settings->hovered_pos_y != (i32)position.y) {

            Tile& tile{tile_map.cells[(i32)position.y][(i32)position.x]};
            if (adding_tiles) {
                tile.left.texture_id = tile.right.texture_id = tile.bottom.texture_id = tile.top.texture_id = 12;
                tile.is_full = true;
            } else if (removing_tiles)
                tile.is_full = false;

            if (adding_tiles || removing_tiles) {
                generateTileMapEdges(tile_map);
                moveTileMap(tile_map, ray_caster.position);
                if (useGPU) uploadLocalEdges(tile_map.local_edges);
                generateWallHits(tile_map);
            }
        }

        settings->hovered_pos_x = position.x;
        settings->hovered_pos_y = position.y;
        if (useGPU) uploadSettings(settings);
    }

    void onEditLeftMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (settings->flags & EDITING_WALLS) {
            onStopEditing();
            adding_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if (settings->flags & EDITING_COLUMNS && tile_map.columns.size < MAX_COLUMN_COUNT) {
            adding_column = true;
            onEditHover(tile_map, mouse_pos, true);
        }
    }

    void onEditRightMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (settings->flags & EDITING_WALLS) {
            onStopEditing();
            removing_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if (settings->flags & EDITING_COLUMNS &&
                   tile_map.columns.size &&
                   wall_hits[mouse_pos.x].column_id != INVALID_COLUMN_ID) {
            tile_map.columns[wall_hits[mouse_pos.x].column_id] = tile_map.columns[--tile_map.columns.size];
            generateWallHits(tile_map);
        }
    }

    void renderOnCPU(u32* window_content) {
        u32 offset = 0;
        for (u16 y = 0; y < ray_caster.screen_height; y++) {
            GroundHit ground_hit = ground_hits[y];
            for (u16 x = 0; x < ray_caster.screen_width; x++, offset++) {
                window_content[offset] = PixelShader{*settings}.shade(
                    ground_hit,
                    wall_hits[x],
                    ray_caster.position,
                    y,
                    ray_caster.mid_point).asContent();
            }
        }
    }

    void init(RayCasterSettings* render_settings, const Dimensions& dim, const Camera& camera, TileMap& tile_map)
    {
        settings = render_settings;

        Texture &texture{settings->textures[0]};
        ray_caster.texel_size = 1.0f / (f32)texture.width;
        ray_caster.last_mip = (u8)(texture.mip_count - 1);

        initDataOnGPU(*settings);

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
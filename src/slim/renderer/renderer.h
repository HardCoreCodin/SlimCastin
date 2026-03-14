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
void uploadVisibleEdgeIds(const Slice<u16>& visible_edge_ids) {}
void uploadEdges(const Slice<TileEdge>& edges) {}
void uploadColumns(const Slice<Circle>& columns) {}
void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {}
void generateWallHitsOnGPU(const RayCaster &ray_caster) {}
void uploadWallHits(WallHit* wall_hits, u16 wall_hits_count)  {}
void downloadWallHits(WallHit* wall_hits, u16 wall_hits_count)  {}
#endif

WallHit wall_hits[MAX_WALL_HITS_COUNT];
GroundHit ground_hits[MAX_GROUND_HITS_COUNT];

#define INVALID_PROJECTILE_INDEX ((u8)(-1))

namespace ray_cast_renderer {
    RayCasterSettings* settings;
    RayCaster ray_caster;
    f32 prior_up_aim;
    u16 prior_screen_height;
    bool useGPU = false;
    bool adding_column = false;
    bool adding_tiles = false;
    bool removing_tiles = false;

    RenderState render_state;

    SpinningProjectile projectiles[MAX_POINT_LIGHTS];
    u8 projectile_count = 0;

    Color torch_light_color{1.0f, 0.6f, 0.35f};
    f32 torch_light_intensity = 4.0f;

    struct PortalState {
        Color color;
        f32 spawned_time;
        volatile u16 projectile_index;

        f32 computeRadius(const f32 time) {
            const f32 elapsed_time = time - spawned_time;
            return elapsed_time <= PORTAL_GROW_TIME ?
                (INITIAL_PORTAL_RADIUS + 0.04f + (FINAL_PORTAL_RADIUS - INITIAL_PORTAL_RADIUS) * smoothStep(0.0f, 1.0f, elapsed_time / PORTAL_GROW_TIME)) :
                (FINAL_PORTAL_RADIUS + cos((elapsed_time - PORTAL_GROW_TIME) * 2.0f) * 0.04f);
        }
    };

    PortalState portal_from{Cyan, 0.0f, INVALID_PROJECTILE_INDEX};
    PortalState portal_to{Magenta, 0.0f, INVALID_PROJECTILE_INDEX};

    void toggleUseOfGPU(const TileMap& tile_map) {
#ifdef __CUDACC__
        if (useGPU) {
            downloadWallHits(wall_hits, ray_caster.screen_width);
            useGPU = false;
        } else {
            uploadWallHits(wall_hits, ray_caster.screen_width);
            uploadColumns(tile_map.columns);
            uploadEdges(tile_map.edges);
            uploadVisibleEdgeIds(tile_map.visible_edge_ids);
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
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        Y = 1.0f - ray_caster.up_aim;
        priorZ = 1.0f / (Y + screen_pixel_height);
        y = ray_caster.screen_height - 1;

        for (; y > ray_caster.mid_point; y--, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        uploadGroundHits(ground_hits, ray_caster.screen_height);
    }

    void generateWallHits(const TileMap& tile_map) {
        if (useGPU) {
            generateWallHitsOnGPU(ray_caster);
            downloadWallHits(wall_hits, ray_caster.screen_width);
        } else {
            WallHit wall_hit;
            vec2 ray_direction;
            Ray ray;
            RayHit closest_hit;
            ray_direction = ray_caster.first_ray_direction;
            for (u16 x = 0; x < ray_caster.screen_width; x++, ray_direction += ray_caster.right_step) {
                ray_caster.generateWallHit(wall_hit, ray_direction, ray, closest_hit, tile_map.visible_edge_ids, tile_map.edges, tile_map.columns);
                wall_hits[x] = wall_hit;
            }
        }
    }

    void addLightProjectile(const f32 time, const Color color) {
        SpinningProjectile& projectile{projectiles[projectile_count++]};
        PointLight& point_light{render_state.lights[render_state.light_count++]};

        projectile.init(ray_caster.position, ray_caster.forward, ray_caster.up_aim, settings->projectile_radius, time);

        point_light.position = projectile.position;
        point_light.color = color;
        point_light.intensity = torch_light_intensity * 0.25f;
    }

    void fireFlare(const f32 time) {
        if (render_state.light_count < (MAX_POINT_LIGHTS - 2))
            addLightProjectile(time, torch_light_color);
    }

    void launchPortalFrom(const f32 time) {
        portal_from.projectile_index = projectile_count;
        addLightProjectile(time, portal_from.color);
    }

    void launchPortalTo(const f32 time) {
        portal_to.projectile_index = projectile_count;
        addLightProjectile(time, portal_to.color);
    }

    void update(const f32 time, const f32 delta_time, const TileMap& tile_map) {
        if (render_state.portal_from.edge_id != INVALID_EDGE_ID)
            render_state.portal_from.radius = portal_from.computeRadius(time);

        if (render_state.portal_to.edge_id != INVALID_EDGE_ID)
            render_state.portal_to.radius = portal_to.computeRadius(time);

        if (projectile_count == 0)
            return;

        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(settings->tile_map_width - 1),
            (f32)(settings->tile_map_height - 1)
        };
        for (u16 i = 0; i < projectile_count; i++) {
            SpinningProjectile& projectile{projectiles[i]};
            const f32 elapsed_time = time - projectile.spawned_time;
            vec3 projectile_position = projectile.position;
            projectile.updatePosition(delta_time * settings->projectile_speed);

            bool above_or_below = projectile.position.y >= 1.0f ||
                                  projectile.position.y <= -1.0f;
            bool remove = above_or_below ||
                          !inRange(start, {projectile.position.x, projectile.position.z}, end) ||
                          tile_map.cells[(i32)projectile.position.z][(i32)projectile.position.x].is_full;
            if (remove && !above_or_below && (i == portal_from.projectile_index || i == portal_to.projectile_index)) {
                Ray ray;
                TileEdge edge;

                vec3 ray_direction_3d = projectile.position - projectile_position;
                vec2 ray_direction_2d = vec2{ray_direction_3d.x, ray_direction_3d.z};
                const f32 distance_2d = ray_direction_2d.length();
                ray.update(vec2{projectile_position.x, projectile_position.z}, ray_direction_2d / distance_2d);
                f32 hit_distance = 1000000.0f;
                u16 closest_hit_edge_id = INVALID_EDGE_ID;
                for (u16 edge_id = 0; edge_id < (u16)tile_map.edges.size; edge_id++) {
                    edge = tile_map.edges.data[edge_id];
                    if (edge.isVisible(ray.origin) && ray.intersectsWithEdge(edge)) {
                        ray.hit.distance = (ray.hit.position - ray.origin).squaredLength();
                        if (ray.hit.distance < hit_distance) {
                            hit_distance = ray.hit.distance;
                            closest_hit_edge_id = edge_id;
                        }
                    }
                }

                projectile_position += ray_direction_3d * (sqrt(hit_distance) / distance_2d);
                if (abs(projectile_position.y) < (FINAL_PORTAL_RADIUS * 0.6f)) {
                    const bool is_from = portal_from.projectile_index == i;

                    Portal& portal{      is_from ? render_state.portal_from : render_state.portal_to};
                    Portal& other_portal{is_from ? render_state.portal_to : render_state.portal_from};

                    if (other_portal.edge_id == INVALID_EDGE_ID ||
                        (other_portal.position - projectile_position).length() > (2 * FINAL_PORTAL_RADIUS)) {
                        PortalState& portal_state{is_from ? portal_from : portal_to};
                        portal_state.spawned_time = time;
                        portal_state.projectile_index = INVALID_PROJECTILE_INDEX;

                        portal.position = projectile_position;
                        portal.edge_id = closest_hit_edge_id;
                        portal.edge_is = tile_map.edges[closest_hit_edge_id].is;
                        portal.radius = INITIAL_PORTAL_RADIUS;
                        portal.color = portal_state.color;
                    }
                }
            }
            if (!remove) {
                for (u8 c = 0; c < tile_map.columns.size; c++) {
                    const Circle& column{tile_map.columns.data[c]};
                    const f32 distance_squared = (column.position - vec2{projectile.position.x, projectile.position.z}).squaredLength();
                    if (distance_squared < (column.radius * column.radius)) {
                        remove = true;
                        break;
                    }
                }
            }
            if (remove) {
                projectile_count--;
                render_state.light_count--;
                if (projectile_count == 0) {
                    if (portal_from.projectile_index == 0)
                        portal_from.projectile_index = INVALID_EDGE_ID;
                    if (portal_to.projectile_index == 0)
                        portal_to.projectile_index = INVALID_EDGE_ID;

                    return;
                }

                if (portal_from.projectile_index == projectile_count)
                    portal_from.projectile_index = i;
                if (portal_to.projectile_index == projectile_count)
                    portal_to.projectile_index = i;

                projectiles[i] = projectiles[projectile_count];
                render_state.lights[i] = render_state.lights[render_state.light_count];
                i--;
                continue;
            }

            PointLight& point_light{render_state.lights[i+1]};
            point_light.position = projectile.position;
            if (i == portal_from.projectile_index)
                point_light.flicker(portal_from.color, torch_light_intensity * 0.25f, elapsed_time);
            else if (i == portal_to.projectile_index)
                point_light.flicker(portal_to.color, torch_light_intensity * 0.25f, elapsed_time);
            else
                point_light.flicker(torch_light_color, torch_light_intensity * 0.25f, elapsed_time);
        }
    }

    void onMove(Camera& camera, TileMap& tile_map) {
        vec2 position = vec2(camera.position.x, camera.position.z);
        vec2 movement = position - ray_caster.position;

        if (movement.x > 0.0f) {
            i32 next_pos = (i32)(position.x + settings->body_radius);
            const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
            if (next_tile.is_full) position.x = (f32)next_pos - settings->body_radius;
        } else if (movement.x < 0.0f) {
            i32 next_pos = (i32)(position.x - settings->body_radius);
            const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
            if (next_tile.is_full) position.x = (f32)(next_pos + 1) + settings->body_radius;
        }

        if (movement.y < 0.0f) {
            i32 next_pos = (i32)(position.y - settings->body_radius);
            const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
            if (next_tile.is_full) position.y = (f32)(next_pos + 1) + settings->body_radius;
        } else if (movement.y > 0.0f) {
            i32 next_pos = (i32)(position.y + settings->body_radius);
            const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
            if (next_tile.is_full) position.y = (f32)next_pos - settings->body_radius;
        }

        for (u32 i = 0; i < tile_map.columns.size; i++) {
            const Circle& column{tile_map.columns.data[i]};

            vec2 vector_to_column = column.position - position;
            f32 distance_to_column = vector_to_column.length();
            f32 min_distance_allowed = settings->body_radius + column.radius;
            if (distance_to_column < min_distance_allowed)
                position -= (vector_to_column / distance_to_column) * (min_distance_allowed - distance_to_column);
        }

        camera.position.x = position.x;
        camera.position.z = position.y;
        ray_caster.position = position;
        moveTileMap(tile_map, position);
        if (useGPU) {
            uploadEdges(tile_map.edges);
            uploadVisibleEdgeIds(tile_map.visible_edge_ids);
        }
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

    void onStopEditing() {
        render_state.hovered_pos = 0.0f;
        adding_column = false;
        adding_tiles = false;
        removing_tiles = false;
    }

    void onEditHover(TileMap& tile_map, vec2i mouse_pos, bool crete_new_column = false) {
        if ((render_state.flags & (EDITING_WALLS | EDITING_COLUMNS)) == 0 ||
            mouse_pos.x < 0 ||
            mouse_pos.y < 0 ||
            mouse_pos.x >= ray_caster.screen_width ||
            mouse_pos.y >= ray_caster.screen_height) {
            return;
        }

        const WallHit& wall_hit{wall_hits[mouse_pos.x]};
        const GroundHit& ground_hit{ground_hits[mouse_pos.y]};

        const vec2 position = ray_caster.position + wall_hit.ray_direction * ground_hit.z;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(settings->tile_map_width - 1),
            (f32)(settings->tile_map_height - 1)
        };
        if (!inRange(start, position, end)) {
            return;
        }

        render_state.hovered_pos = position;

        if (crete_new_column) {
            f32 distance_to_body = (position - ray_caster.position).length() - settings->initial_column_radius - settings->body_radius;
            if (distance_to_body > 0.0f) {
                Circle& column{tile_map.columns[tile_map.columns.size++]};
                column.position = position;
                column.radius = settings->initial_column_radius;
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
                render_state.hovered_pos = 0.0f;
            }
        } else if (adding_column) {
            Circle& column{tile_map.columns[tile_map.columns.size - 1]};
            f32 new_radius = (position - column.position).length();
            f32 distance_to_body = (position - ray_caster.position).length() - new_radius - settings->body_radius;
            if (distance_to_body <= 0.0f) new_radius += distance_to_body;
            new_radius = fmaxf(0.1f, new_radius);
            if (new_radius != column.radius) {
                column.radius = new_radius;
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
            }
            render_state.hovered_pos = 0.0f;
        } else {
            Tile& tile{tile_map.cells[(i32)position.y][(i32)position.x]};
            bool tile_changed = false;
            if (adding_tiles) {
                if (!tile.is_full &&
                    !((i32)position.x == (i32)ray_caster.position.x &&
                      (i32)position.y == (i32)ray_caster.position.y)) {
                    tile_changed = true;
                    tile.is_full = true;
                    tile.left.texture_id = tile.right.texture_id = tile.bottom.texture_id = tile.top.texture_id = 12;
                }
            } else if (removing_tiles && tile.is_full) {
                tile_changed = true;
                tile.is_full = false;
            }

            if (tile_changed) {
                generateTileMapEdges(tile_map);
                moveTileMap(tile_map, ray_caster.position);
                if (useGPU) {
                    uploadVisibleEdgeIds(tile_map.visible_edge_ids);
                    uploadEdges(tile_map.edges);
                }
                generateWallHits(tile_map);
            }
        }
    }

    void onEditLeftMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (render_state.flags & EDITING_WALLS) {
            onStopEditing();
            adding_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if ((render_state.flags & EDITING_COLUMNS) && (tile_map.columns.size < MAX_COLUMN_COUNT)) {
            adding_column = true;
            onEditHover(tile_map, mouse_pos, true);
        }
    }

    void onEditRightMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (render_state.flags & EDITING_WALLS) {
            onStopEditing();
            removing_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if ((render_state.flags & EDITING_COLUMNS) && (tile_map.columns.size != 0)) {
            const WallHit& wall_hit{wall_hits[mouse_pos.x]};
            if (wall_hit.column_id != INVALID_COLUMN_ID) {
                tile_map.columns[wall_hit.column_id] = tile_map.columns[--tile_map.columns.size];
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
            }
        }
    }

    void renderOnCPU(u32* window_content, const TileMap& tile_map) {
        PixelShader pixel_shader{*settings, render_state};
        u32 offset = 0;
        for (u16 y = 0; y < ray_caster.screen_height; y++) {
            GroundHit ground_hit = ground_hits[y];
            for (u16 x = 0; x < ray_caster.screen_width; x++, offset++) {
                window_content[offset] = pixel_shader.shade(
                    ground_hit,
                    wall_hits[x],
                    tile_map.edges,
                    tile_map.columns,
                    ray_caster.position,
                    y,
                    ray_caster.mid_point).asContent();
            }
        }
    }

    void init(RayCasterSettings* render_settings, const Dimensions& dim, Camera& camera, TileMap& tile_map)
    {
        settings = render_settings;
        render_state.init();

        Texture &texture{settings->textures[0]};
        ray_caster.texel_size = 1.0f / (f32)texture.width;
        ray_caster.last_mip = (u8)(texture.mip_count - 1);

        initDataOnGPU(*settings);
        uploadEdges(tile_map.edges);

        prior_screen_height = 0;
        prior_up_aim = 0.0f;

        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(u32* window_content, const TileMap& tile_map) {
        #ifdef __CUDACC__
        if (useGPU) renderOnGPU(ray_caster, render_state, window_content);
        else        renderOnCPU(window_content, tile_map);
        #else
        renderOnCPU(window_content, tile_map);
        #endif
    }
};
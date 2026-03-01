#include "./textures.h"

#include "../slim/viewport/navigation.h"
#include "../slim/renderer/renderer.h"
#include "../slim/draw/hud.h"
#include "../slim/app.h"



// WALLS := `
// 11111111111111111111111111111111
// 1_________1____________________1
// 1__1___11_1____________________1
// 111111_1111____________________1
// 1__1_____11____________________1
// 1____11___1____________________1
// 1_____1_111____________________1
// 1_11111__11____________________1
// 1_________1____________________1
// 11111111111____________________1
// 1______________________________1
// 1___________11111111___________1
// 1__________________1___________1
// 1__________________1___________1
// 1__________________1___________1
// 1__________________11111111____1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 11111111111111111111111111111111
// `;

TileSide TILE_SIDE{Texture_SoneWall9Color};
Tile W_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, true, true, true, true, true};
Tile F_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, true, true, true, true, true};
Tile T_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, true, true, true, true, true};

Tile* F{&F_TILE};
Tile* T{&T_TILE};
Tile* W{&W_TILE};
Tile* I{nullptr};

Tile* WALLS[] = {
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
	W,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,W,I,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,I,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,W,I,I,I,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,W,W,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,W,I,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,W,W,W,W,W,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,W,W,W,W,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,W,W,W,W,W,W,W,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
};
Tile* WALLS2[] = {
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
};


struct DungeonCrawler : SlimApp {
	bool AO_map_enabled = true;
	bool normal_map_enabled = true;
	bool roughness_map_enabled = true;

    // HUD:
    HUDLine FPS {"FPS : "};
    HUDLine GPU {"GPU : ", "Off", "On", &ray_cast_renderer::useGPU};
    HUDLine Mode{"Mode: ", "Beauty"};
    HUDLine BRDF{"BRDF: ", "GGX"};
    HUDLine Normal{"Normal : ", "Off", "On", &normal_map_enabled};
    HUDLine AO{"AO : ", "Off", "On", &AO_map_enabled};
    HUDLine Roughness{"Roughness : ", "Off", "On", &roughness_map_enabled};
    HUD hud{{7}, &FPS};


	// Viewport:
    Camera camera{{0, 0 * DEG_TO_RAD, 0}, {13, 0, 3}};
    Navigation navigation;
    Dimensions dimensions;

    TileMap tile_map;
	Slice<Texture> textures_slice{textures, Texture_Count};

	RayCasterSettings settings;

	bool initted = false;

	Color light_color{1.0f, 0.75f, 0.5f};
	f32 light_intensity = 4.0f;
	f32 time = 0.0f;

    void OnUpdate(f32 delta_time) override {
        i32 fps = (i32)render_timer.average_frames_per_second;
        FPS.value = fps;
        FPS.value_color = fps >= 60 ? Green : (fps >= 24 ? Cyan : (fps < 12 ? Red : Yellow));

		bool tile_map_changed = false;

        if (!controls::is_pressed::alt) {
	        navigation.update(camera, delta_time);
        	if (navigation.moved || tile_map_changed) ray_cast_renderer::onMove(camera, tile_map);
        	if (navigation.moved || tile_map_changed ||
	            navigation.turned ||
	            navigation.zoomed) ray_cast_renderer::onScreenChanged(camera, tile_map);
        }
    	navigation.moved = navigation.turned = navigation.zoomed = false;

    	time += delta_time;
    	settings.light_intensity = light_intensity * 0.95f + sinf(time*17.0f) * light_intensity * 0.055f + cosf(time*23.0f) * light_intensity * 0.075f;
    	// settings.light_position_x = ray_cast_renderer::ray_caster.position.x + camera.orientation.X.x  * (sinf(time*2.7f) * 0.19f + cosf(time*2.6f) * 0.19f) + camera.orientation.Z.x  * (cosf(time*2.5f) * 0.23f + sinf(time*2.6f) * 0.155f);
    	// settings.light_position_y = ray_cast_renderer::ray_caster.position.y + sinf(time*2.0f) * 0.15f + cosf(time*2.6f) * 0.07f;
    	// settings.light_position_z = 0.15f + sinf(time*2.70f) * 0.25f + cosf(time*2.50f) * 0.15f;

    	vec3 light_pos = vec3{ray_cast_renderer::ray_caster.position.x, 0.0f, ray_cast_renderer::ray_caster.position.y};
    	light_pos += camera.orientation.X * (sinf(time*2.7f) * 0.09f + cosf(time*2.6f) * 0.09f);
    	light_pos += camera.orientation.Z * 0.2f;
    	light_pos.y += sin(time * 2.0f) * 0.3f + 0.1f;
    	settings.light_position_x = light_pos.x;
    	settings.light_position_y = light_pos.y;
    	settings.light_position_z = light_pos.z;

    	settings.light_color_r = light_color.r;
    	settings.light_color_g = light_color.g - (sinf(time*29.0f) * 0.07f + cosf(time*29.0f) * 0.07f);
    	settings.light_color_b = light_color.b - (sinf(time*19.0f) * 0.06f + cosf(time*19.0f) * 0.06f);

    	ray_cast_renderer::onSettingsChanged();
    }

    void OnRender() override {
        ray_cast_renderer::render(window::content);
        if (hud.enabled)
            drawHUD(hud, window::content, dimensions);
    }

    void OnKeyChanged(u8 key, bool is_pressed) override {
        if (!is_pressed) {
        	RenderMode prior_render_mode = settings.render_mode;
        	u8 prior_flags = settings.flags;
            if (key == controls::key_map::tab) hud.enabled = !hud.enabled;
        	if (key == 'L') settings.flags = BRDF_Lambert | (settings.flags & USE_MAPS_MASK);
        	if (key == 'B') settings.flags = BRDF_Blinn | (settings.flags & USE_MAPS_MASK);
        	if (key == 'X') settings.flags = BRDF_GGX | (settings.flags & USE_MAPS_MASK);
        	if (key == 'P') settings.flags = BRDF_Phong | (settings.flags & USE_MAPS_MASK);
        	if (key == 'O') settings.flags = settings.flags & USE_AO_MAP ? (settings.flags & ~USE_AO_MAP) : (settings.flags | USE_AO_MAP);
        	if (key == 'N') settings.flags = settings.flags & USE_NORMAL_MAP ? (settings.flags & ~USE_NORMAL_MAP) : (settings.flags | USE_NORMAL_MAP);
        	if (key == 'R') settings.flags = settings.flags & USE_ROUGHNESS_MAP ? (settings.flags & ~USE_ROUGHNESS_MAP) : (settings.flags | USE_ROUGHNESS_MAP);
            if (key == 'G' && USE_GPU_BY_DEFAULT) ray_cast_renderer::toggleUseOfGPU();
            if (key == '1') settings.render_mode = RenderMode_Beauty;
            if (key == '2') settings.render_mode = RenderMode_Untextured;
            if (key == '3') settings.render_mode = RenderMode_Depth;
            if (key == '4') settings.render_mode = RenderMode_MipLevel;
        	if (key == '5') settings.render_mode = RenderMode_UVs;
        	if (key == '6') settings.render_mode = RenderMode_Color;
        	if (key == '7') settings.render_mode = RenderMode_Roughness;
        	if (key == '8') settings.render_mode = RenderMode_AO;
        	if (key == '9') settings.render_mode = RenderMode_Normal;
        	if (key == '0') settings.render_mode = RenderMode_Light;
            switch (settings.render_mode) {
                case RenderMode_Beauty:     Mode.value.string = "Beauty"; break;
                case RenderMode_Untextured: Mode.value.string = "Untextured"; break;
                case RenderMode_Depth:      Mode.value.string = "Depth"; break;
                case RenderMode_MipLevel:   Mode.value.string = "Mip Level"; break;
            	case RenderMode_UVs:        Mode.value.string = "UVs"; break;
            	case RenderMode_Color:      Mode.value.string = "Color"; break;
            	case RenderMode_Roughness:  Mode.value.string = "Roughness"; break;
            	case RenderMode_AO:         Mode.value.string = "AO"; break;
            	case RenderMode_Normal:     Mode.value.string = "Normal"; break;
            	case RenderMode_Light:      Mode.value.string = "Lighting"; break;
            }
        	AO_map_enabled = settings.flags & USE_AO_MAP;
        	normal_map_enabled = settings.flags & USE_NORMAL_MAP;
        	roughness_map_enabled = settings.flags & USE_ROUGHNESS_MAP;
        	switch ((BRDFType)(settings.flags & 3)) {
        		case BRDF_Lambert: BRDF.value.string = "Lambert"; break;
        		case BRDF_Blinn: BRDF.value.string = "Blinn"; break;
        		case BRDF_Phong: BRDF.value.string = "Phong"; break;
        		case BRDF_GGX: BRDF.value.string = "GGX"; break;
        	}
        	if (settings.render_mode != prior_render_mode ||
        		settings.flags != prior_flags)
        		ray_cast_renderer::onSettingsChanged();
        }
        Move &move = navigation.move;
        Turn &turn = navigation.turn;
        if (key == 'Q') turn.left     = is_pressed;
        if (key == 'E') turn.right    = is_pressed;
        if (key == 'W') move.forward  = is_pressed;
        if (key == 'S') move.backward = is_pressed;
        if (key == 'A') move.left     = is_pressed;
        if (key == 'D') move.right    = is_pressed;
    }

    void OnWindowResize(u16 width, u16 height) override {
        dimensions.update(width, height);

    	if (initted) ray_cast_renderer::onResize(width, height, camera, tile_map);
    	else {
    		initted = true;
    		ray_cast_renderer::useGPU = USE_GPU_BY_DEFAULT;

    		// F->bottom.portal_to = &T->right;
    		// T->right.portal_to = &F->bottom;

    		initTileMap(tile_map);
    		readTileMap(tile_map, SliceFromStaticArray(Tile*, WALLS));
    		generateTileMapEdges(tile_map);

    		settings.init(textures_slice, Texture_SoneWall9Color, Texture_SoneWall6Color, tile_map.width, tile_map.height);

    		ray_cast_renderer::init(&settings, dimensions, camera, tile_map);
    	}
    }

    void OnMouseButtonDown(mouse::Button &mouse_button) override {
        mouse::pos_raw_diff_x = mouse::pos_raw_diff_y = 0;
    }

    void OnMouseButtonDoubleClicked(mouse::Button &mouse_button) override {
        if (&mouse_button == &mouse::left_button) {
            mouse::is_captured = !mouse::is_captured;
            os::setCursorVisibility(!mouse::is_captured);
            os::setWindowCapture(    mouse::is_captured);
            OnMouseButtonDown(mouse_button);
        }
    }
};

SlimApp* createApp() {
	return (SlimApp*)new DungeonCrawler();
}
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
	bool fired = false;
	f32 time = 0.0f;

    void OnUpdate(f32 delta_time) override {
        i32 fps = (i32)render_timer.average_frames_per_second;
        FPS.value = fps;
        FPS.value_color = fps >= 60 ? Green : (fps >= 24 ? Cyan : (fps < 12 ? Red : Yellow));

		bool tile_map_changed = false;

        if (mouse::is_captured) {
	        navigation.update(camera, delta_time);
        	if (navigation.moved || tile_map_changed) ray_cast_renderer::onMove(camera, tile_map);
        	if (navigation.moved || tile_map_changed ||
	            navigation.turned ||
	            navigation.zoomed) ray_cast_renderer::onScreenChanged(camera, tile_map);
        } else ray_cast_renderer::onEditHover(tile_map, {mouse::pos_x, mouse::pos_y});
    	navigation.moved = navigation.turned = navigation.zoomed = false;

    	time += delta_time;

    	PointLight& torch{ray_cast_renderer::render_state.lights[0]};
    	torch.position = vec3{ray_cast_renderer::ray_caster.position.x, 0.0f, ray_cast_renderer::ray_caster.position.y};
    	torch.position += camera.orientation.X * (sinf(time*2.7f) * 0.09f + cosf(time*2.6f) * 0.09f);
    	torch.position += camera.orientation.Z * 0.2f;
    	torch.position.y += sinf(time * 2.0f) * 0.3f + 0.1f;

    	torch.flicker(ray_cast_renderer::torch_light_color, ray_cast_renderer::torch_light_intensity, time);

    	if (ray_cast_renderer::projectile_count)
    		ray_cast_renderer::updateProjectiles(time, delta_time, tile_map);

    	if (fired) {
    		fired = false;
    		ray_cast_renderer::shoot(time);
    	}
    }

    void OnRender() override {
        ray_cast_renderer::render(window::content);
        if (hud.enabled)
            drawHUD(hud, window::content, dimensions);
    }

    void OnKeyChanged(u8 key, bool is_pressed) override {
    	u8& flags{ray_cast_renderer::render_state.flags};
    	RenderMode& render_mode{ray_cast_renderer::render_state.render_mode};
    	if (is_pressed) {
    		if (key == controls::key_map::ctrl) flags |= EDITING_WALLS;
    		if (key == controls::key_map::alt) flags |= EDITING_COLUMNS;
    	} else {
    		if (key == controls::key_map::space && ray_cast_renderer::projectile_count < 7) fired = true;
    		if (key == controls::key_map::ctrl) flags &= ~EDITING_WALLS;
    		if (key == controls::key_map::alt) flags &= ~EDITING_COLUMNS;
    		if ((flags & (EDITING_WALLS | EDITING_COLUMNS)) == 0) ray_cast_renderer::onStopEditing();
            if (key == controls::key_map::tab) hud.enabled = !hud.enabled;
        	if (key == 'L') flags = BRDF_Lambert | (flags & USE_MAPS_MASK);
        	if (key == 'B') flags = BRDF_Blinn | (flags & USE_MAPS_MASK);
        	if (key == 'X') flags = BRDF_GGX | (flags & USE_MAPS_MASK);
        	if (key == 'P') flags = BRDF_Phong | (flags & USE_MAPS_MASK);
        	if (key == 'O') flags = flags & USE_AO_MAP ? (flags & ~USE_AO_MAP) : (flags | USE_AO_MAP);
        	if (key == 'N') flags = flags & USE_NORMAL_MAP ? (flags & ~USE_NORMAL_MAP) : (flags | USE_NORMAL_MAP);
        	if (key == 'R') flags = flags & USE_ROUGHNESS_MAP ? (flags & ~USE_ROUGHNESS_MAP) : (flags | USE_ROUGHNESS_MAP);
            if (key == 'G' && USE_GPU_BY_DEFAULT) ray_cast_renderer::toggleUseOfGPU();
            if (key == '1') render_mode = RenderMode_Beauty;
            if (key == '2') render_mode = RenderMode_Untextured;
            if (key == '3') render_mode = RenderMode_Depth;
            if (key == '4') render_mode = RenderMode_MipLevel;
        	if (key == '5') render_mode = RenderMode_UVs;
        	if (key == '6') render_mode = RenderMode_Color;
        	if (key == '7') render_mode = RenderMode_Roughness;
        	if (key == '8') render_mode = RenderMode_AO;
        	if (key == '9') render_mode = RenderMode_Normal;
        	if (key == '0') render_mode = RenderMode_Light;
            switch (render_mode) {
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
        	AO_map_enabled = flags & USE_AO_MAP;
        	normal_map_enabled = flags & USE_NORMAL_MAP;
        	roughness_map_enabled = flags & USE_ROUGHNESS_MAP;
        	switch ((BRDFType)(flags & 3)) {
        		case BRDF_Lambert: BRDF.value.string = "Lambert"; break;
        		case BRDF_Blinn: BRDF.value.string = "Blinn"; break;
        		case BRDF_Phong: BRDF.value.string = "Phong"; break;
        		case BRDF_GGX: BRDF.value.string = "GGX"; break;
        	}
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
    	if (ray_cast_renderer::render_state.flags & (EDITING_COLUMNS | EDITING_WALLS)) {
    		if (&mouse_button == &mouse::left_button) ray_cast_renderer::onEditLeftMouseButtonDown(tile_map, {mouse::pos_x, mouse::pos_y});
    		if (&mouse_button == &mouse::right_button) ray_cast_renderer::onEditRightMouseButtonDown(tile_map, {mouse::pos_x, mouse::pos_y});
    	}
    }

	void OnMouseButtonUp(mouse::Button &mouse_button) override {
    	if (&mouse_button == &mouse::left_button ||
			&mouse_button == &mouse::right_button)
    		ray_cast_renderer::onStopEditing();
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
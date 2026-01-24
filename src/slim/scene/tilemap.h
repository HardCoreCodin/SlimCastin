#pragma once

#include "../math/vec2.h"

#include <unordered_map>


#define MAX_TILE_MAP_VIEW_DISTANCE 42
#define MAX_TILE_MAP_WIDTH 32
#define MAX_TILE_MAP_HEIGHT 32
#define MAX_TILE_MAP_SIZE (MAX_TILE_MAP_WIDTH * MAX_TILE_MAP_HEIGHT)
#define MAX_TILE_MAP_VERTICES ((MAX_TILE_MAP_WIDTH + 1) * (MAX_TILE_MAP_HEIGHT + 1))
#define MAX_TILE_MAP_EDGES (MAX_TILE_MAP_WIDTH * (MAX_TILE_MAP_HEIGHT + 1) + MAX_TILE_MAP_HEIGHT * (MAX_TILE_MAP_WIDTH + 1))

#define MAX_COLUMN_COUNT 16


struct TopLeft2Df { f32 left, top; };
struct TopLeft2Di { i32 left, top; };
struct BottomRight2Df { f32 right, bottom; };
struct BottomRight2Di { i32 right, bottom; };

// union TopLeft2DfMin { vec2 min; TopLeft2Df top_left; };
// union TopLeft2DiMin { vec2i min; TopLeft2Di top_left; };
// union BottomRight2DfMax { vec2 max;  BottomRight2Df bottom_right; };
// union BottomRight2DiMax { vec2i max; BottomRight2Di bottom_right; };

struct Bounds2Df { TopLeft2Df tl; BottomRight2Df br; };
struct Bounds2Di { TopLeft2Di tl; BottomRight2Di br; };

INLINE_XPU bool inRange(i32 start, i32 value, i32 end) { return value >= start && value <= end; }
INLINE_XPU bool inRange(f32 start, f32 value, f32 end) { return value >= start && value <= end; }

bool inBounds(const Bounds2Df &b, const vec2 &p) { return inRange(b.tl.left, p.x, b.br.right) && inRange(b.tl.top, p.y, b.br.bottom); }
bool inBounds(const Bounds2Di &b, const vec2i &p) { return inRange(b.tl.left, p.x, b.br.right) && inRange(b.tl.top, p.y, b.br.bottom); }

// Rect2Df :: struct {using bounds: Bounds2Df, using size: Size2Df, position: vec2}
// Rect2Di :: struct {using bounds: Bounds2Di, using size: Size2Di, position: vec2i}


struct Circle {
	vec2 position;
	f32 radius;
};


struct LocalEdge {
	vec2* from;
	vec2* to;
	LocalEdge * portal_to;
	bool is_above;
	bool is_below;
	bool is_left;
	bool is_right;
};


struct TileEdge {
	LocalEdge local;
	vec2i* from;
	vec2i* to;
	i32 length;
	i32 portal_ray_rotation;
	TileEdge* portal_to;
	bool portal_edge_dir_flip;
	u8 texture_id;
	bool
		is_visible,
	    is_vertical,
	    is_horizontal,
	    is_facing_up,
	    is_facing_down,
	    is_facing_left,
	    is_facing_right,
		is_facing_forward;
};


struct TileSide {
	TileSide* portal_to;
	TileSide* portal_from;
	TileEdge* edge;
	u8 texture_id;
};


struct Tile {
	TileSide top, bottom, left, right;

	Bounds2Di bounds;

	bool
	is_full,
	has_left_edge,
	has_right_edge,
	has_top_edge,
	has_bottom_edge;
};

typedef Slice<Tile> TileRow;


struct TileMap : Grid<Tile> {
	Slice<TileEdge> edges;

	i32 edge_count;
	i32 vertex_count;
	i32 column_count;

	Slice<vec2i> vertices;
	Slice<vec2> vertices_in_local_space;

	Circle columns[MAX_COLUMN_COUNT];
	u8 columns_texture_id;

	i32 portal_sides_count;
	Slice<TileSide*> portal_sides;
	std::unordered_map<TileSide*, Tile*> side_to_tile;

	TileSide* all_portal_sides[MAX_TILE_MAP_EDGES];
	TileRow all_rows[MAX_TILE_MAP_HEIGHT];
	Tile all_tiles[MAX_TILE_MAP_SIZE];
	TileEdge all_edges[MAX_TILE_MAP_EDGES];
	vec2i all_vertices[MAX_TILE_MAP_VERTICES];
	vec2 all_vertices_in_local_space[MAX_TILE_MAP_VERTICES];
};


void initTileSide(TileSide* ts) {
	ts->portal_from = nullptr;
	ts->portal_to = nullptr;
	ts->edge = nullptr;
	ts->texture_id = 0;
}


void initTile(Tile* t) {
	initTileSide(&t->top);
	initTileSide(&t->bottom);
	initTileSide(&t->left);
	initTileSide(&t->right);

	t->is_full = false;

	t->has_left_edge = false;
	t->has_right_edge = false;
	t->has_top_edge = false;
	t->has_bottom_edge = false;

	t->bounds.tl.top = 0;
	t->bounds.tl.left = 0;
	t->bounds.br.bottom = 0;
	t->bounds.br.right = 0;
}


void initTileEdge(TileEdge* te) {
	te->local.from = nullptr;
	te->local.to = nullptr;
	te->local.portal_to = nullptr;
	te->local.is_above = false;
	te->local.is_below = false;
	te->local.is_left = false;
	te->local.is_right = false;

	te->length = 1;
	te->texture_id = 0;

	te->from = nullptr;
	te->to = nullptr;

	te->portal_ray_rotation= 0;
	te->portal_edge_dir_flip = false;
	te->portal_to = nullptr;

	te->is_visible = false;
	te->is_vertical = false;

	te->is_facing_left = false;
	te->is_facing_right = false;
	te->is_facing_up = false;
	te->is_facing_down = false;
	te->is_facing_forward = false;
}


void initTileMap(TileMap& tm, u16 Width = MAX_TILE_MAP_WIDTH, u16 Height = MAX_TILE_MAP_HEIGHT) {
	tm.width = Width;
	tm.height = Height;
	tm.columns_texture_id = 0;
	Slice<Tile> all_tiles;
	setSliceToStaticArray(all_tiles, tm.all_tiles);
	for (int i = 0; i < MAX_TILE_MAP_SIZE; i++) initTile(all_tiles.data + i);
	setSliceToStaticArray(tm.portal_sides, tm.all_portal_sides);
	setSliceToStaticArray(tm.edges, tm.all_edges);
	setSliceToStaticArray(tm.vertices, tm.all_vertices);
	setSliceToStaticArray(tm.vertices_in_local_space, tm.all_vertices_in_local_space);
	setSliceToStaticArray(tm.portal_sides, tm.all_portal_sides);
	initGrid<Tile>(tm, Width, Height, all_tiles);
}


void readTileMap(TileMap& tm, Slice<Tile*> map_grid) {
	u32 offset = 0;
    Bounds2Di current_bounds;

	current_bounds.tl.top = 0;
	current_bounds.tl.left = 0;
	current_bounds.br.bottom = 1;
	current_bounds.br.right = 1;

	std::unordered_map<TileSide*, TileSide*> cell_side_to_tile_side;
	// bool has_portals = false;
	tm.portal_sides_count = 0;

	Slice<Tile>* row = nullptr;
	Tile* tile = nullptr;

	iterSlice(tm.cells, row, y) {
		iterSlice((*row), tile, x) {
			tm.side_to_tile[&tile->left] = tile;
			tm.side_to_tile[&tile->right] = tile;
			tm.side_to_tile[&tile->top] = tile;
			tm.side_to_tile[&tile->bottom] = tile;
		}
	}

	iterSlice(tm.cells, row, y) {
		iterSlice((*row), tile, x) {
			initTile(tile);
			tile->bounds = current_bounds;

			Tile* map_cell = map_grid[offset];
			tile->is_full = map_cell != nullptr;
			if (tile->is_full) {
				tile->left = map_cell->left;
				tile->right = map_cell->right;
				tile->top = map_cell->top;
				tile->bottom = map_cell->bottom;

				cell_side_to_tile_side[&map_cell->left] = &tile->left;
				cell_side_to_tile_side[&map_cell->right] = &tile->right;
				cell_side_to_tile_side[&map_cell->top] = &tile->top;
				cell_side_to_tile_side[&map_cell->bottom] = &tile->bottom;

				if (map_cell->left.portal_to != nullptr) {
					tm.portal_sides[tm.portal_sides_count] = &tile->left;
					tm.portal_sides_count += 1;
				}
				if (map_cell->right.portal_to != nullptr) {
					tm.portal_sides[tm.portal_sides_count]= &tile->right;
					tm.portal_sides_count += 1;
				}
				if (map_cell->top.portal_to != nullptr) {
					tm.portal_sides[tm.portal_sides_count]= &tile->top;
					tm.portal_sides_count += 1;
				}
				if (map_cell->bottom.portal_to != nullptr) {
					tm.portal_sides[tm.portal_sides_count]= &tile->bottom;
					tm.portal_sides_count += 1;
				}
			} else {
				tile->is_full = false;
			}

			current_bounds.tl.left += 1;
			current_bounds.br.right += 1;
			offset += 1;
		}

        current_bounds.tl.left = 0;
		current_bounds.br.right = 1;
		current_bounds.tl.top += 1;
		current_bounds.br.bottom += 1;
    }

	if (tm.portal_sides_count != 0) {
		offset = 0;
		iterSlice(tm.cells, row, y) {
			iterSlice((*row), tile, x) {
				Tile* map_cell = map_grid[offset];
				if (map_cell) {
					if (map_cell->left.portal_to != nullptr) {
						tile->left.portal_to = cell_side_to_tile_side[map_cell->left.portal_to];
						tile->left.portal_to->portal_from = &tile->left;
					}
					if (map_cell->right.portal_to != nullptr) {
						tile->right.portal_to = cell_side_to_tile_side[map_cell->right.portal_to];
						tile->right.portal_to->portal_from = &tile->right;
					}
					if (map_cell->top.portal_to != nullptr) {
						tile->top.portal_to = cell_side_to_tile_side[map_cell->top.portal_to];
						tile->top.portal_to->portal_from = &tile->top;
					}
					if (map_cell->bottom.portal_to != nullptr) {
						tile->bottom.portal_to = cell_side_to_tile_side[map_cell->bottom.portal_to];
						tile->bottom.portal_to->portal_from = &tile->bottom;
					}
				}

				offset += 1;
			}
		}
	}
}


void moveTileMap(TileMap& tm, const vec2& origin) {
	vec2i* vertex = nullptr;
	iterSlice(tm.vertices, vertex, i) {
		tm.vertices_in_local_space[i].x = (f32)vertex->x - origin.x;
		tm.vertices_in_local_space[i].y = (f32)vertex->y - origin.y;
	}

	TileEdge* edge = nullptr;
	iterSlice(tm.edges, edge, i) {
		edge->local.is_right = edge->local.from->x > 0;
		edge->local.is_below = edge->local.from->y > 0;
		edge->local.is_left  = edge->local.to->x < 0;
		edge->local.is_above = edge->local.to->y < 0;

		edge->is_facing_forward = edge->is_vertical ?
			(edge->is_facing_left && edge->local.is_right || edge->is_facing_right && edge->local.is_left) :
			(edge->is_facing_down && edge->local.is_above || edge->is_facing_up    && edge->local.is_below);
	}
}


struct TileCheck {
	bool exists;
	Tile* tile;
	Slice<Tile> row;
};


void generateTileMapEdges(TileMap& tm) {
	TileCheck above, below, left, right;

	vec2i position;
	u16 vertex_id = 0;
	u16 edge_id = 0;

	Slice<Tile>* row = nullptr;
	Tile* current_tile = nullptr;

	Slice<Tile> _{nullptr, 0};

	iterSlice(tm.cells, row, y) {
		above.exists = y > 0;
		below.exists = (i32)y < tm.height - 1;

		above.row = above.exists ? tm.cells[y - 1] : _;
		below.row = below.exists ? tm.cells[y + 1] : _;

		iterSlice((*row), current_tile, x) {
        	left.exists  = x > 0;
        	right.exists = (i32)x < (tm.width - 1);

        	left.tile  = left.exists  ? &(*row)[x - 1] : nullptr;
        	right.tile = right.exists ? &(*row)[x + 1] : nullptr;
        	above.tile = above.exists ? &above.row[x] : nullptr;
        	below.tile = below.exists ? &below.row[x] : nullptr;

        	if (current_tile->is_full) {
				current_tile->has_left_edge   = left.exists  &&  !left.tile->is_full;
	        	current_tile->has_right_edge  = right.exists && !right.tile->is_full;
	        	current_tile->has_top_edge    = above.exists && !above.tile->is_full;
	        	current_tile->has_bottom_edge = below.exists && !below.tile->is_full;

	        	if (current_tile->has_left_edge) { // Create/extend left edge:
		        	if (above.exists && above.tile->has_left_edge &&
					    above.tile->left.portal_to == current_tile->left.portal_to && current_tile->left.portal_from == nullptr) { // Tile above has a left edge, extend it:
		        		current_tile->left.edge = above.tile->left.edge;
		        		current_tile->left.edge->length += 1;
		        		current_tile->left.edge->to->y += 1;
		        	} else { // No left edge above - create new one:
		        		current_tile->left.edge = &tm.all_edges[edge_id];
		        		initTileEdge(current_tile->left.edge);
						current_tile->left.edge->texture_id = current_tile->left.texture_id;
						edge_id += 1;

		        		current_tile->left.edge->from = nullptr;
		        		current_tile->left.edge->local.from = nullptr;
		        		if (left.exists && above.exists) {
		        			Tile* top_left = &above.row[x-1];
		        			if (top_left->is_full &&
		        			    top_left->has_right_edge &&
		        			    top_left->has_bottom_edge) {
		        				current_tile->left.edge->from = top_left->bottom.edge->to;
		        				current_tile->left.edge->local.from = top_left->bottom.edge->local.to;
		        			}
		        		}

		        		if (current_tile->left.edge->from == nullptr) {
		        			current_tile->left.edge->from = &tm.all_vertices[vertex_id];
		        			current_tile->left.edge->local.from = &tm.all_vertices_in_local_space[vertex_id];
		        			vertex_id += 1;

		        			*current_tile->left.edge->from = position;
		        		}

		        		current_tile->left.edge->to = &tm.all_vertices[vertex_id];
		        		current_tile->left.edge->local.to = &tm.all_vertices_in_local_space[vertex_id];
		        		vertex_id += 1;

		        		*current_tile->left.edge->to = position;
		        		current_tile->left.edge->to->y += 1;

		        		current_tile->left.edge->is_vertical = true;
		        		current_tile->left.edge->is_horizontal = false;
		        		current_tile->left.edge->is_facing_left = true;
			        }
			    }

				if (current_tile->has_right_edge) { // Create/extend right edge:
		        	if (above.exists && above.tile->has_right_edge &&
					    above.tile->right.portal_to == current_tile->right.portal_to && current_tile->right.portal_from == nullptr) { // Tile above has a right edge, extend it:
		        		current_tile->right.edge = above.tile->right.edge;
		        		current_tile->right.edge->length += 1;
		        		current_tile->right.edge->to->y += 1;
		        	} else { // No right edge above - create new one:
		        		current_tile->right.edge = &tm.all_edges[edge_id];
		        		initTileEdge(current_tile->right.edge);
						current_tile->right.edge->texture_id = current_tile->right.texture_id;
						edge_id += 1;

						current_tile->right.edge->from = nullptr;
		        		current_tile->right.edge->local.from = nullptr;
		        		if (right.exists && above.exists) {
		        			Tile* top_right = &above.row[x+1];
		        			if (top_right->is_full &&
		        			    top_right->has_left_edge &&
		        			    top_right->has_bottom_edge) {
		        				current_tile->right.edge->from = top_right->bottom.edge->from;
		        				current_tile->right.edge->local.from = top_right->bottom.edge->local.from;
		        			}
		        		}

		        		if (current_tile->right.edge->from == nullptr) {
		        			current_tile->right.edge->from = &tm.all_vertices[vertex_id];
		        			current_tile->right.edge->local.from = &tm.all_vertices_in_local_space[vertex_id];
		        			vertex_id += 1;

		        			*current_tile->right.edge->from = position;
		        			current_tile->right.edge->from->x += 1;
		        		}

						current_tile->right.edge->to = &tm.all_vertices[vertex_id];
		        		current_tile->right.edge->local.to = &tm.all_vertices_in_local_space[vertex_id];
		        		vertex_id += 1;

		        		*current_tile->right.edge->to = position;
		        		current_tile->right.edge->to->x += 1;
		        		current_tile->right.edge->to->y += 1;

		        		current_tile->right.edge->is_vertical = true;
		        		current_tile->right.edge->is_horizontal = false;
		        		current_tile->right.edge->is_facing_right = true;
			        }
				}

		        if (current_tile->has_top_edge) { // Create/extend top edge:
		        	if (left.exists && left.tile->has_top_edge &&
						left.tile->top.portal_to == current_tile->top.portal_to && current_tile->top.portal_from == nullptr) { // Tile on the left has a top edge, extend it:
		        		current_tile->top.edge = left.tile->top.edge;
		        		current_tile->top.edge->length += 1;
		        		current_tile->top.edge->to->x += 1;
		        	} else { // No top edge on the left - create new one:
		        		current_tile->top.edge = &tm.all_edges[edge_id];
		        		initTileEdge(current_tile->top.edge);
						current_tile->top.edge->texture_id = current_tile->top.texture_id;
						edge_id += 1;

						current_tile->top.edge->from = nullptr;
		        		current_tile->top.edge->local.from = nullptr;
		        		if (left.exists && above.exists) {
		        			Tile* top_left = &above.row[x-1];
		        			if (top_left->is_full &&
		        			   top_left->has_right_edge &&
		        			   top_left->has_bottom_edge) {
		        				current_tile->top.edge->from = top_left->bottom.edge->to;
		        				current_tile->top.edge->local.from = top_left->bottom.edge->local.to;
		        			}
		        		}

						current_tile->top.edge->to = nullptr;
		        		current_tile->top.edge->local.to = nullptr;
		        		if (right.exists && above.exists) {
		        			Tile* top_right = &above.row[x+1];
		        			if (top_right->is_full &&
		        			   top_right->has_left_edge &&
		        			   top_right->has_bottom_edge) {
		        				current_tile->top.edge->to = top_right->bottom.edge->from;
		        				current_tile->top.edge->local.to = top_right->bottom.edge->local.from;
		        			}
		        		}

		        		if (current_tile->top.edge->from == nullptr) {
		        			current_tile->top.edge->from = &tm.all_vertices[vertex_id];
		        			current_tile->top.edge->local.from = &tm.all_vertices_in_local_space[vertex_id];
		        			vertex_id += 1;

		        			*current_tile->top.edge->from = position;
		        		}

						if (current_tile->top.edge->to == nullptr) {
		        			current_tile->top.edge->to = &tm.all_vertices[vertex_id];
		        			current_tile->top.edge->local.to = &tm.all_vertices_in_local_space[vertex_id];
		        			vertex_id += 1;

		        			*current_tile->top.edge->to = position;
		        			current_tile->top.edge->to->x += 1;
		        		}

		        		current_tile->top.edge->is_vertical = false;
		        		current_tile->top.edge->is_horizontal = true;
		        		current_tile->top.edge->is_facing_up = true;
			        }
		        }

		        if (current_tile->has_bottom_edge) { // Create/extend bottom edge:
		        	if (left.exists && left.tile->has_bottom_edge &&
						left.tile->bottom.portal_to == current_tile->bottom.portal_to && current_tile->bottom.portal_from == nullptr) {// Tile on the left has a bottom edge, extend it:
		        		current_tile->bottom.edge = left.tile->bottom.edge;
		        		current_tile->bottom.edge->length += 1;
		        		current_tile->bottom.edge->to->x += 1;
		        	} else { // No bottom edge on the left - create new one:
		        		current_tile->bottom.edge = &tm.all_edges[edge_id];
		        		initTileEdge(current_tile->bottom.edge);
						current_tile->bottom.edge->texture_id = current_tile->bottom.texture_id;
						edge_id += 1;

	        			current_tile->bottom.edge->from = &tm.all_vertices[vertex_id];
	        			current_tile->bottom.edge->local.from = &tm.all_vertices_in_local_space[vertex_id];
	        			vertex_id += 1;

	        			*current_tile->bottom.edge->from = position;
	        			current_tile->bottom.edge->from->y += 1;

	        			current_tile->bottom.edge->to = &tm.all_vertices[vertex_id];
	        			current_tile->bottom.edge->local.to = &tm.all_vertices_in_local_space[vertex_id];
	        			vertex_id += 1;

	        			*current_tile->bottom.edge->to = position;
	        			current_tile->bottom.edge->to->x += 1;
	        			current_tile->bottom.edge->to->y += 1;

		        		current_tile->bottom.edge->is_horizontal = true;
		        		current_tile->bottom.edge->is_facing_down = true;
			        }
	        	}
        	} else {
        		current_tile->has_left_edge   = false;
	        	current_tile->has_right_edge  = false;
	        	current_tile->has_top_edge    = false;
	        	current_tile->has_bottom_edge = false;
        	}

	        current_tile->bounds.tl.left = position.x;
	        current_tile->bounds.br.right = position.x + 1;

	        current_tile->bounds.tl.top = position.y;
	        current_tile->bounds.br.bottom = position.y + 1;

			position.x += 1;
        }

        position.x  = 0;
        position.y += 1;
    }

	setSliceToRangeOfStaticArray(tm.edges, tm.all_edges, 0, edge_id);
	setSliceToRangeOfStaticArray(tm.vertices, tm.all_vertices, 0, vertex_id);
	setSliceToRangeOfStaticArray(tm.vertices_in_local_space, tm.all_vertices_in_local_space, 0, edge_id);

	if (tm.portal_sides_count) {
		for (i32 i = 0; i < tm.portal_sides_count; i++) {
			const TileSide& side = *tm.portal_sides[i];
			TileEdge& from_edge = *side.edge;
			TileEdge& to_edge = *side.portal_to->edge;
			from_edge.portal_to = &to_edge;

			//

			if (from_edge.is_vertical) {
				if (to_edge.is_horizontal) {
					from_edge.portal_edge_dir_flip = to_edge.is_facing_up;
					if (from_edge.is_facing_right) {
						from_edge.portal_ray_rotation = to_edge.is_facing_up ? 90 : -90; // to_edge.is_facing_down
					} else {
						from_edge.portal_ray_rotation = to_edge.is_facing_down ? 90: -90; // to_edge.is_facing_up
					}
				} else {
					from_edge.portal_edge_dir_flip = from_edge.is_facing_right != to_edge.is_facing_right;
					from_edge.portal_ray_rotation = from_edge.portal_edge_dir_flip ? 180 : 0;
				}
			} else {
				if (to_edge.is_vertical) {
					from_edge.portal_edge_dir_flip = to_edge.is_facing_left;
					if (from_edge.is_facing_up) {
						from_edge.portal_ray_rotation = to_edge.is_facing_left ? 90 : -90; // to_edge.is_facing_right
					} else {
						from_edge.portal_ray_rotation = to_edge.is_facing_right ? 90 : -90; // to_edge.is_facing_left
					}
				} else {
					from_edge.portal_edge_dir_flip = from_edge.is_facing_up != to_edge.is_facing_up;
					from_edge.portal_ray_rotation = from_edge.portal_edge_dir_flip ? 180 : 0;
				}
			}
		}
	}
}
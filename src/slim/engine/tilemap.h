#pragma once

#include "tilemap_base.h"


struct TileSide {
	u8 texture_id = 5;
	u16 edge_id = (u16)(-1);
};


struct Tile {
	TileSide top{}, bottom{}, left{}, right{};
	Corner top_left{}, top_right{}, bottom_left{}, bottom_right{};

	bool is_full = false;
	bool has_left_edge = false;
	bool has_right_edge = false;
	bool has_top_edge = false;
	bool has_bottom_edge = false;
};


struct TileMap : Grid<Tile> {
	typedef Slice<Tile> TileRow;

	Slice<Circle> columns;
	Slice<TileEdge> edges;

	u8 columns_texture_id;

	TileRow all_rows[MAX_TILE_MAP_HEIGHT];
	Tile all_tiles[MAX_TILE_MAP_SIZE];
	TileEdge all_edges[MAX_TILE_MAP_EDGES];
	Circle all_columns[MAX_COLUMN_COUNT];

	struct TileCheck {
		bool exists;
		Tile* tile;
		Slice<Tile> row;
	};

	void init(u16 Width = MAX_TILE_MAP_WIDTH, u16 Height = MAX_TILE_MAP_HEIGHT) {
		width = Width;
		height = Height;
		columns_texture_id = 0;

		setSliceToStaticArray(columns, all_columns);
		setSliceToStaticArray(edges, all_edges);

		columns.size = edges.size = 0;
		initGrid<Tile>(*this, Width, Height, {&all_tiles[0], ARRAY_SIZE(all_tiles)});
	}

	void read(Slice<Tile*> map_grid) {
		u32 offset = 0;
		Slice<Tile>* row = nullptr;
		Tile* tile = nullptr;

		iterSlice(cells, row, y) {
			iterSlice((*row), tile, x) {
				*tile = {};
				Tile* map_cell = map_grid[offset++];
				tile->is_full = map_cell != nullptr;
				if (tile->is_full) {
					tile->left = map_cell->left;
					tile->right = map_cell->right;
					tile->top = map_cell->top;
					tile->bottom = map_cell->bottom;
				} else
					tile->is_full = false;
			}
	    }
	}

	void generateEdges() {
		TileCheck above, below, left, right;

		vec2i position;
		edges.size = 0;

		Slice<Tile>* row = nullptr;
		Tile* current_tile = nullptr;

		Slice<Tile> _{nullptr, 0};

		iterSlice(cells, row, y) {
			above.exists = y > 0;
			below.exists = (i32)y < height - 1;

			above.row = above.exists ? cells[y - 1] : _;
			below.row = below.exists ? cells[y + 1] : _;

			iterSlice((*row), current_tile, x) {
        		left.exists  = x > 0;
        		right.exists = (i32)x < (width - 1);

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
		        		if (above.exists && above.tile->has_left_edge) {// &&
		        			current_tile->left.edge_id = above.tile->left.edge_id;
		        			TileEdge& left_edge = edges[current_tile->left.edge_id];
		        			left_edge.to.y++;
		        		} else { // No left edge above - create new one:
		        			current_tile->left.edge_id = (u16)edges.size;
		        			TileEdge& left_edge = edges.data[edges.size++];
		        			left_edge.facing = Facing::Left;
							left_edge.texture_id = current_tile->left.texture_id;
		        			left_edge.to = left_edge.from = position;
		        			left_edge.to.y++;
				        }
				    }

					if (current_tile->has_right_edge) { // Create/extend right edge:
		        		if (above.exists && above.tile->has_right_edge) {// &&
		        			current_tile->right.edge_id = above.tile->right.edge_id;
		        			TileEdge& right_edge = edges.data[above.tile->right.edge_id];
		        			right_edge.to.y++;
		        		} else { // No right edge above - create new one:
		        			current_tile->right.edge_id = (u16)edges.size;
		        			TileEdge& right_edge = edges[edges.size++];
		        			right_edge.facing = Facing::Right;
							right_edge.texture_id = current_tile->right.texture_id;
		        			right_edge.from = right_edge.to = position;
		        			right_edge.from.x++;
		        			right_edge.to.x++;
		        			right_edge.to.y++;
				        }
					}

			        if (current_tile->has_top_edge) { // Create/extend top edge:
		        		if (left.exists && left.tile->has_top_edge) {// &&
		        			current_tile->top.edge_id = left.tile->top.edge_id;
		        			TileEdge& top_edge = edges[left.tile->top.edge_id];
		        			top_edge.to.x++;
		        		} else { // No top edge on the left - create new one:
		        			current_tile->top.edge_id = (u16)edges.size;
		        			TileEdge& top_edge = edges.data[edges.size++];
		        			top_edge.facing = Facing::Up;
		        			top_edge.texture_id = current_tile->top.texture_id;
		        			top_edge.from = top_edge.to = position;
		        			top_edge.to.x++;
				        }
			        }

			        if (current_tile->has_bottom_edge) { // Create/extend bottom edge:
		        		if (left.exists && left.tile->has_bottom_edge) { // &&
		        			current_tile->bottom.edge_id = left.tile->bottom.edge_id;
		        			TileEdge& bottom_edge = edges[left.tile->bottom.edge_id];
		        			bottom_edge.to.x++;
		        		} else { // No bottom edge on the left - create new one:
		        			current_tile->bottom.edge_id = (u16)edges.size;
		        			TileEdge& bottom_edge = edges.data[edges.size++];
		        			bottom_edge.facing = Facing::Down;
		        			bottom_edge.texture_id = current_tile->bottom.texture_id;
		        			bottom_edge.from = bottom_edge.to = position;
		        			bottom_edge.from.y++;
		        			bottom_edge.to.x++;
		        			bottom_edge.to.y++;
				        }
	        		}
        		} else {
        			current_tile->has_left_edge   = false;
	        		current_tile->has_right_edge  = false;
	        		current_tile->has_top_edge    = false;
	        		current_tile->has_bottom_edge = false;
        		}
				position.x += 1;
	        }

	        position.x  = 0;
	        position.y += 1;
	    }

		for (u16 i = 0; i < edges.size; i++) {
			TileEdge& edge = edges.data[i];
			TileEdge* other_edge_from = nullptr;
			TileEdge* other_edge_to = nullptr;
			const bool is_horizontal = isHorizontal(edge.facing);
			for (u16 j = 0; j < edges.size; j++) {
				TileEdge* other_edge = &edges.data[j];
				if (j == i || is_horizontal == isHorizontal(other_edge->facing))
					continue;

				if (other_edge->to == edge.to ||
					other_edge->from == edge.to) {
					other_edge_to = other_edge;
				} else if (other_edge->to == edge.from ||
					other_edge->from == edge.from) {
					other_edge_from = other_edge;
				}
			}

			if (other_edge_to && other_edge_from)
				switch (edge.facing) {
					case Facing::Down: {
						if (other_edge_from->facing == Facing::Right) {
							edge.from_corner.rounding = Rounding::Concave;
							edge.from_corner.horizontal = Facing::Down;
							edge.from_corner.vertical = Facing::Right;
							cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
							// cells[edge.from.y - 1][edge.from.x].bottom_left = edge.from_corner;
							// cells[edge.from.y][edge.from.x - 1].top_right = edge.from_corner;
						} else {
							edge.from_corner.rounding = Rounding::Convex;
							edge.from_corner.horizontal = Facing::Down;
							edge.from_corner.vertical = Facing::Left;
							cells[edge.from.y - 1][edge.from.x].bottom_left = edge.from_corner;
						}
						if (other_edge_to->facing == Facing::Right) {
							edge.to_corner.rounding = Rounding::Convex;
							edge.to_corner.horizontal = Facing::Down;
							edge.to_corner.vertical = Facing::Right;
							cells[edge.to.y - 1][edge.to.x - 1].bottom_right = edge.to_corner;
						} else {
							edge.to_corner.rounding = Rounding::Concave;
							edge.to_corner.horizontal = Facing::Down;
							edge.to_corner.vertical = Facing::Left;
							cells[edge.to.y][edge.to.x - 1].top_right = edge.to_corner;
							// cells[edge.to.y - 1][edge.to.x - 1].bottom_right = edge.to_corner;
							// cells[edge.to.y][edge.to.x].top_left = edge.to_corner;
						}
						break;
					}
					case Facing::Up: {
						if (other_edge_from->facing == Facing::Right) {
							edge.from_corner.rounding = Rounding::Concave;
							edge.from_corner.horizontal = Facing::Up;
							edge.from_corner.vertical = Facing::Right;
							cells[edge.from.y - 1][edge.from.x].bottom_left = edge.from_corner;
							// cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
							// cells[edge.from.y - 1][edge.from.x - 1].bottom_right = edge.from_corner;
						} else {
							edge.from_corner.rounding = Rounding::Convex;
							edge.from_corner.horizontal = Facing::Up;
							edge.from_corner.vertical = Facing::Left;
							cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
						}
						if (other_edge_to->facing == Facing::Right) {
							edge.to_corner.rounding = Rounding::Convex;
							edge.to_corner.horizontal = Facing::Up;
							edge.to_corner.vertical = Facing::Right;
							cells[edge.to.y][edge.to.x - 1].top_right = edge.to_corner;
						} else {
							edge.to_corner.rounding = Rounding::Concave;
							edge.to_corner.horizontal = Facing::Up;
							edge.to_corner.vertical = Facing::Left;
							cells[edge.to.y - 1][edge.to.x - 1].bottom_right = edge.to_corner;
							// cells[edge.to.y][edge.to.x - 1].top_right = edge.to_corner;
							// cells[edge.to.y - 1][edge.to.x].bottom_left = edge.to_corner;
						}
						break;
					}
					case Facing::Right: {
						if (other_edge_from->facing == Facing::Down) {
							edge.from_corner.rounding = Rounding::Concave;
							edge.from_corner.horizontal = Facing::Down;
							edge.from_corner.vertical = Facing::Right;
							cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
							// cells[edge.from.y][edge.from.x - 1].top_right = edge.from_corner;
							// cells[edge.from.y - 1][edge.from.x].bottom_left = edge.from_corner;
						} else {
							edge.from_corner.rounding = Rounding::Convex;
							edge.from_corner.horizontal = Facing::Up;
							edge.from_corner.vertical = Facing::Right;
							cells[edge.from.y][edge.from.x - 1].top_right = edge.from_corner;
						}
						if (other_edge_to->facing == Facing::Down) {
							edge.to_corner.rounding = Rounding::Convex;
							edge.to_corner.horizontal = Facing::Down;
							edge.to_corner.vertical = Facing::Right;
							cells[edge.to.y - 1][edge.from.x - 1].bottom_right = edge.to_corner;
						} else {
							edge.to_corner.rounding = Rounding::Concave;
							edge.to_corner.horizontal = Facing::Up;
							edge.to_corner.vertical = Facing::Right;
							cells[edge.to.y - 1][edge.to.x].bottom_left = edge.to_corner;
							// cells[edge.to.y - 1][edge.to.x - 1].bottom_right = edge.to_corner;
							// cells[edge.to.y][edge.to.x].top_left = edge.to_corner;
						}
						break;
					}
					case Facing::Left: {
						if (other_edge_from->facing == Facing::Down) {
							edge.from_corner.rounding = Rounding::Concave;
							edge.from_corner.horizontal = Facing::Down;
							edge.from_corner.vertical = Facing::Left;
							cells[edge.from.y][edge.from.x - 1].top_right = edge.from_corner;
							// cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
							// cells[edge.from.y - 1][edge.from.x - 1].bottom_right = edge.from_corner;
						} else {
							edge.from_corner.rounding = Rounding::Convex;
							edge.from_corner.horizontal = Facing::Up;
							edge.from_corner.vertical = Facing::Left;
							cells[edge.from.y][edge.from.x].top_left = edge.from_corner;
						}
						if (other_edge_to->facing == Facing::Down) {
							edge.to_corner.rounding = Rounding::Convex;
							edge.to_corner.horizontal = Facing::Down;
							edge.to_corner.vertical = Facing::Left;
							cells[edge.to.y - 1][edge.to.x].bottom_left = edge.to_corner;
						} else {
							edge.to_corner.rounding = Rounding::Concave;
							edge.to_corner.horizontal = Facing::Up;
							edge.to_corner.vertical = Facing::Left;
							cells[edge.to.y - 1][edge.to.x - 1].bottom_right = edge.to_corner;
							// cells[edge.to.y - 1][edge.to.x].bottom_left = edge.to_corner;
							// cells[edge.to.y][edge.to.x - 1].top_right = edge.to_corner;
						}
						break;
					}
					default: break;
				}
		}
	}

	void load(Slice<Tile*> map_grid) {
		init();
		read(map_grid);
		generateEdges();
	}
};
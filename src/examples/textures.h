#pragma once

#include "../slim/serialization/texture.h"
#include "../slim/core/string.h"


const char* texture_files[] = {
    "purple_stone.texture",
    "cobblestone2.texture",
    "colored_stone.texture",
    "red_stone.texture",
    "pebble_stone6c.texture",
    "pebble_stone6r.texture",
    "pebble_stone6n.texture",
    "pebble_stone6a.texture",
    "stone6c.texture",
    "stone6r.texture",
    "stone6n.texture",
    "stone6a.texture",
    "stone9c.texture",
    "stone9r.texture",
    "stone9n.texture",
    "stone9a.texture",
    "stone12c.texture",
    "stone12r.texture",
    "stone12n.texture",
    "stone12a.texture",
    "ground17c.texture",
    "ground17r.texture",
    "ground17n.texture",
    "ground17a.texture",
};

enum TextureID {
    Texture_PurpleStone = 0,
    Texture_CobbleStone,
    Texture_ColoredStone,
    Texture_RedStone,
    Texture_PebbleStone6Color,
    Texture_PebbleStone6Roughness,
    Texture_PebbleStone6Normal,
    Texture_PebbleStone6AO,
    Texture_SoneWall6Color,
    Texture_SoneWall6Roughness,
    Texture_SoneWall6Normal,
    Texture_SoneWall6AO,
    Texture_SoneWall9Color,
    Texture_SoneWall9Roughness,
    Texture_SoneWall9Normal,
    Texture_SoneWall9AO,
    Texture_SoneWall12Color,
    Texture_SoneWall12Roughness,
    Texture_SoneWall12Normal,
    Texture_SoneWall12AO,
    Texture_Ground17Color,
    Texture_Ground17Roughness,
    Texture_Ground17Normal,
    Texture_Ground17AO,

    Texture_Count
};

Texture textures[Texture_Count];
char textures_string_buffers[Texture_Count][200]{};
String texture_strings[Texture_Count] {
    {textures_string_buffers[0], 200},
    {textures_string_buffers[1], 200},
    {textures_string_buffers[2], 200},
    {textures_string_buffers[3], 200},
    {textures_string_buffers[4], 200},
    {textures_string_buffers[5], 200},
    {textures_string_buffers[6], 200},
    {textures_string_buffers[7], 200},
    {textures_string_buffers[8], 200},
    {textures_string_buffers[9], 200},
    {textures_string_buffers[10], 200},
    {textures_string_buffers[11], 200},
    {textures_string_buffers[12], 200},
    {textures_string_buffers[13], 200},
    {textures_string_buffers[14], 200},
    {textures_string_buffers[15], 200},
    {textures_string_buffers[16], 200},
    {textures_string_buffers[17], 200},
    {textures_string_buffers[18], 200},
    {textures_string_buffers[19], 200},
    {textures_string_buffers[20], 200},
    {textures_string_buffers[21], 200},
    {textures_string_buffers[22], 200},
    {textures_string_buffers[23], 200}
};
TexturePack texture_pack{Texture_Count, textures, texture_strings, texture_files, __FILE__, Terabytes(3)};
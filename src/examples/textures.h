#pragma once

#include "../slim/serialization/texture.h"
#include "../slim/core/string.h"


const char* texture_files[] = {
    "purple_stone.texture",
    "cobblestone2.texture",
    "colored_stone.texture",
    "red_stone.texture"
};

enum TextureID {
    Texture_PurpleStone = 0,
    Texture_CobbleStone,
    Texture_ColoredStone,
    Texture_RedStone,

    Texture_Count
};

Texture textures[Texture_Count];
char textures_string_buffers[Texture_Count][200]{};
String texture_strings[Texture_Count] {
    {textures_string_buffers[0], 200},
    {textures_string_buffers[1], 200},
    {textures_string_buffers[2], 200},
    {textures_string_buffers[3], 200}
};
TexturePack texture_pack{Texture_Count, textures, texture_strings, texture_files, __FILE__, Terabytes(3)};
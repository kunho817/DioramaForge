from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PRESET = "Fantasy Diorama"


@dataclass(frozen=True)
class StylePreset:
    name: str
    prompt: str
    palette: tuple[tuple[int, int, int], ...]
    clip_hint: str
    material_prompt: str
    lighting_prompt: str
    color_prompt: str
    camera_prompt: str
    meshy_prompt: str


STYLE_PRESETS: dict[str, StylePreset] = {
    "Fantasy Diorama": StylePreset(
        name="Fantasy Diorama",
        prompt=(
            "a handcrafted fantasy tabletop diorama of the source scene"
        ),
        palette=((54, 84, 69), (127, 146, 88), (198, 166, 103), (124, 82, 71), (52, 45, 65)),
        clip_hint="fantasy tabletop miniature diorama, sculpted terrain, painted backdrop",
        material_prompt=(
            "sculpted terrain foam, hand-painted moss, miniature grass tufts, carved stone accents, "
            "painted backdrop sky, tactile handmade scale-model materials, no new water surfaces"
        ),
        lighting_prompt="warm studio macro lighting that preserves the source light direction",
        color_prompt="forest greens, warm gold highlights, muted stone gray, small accent colors",
        camera_prompt="wide macro photograph from the same camera position and horizon line",
        meshy_prompt=(
            "fantasy miniature terrain, hand-painted moss, grass tufts, carved stone, compact terrain base, "
            "painted backdrop excluded from solid mesh"
        ),
    ),
    "Animated Miniature": StylePreset(
        name="Animated Miniature",
        prompt=(
            "a cozy hand-painted animated miniature environment based on the source scene"
        ),
        palette=((73, 121, 101), (155, 182, 120), (238, 200, 132), (91, 139, 168), (184, 111, 91)),
        clip_hint="cozy hand-painted animated miniature, tactile model environment",
        material_prompt=(
            "painted foam scenery, soft clay-like terrain, small paper foliage, rounded handmade edges, "
            "gentle miniature set details"
        ),
        lighting_prompt="soft daylight with mild bloom and preserved source shadows",
        color_prompt="fresh greens, soft sky blues, peach and ochre highlights",
        camera_prompt="wide miniature set photograph, same perspective and foreground-background order",
        meshy_prompt="hand-painted miniature terrain, soft clay-like surfaces, paper foliage, clean readable color blocks",
    ),
    "Medieval Village": StylePreset(
        name="Medieval Village",
        prompt=(
            "a handcrafted medieval village diorama adapted from the source scene"
        ),
        palette=((69, 58, 48), (139, 92, 62), (174, 143, 92), (86, 107, 82), (43, 62, 76)),
        clip_hint="medieval village miniature diorama, wood and stone model materials",
        material_prompt=(
            "balsa wood, carved stone, clay roof tiles, cobblestone paths only where paths or structures exist, "
            "small handmade village props without adding new focal buildings"
        ),
        lighting_prompt="small warm lantern accents balanced with the source ambient light",
        color_prompt="weathered wood brown, stone gray, moss green, muted clay red",
        camera_prompt="wide tabletop model photograph, same viewpoint and major silhouettes",
        meshy_prompt="wood, stone, clay roof, cobblestone, moss, hand-painted medieval miniature texture boundaries",
    ),
    "Enchanted Forest": StylePreset(
        name="Enchanted Forest",
        prompt=(
            "an enchanted forest diorama transformation of the source scene"
        ),
        palette=((28, 69, 63), (69, 132, 93), (161, 191, 113), (92, 68, 122), (220, 174, 110)),
        clip_hint="enchanted forest miniature diorama, moss, roots, soft magical glow",
        material_prompt=(
            "mossy terrain, tiny leaves, twisted roots following existing vegetation or terrain shapes, "
            "small glowing mushrooms as surface details, no new characters or creatures"
        ),
        lighting_prompt="low soft magical accent light with preserved depth layers",
        color_prompt="deep green, teal shadows, moss yellow, subtle violet accents",
        camera_prompt="wide macro diorama photograph with the same depth layering",
        meshy_prompt="moss, roots, tiny leaves, glowing mushroom accents, forest miniature terrain, clean PBR-friendly colors",
    ),
    "Ruined City": StylePreset(
        name="Ruined City",
        prompt=(
            "a post-apocalyptic ruined city diorama based on the source scene"
        ),
        palette=((59, 70, 73), (111, 116, 100), (150, 126, 89), (76, 103, 79), (136, 76, 58)),
        clip_hint="ruined city miniature diorama, cracked concrete, overgrown vines",
        material_prompt=(
            "cracked concrete, rusted metal, broken masonry, overgrown vines, weathered miniature props, "
            "preserve existing streets and buildings without inventing new landmarks"
        ),
        lighting_prompt="dramatic but readable miniature lighting with preserved source direction",
        color_prompt="weathered gray, olive green, rust red, dusty beige accents",
        camera_prompt="wide miniature photography, same camera angle and scene boundaries",
        meshy_prompt="cracked concrete, rusted metal, broken masonry, vines, weathered hand-painted miniature textures",
    ),
}

CLIP_STYLE_HINTS = {
    "Fantasy Diorama": "fantasy miniature tabletop diorama, sculpted terrain, painted backdrop",
    "Animated Miniature": "hand-painted cozy miniature diorama",
    "Medieval Village": "medieval village miniature diorama",
    "Enchanted Forest": "enchanted forest miniature diorama",
    "Ruined City": "ruined city miniature diorama",
}


def preset_names() -> list[str]:
    return list(STYLE_PRESETS.keys())


def get_preset(name: str) -> StylePreset:
    return STYLE_PRESETS.get(name, STYLE_PRESETS[DEFAULT_PRESET])


def build_prompt(preset_name: str, custom_prompt: str | None) -> str:
    preset = get_preset(preset_name)
    custom = (custom_prompt or "").strip()
    if not custom:
        return preset.prompt
    return f"{preset.prompt}, {custom}"


def build_clip_prompt(preset_name: str, custom_prompt: str | None) -> str:
    style = get_preset(preset_name).clip_hint or CLIP_STYLE_HINTS.get(preset_name, "miniature diorama")
    custom = (custom_prompt or "").strip()
    custom_part = f", {custom[:90]}" if custom else ""
    return (
        f"{style}, preserve source composition, same camera, same horizon, "
        f"same foreground and background{custom_part}"
    )

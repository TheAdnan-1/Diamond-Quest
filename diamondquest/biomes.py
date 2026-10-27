"""The three cave themes. Each map picks one; colours, decorations, lights and music key change."""

BIOMES = {
    "jungle": {
        "name": "Jungle Temple",
        "blurb": "Moss-choked ruins of a forgotten temple",
        "ambient": (100, 104, 94),
        "fog": (10, 16, 12),
        "floor": ("#7d8270", "#9a9c86", "#3b4034"),     # stone a, stone b, grout
        "moss": ("#4f7a2c", "#7fae3c"),
        "dirt": ("#4a3424", "#6b4a30", "#2c1d14"),       # dark, light, deep
        "wall": ("#6b6552", "#857c63", "#2f2a21"),       # brick a, brick b, mortar
        "wall_front": ("#8a7d60", "#3d3428"),
        "rock": ("#7c7a6c", "#9d9a86"),
        "snake": ("#3f9a3a", "#d9d24a", "#1f4d22"),      # body, pattern, belly-dark
        "torch": (255, 170, 80),
        "crystal": None,
        "particles": "dust",
        "accent": (120, 200, 90),
        "key": "D",
    },
    "frost": {
        "name": "Frost Cavern",
        "blurb": "Shadowed slate passages beneath ancient ice",
        "ambient": (78, 87, 94),
        "fog": (7, 12, 18),
        "floor": ("#344955", "#405560", "#172730"),
        "moss": ("#e6f3ff", "#ffffff"),
        "dirt": ("#49473f", "#625e54", "#272a29"),
        "wall": ("#142530", "#243745", "#080f17"),
        "wall_front": ("#354754", "#111b25"),
        "rock": ("#7f8fa6", "#a9b8cc"),
        "snake": ("#d8eef7", "#3fb6c9", "#5d7a8c"),
        "torch": (116, 158, 179),
        "crystal": (110, 163, 186),
        "particles": "snow",
        "accent": (120, 220, 255),
        "key": "E",
    },
    "ember": {
        "name": "Ember Dungeon",
        "blurb": "A castle dungeon sinking into volcanic rock",
        "ambient": (106, 92, 86),
        "fog": (20, 8, 6),
        "floor": ("#5e5658", "#776d6c", "#241e1f"),
        "moss": ("#ff7a2a", "#ffc160"),
        "dirt": ("#463730", "#634e43", "#1e1513"),
        "wall": ("#3d3638", "#524a4c", "#140f10"),
        "wall_front": ("#6e6064", "#251e20"),
        "rock": ("#57504f", "#7a706c"),
        "snake": ("#d2471f", "#ffd34a", "#3a0d06"),
        "torch": (255, 150, 60),
        "crystal": None,
        "particles": "embers",
        "accent": (255, 120, 60),
        "key": "A",
    },
}

ORDER = ["jungle", "frost", "ember"]

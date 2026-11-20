# Expedition visual and audio sources

The human player is the spiky-haired boy in a blue gi, white sash and red wristbands,
wearing a circular cream **S** chest badge. The AI rival is the red-haired ponytail
adventurer with a cream top, brown shorts, gloves and boots, wearing a matching **A**
badge. Her front-facing walking, hammer poses, portrait and menu cutout face forward.
Neither carries the reference torch or staff. Walking, tools, HUD portraits and title
art all use this role assignment; the logical player IDs and AI rules are unchanged.

The character atlases and HD cutouts were created with the built-in image generator
using those references. Prompts specified consistent sculpted chibi game art, front,
side and back walking keys; hammer wind-up/strike/recovery; empty-palmed pushes; and
hook casts and pulls. Separate front/back hammer poses complete directional actions.
Badge orientation is preserved when left-facing sprites are prepared.

`diamondquest/art/explorer.png` and `automaton.png` keep the historical internal
filenames: `explorer` contains the girl and `automaton` contains the boy.
`CHARACTER_SOURCE` maps logical player 0 to the boy and AI 1 to the girl. Their corresponding `_actions`
sheets and `hammer_vertical.png` also use the new designs. `art/identity/` contains
references and HD menu/portrait cutouts. These source files have real alpha transparency.

`snake_bite.png` contains four authored strike keys for each biome. `snake_stun.png`
contains impact recoil, collapsed and recovering keys. `Production/bake_vfx.py`
prepares optical-flow inbetweens under `art/baked/`: 42 bite frames and 40 stun frames
per biome. OpenCV is only a build-time dependency, never a game dependency.
`burn_mantle.png` contains 16 ignition, rising-flame, ember and soot stages.

`cinematics.py` connects authored poses to recoil, weight shifts, a braided hook
tether, sparks, fragments, dust, venom and freezing. It keeps visual ghosts for
snakes that the rules immediately remove after biting. It never changes a logical
World. Guardian profile art faces its beam; the actual lens anchor is shared by
beam and lighting. Mines use flat code-native facets. Cell borders and a read-only
BFS walking-distance overlay remain available with G.

Character masters use independent HD sizes rather than the floor tile's texture
size. LOD selection preserves oversampling for characters and action poses, and
SDL uses its best supported texture filtering. Output is rendered at the window's
native resolution. This improves texture clarity; it does not create unlimited
source detail at arbitrary zoom.

## Sound production

`sound_design.py` supplies 36 sound families with 190 bundled stereo clips,
including material Foley, modal metal, stone and rope, ignition, fangs, venom,
crystal fractures, UI, local fire/beam loops and musical stings. Temporal layering,
early reflections, filtered tails and headroom keep effects readable under music.
The boy uses cloth and soft shoe Foley; the girl uses leather-boot Foley.

`Production/compose_action_thriller.py` produces the active 48-bar, 126–138 BPM
background score. The main stem includes the beat immediately: hybrid kick/snare,
shakers, a bright harmonic bass groove, layered pizzicato and short strings, a
recurring violin motif, restrained brass answers, tom fills and riser/crash builds.
The tension stem adds sixteenth-note strings and extra percussion. Both share exact
bar length. Music mastering targets -17 LUFS for the main stem and -22 LUFS for the
additive stem, with controlled peaks. The game retains its existing volume controls
and adaptive mix; sound-effect samples and all playback code are unchanged.

The themes are Danger in the Ruins (126 BPM), Icebound Pursuit (130 BPM) and Cinder
Run (138 BPM). MIDI/event sources are under `Production/music_source`. The game
plays the bundled Ogg files directly. `compose_score.py` supplies the shared sample
renderer. Earlier composition scripts remain as production history; the active
files are identified by `music_bank/index.json` version 5.
`Production/render_sound_bank.py` renders the full effects bank and sampled stings.
The TimGM6mb sample instrument source and its GPL-2 license/copyright notices are
included. See `Licenses/MUSIC_CREDITS.txt`. This notice concerns the sample source;
it does not assert a new license over the user's game code.

Noto Serif Bold and Open Sans are bundled with their font license notices. Art,
fonts, all music and all effects work offline. No API keys are needed.

## Frost Cavern

`frost_materials.png` contains four dedicated matte terrain swatches: a smooth
slate-blue open path, granular gray-brown diggable earth, near-black jagged wall
caps and heavy masonry faces. The renderer adds inset stone borders and a dark
wall foot to distinguish blocked volume independently of hue. Crystal lights,
ambient illumination and snow decorations are reduced. Surface hover labels and
the map legend identify open, diggable and solid terrain.

## Validation scope

The actual pygame/SDL renderer was checked under Linux software rendering: startup,
menus, a generated match, movement, cameras, map, all three biome scenes, both
characters' tools, burn, bite, stun/recovery, freeze, explosions, boulder impact,
mines and cell distances. The core rules, AI and map modules were compared byte for
byte with the original archive. Windows GPU performance has not been measured.

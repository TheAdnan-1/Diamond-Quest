# Diamond Quest — Expedition Edition

A turn-based diamond race on a 50×50 cave map: **you vs. an AI rival**.
Dig, collect violet and red diamonds, dodge snakes, fire and falling boulders, use tools to
block your rival, and reach the exit with the higher **points + life** total.

This edition replaces the visual presentation with authored character, environment and
prop atlases, directional character animation, animated vipers and layered fire effects.
The original game rules, AI, balance and map generation are preserved.
The action effects and entire sound-effects bank have been rebuilt for this edition.
All artwork, fonts and new sound effects are bundled; the game does not download assets while playing.

On Windows, extract the complete folder and double-click **Play Diamond Quest.bat**.
See **START_HERE.txt** for the short setup guide. Press **C** to switch cameras;
the initial view follows your character so the artwork stays readable.
You control the blue-gi boy wearing **S**; the red-haired girl wearing **A** is the AI rival. Press **G** to
show/hide cell borders. Hover a cell to see its coordinates, shortest walking-move
count, numbered route and whether that route crosses a hazard.

## Install and run

Requires Python 3.9+ (64-bit).

```bash
pip install -r requirements.txt        # pygame + numpy
python main.py                          # or: python main.py 1234   (a specific map seed)
```

**First launch** prepares and caches display textures in `~/.diamond_quest/cache`.
All music, effects and animation frames are already bundled. Preparation time depends
on your CPU and screen resolution; later launches reuse the texture cache.

## Cave design

Each map is a network of chambers carved out of solid rock:
- **Vaults:** all 12 red diamonds sit in small sealed vaults off the chambers. Each vault
  is closed by a **cracked wall** (Hammer) or a **boulder plug** (Push it in, or Hook it out).
  You can't reach a red diamond without paying for it.
- **Cracked walls on the paths:** most tunnels between chambers are barred by one or two
  cracked walls, so you'll swing the hammer just to get around.
- **Rock gates:** some tunnels are blocked by a boulder sitting between two pits. Push (or
  hook) it sideways so it drops into a pit, and the way opens. There is no way round.
- **Lens statues:** stone idols two cells tall, a head with a glowing lens eye on top of a
  body. Some guard tunnels from the side wall; others stand against a chamber wall and sweep
  a whole row. See the rules below.
- **Boulders on soil:** dig underneath one and it falls, so don't rush.
- **Fair, safe starts:** nothing in either start chamber can hurt you (no fire, snakes or
  lens beams), and from both starts you can walk out to a good share of the diamonds without
  taking damage you can't avoid (fire, or a beam you can't hook away from your side). Both
  starts get a similar share.
- Every map is checked: every diamond and the exit can be reached with tools; red diamonds
  can't be reached without them; and from each start, at least half the diamonds need the
  hammer and at least 10 need a push. The setup screen shows these counts.

## Map

The minimap (bottom right) shows real diamond icons (red ones pulse), the exit portal (with
a lock until you hold 10 diamonds), snakes, cracked walls, fire, boulders, both explorers
and what's on screen. Press **M** for a large map with a legend. The camera keeps both
explorers out from under the minimap (and you can hide it with **Tab**). When the exit or the AI is
off screen, an arrow at the screen edge points to it with the distance.

## Graphics

- **Native resolution, GPU rendered.** The game draws at your monitor's real pixel count
  (1080p, 1440p, 4K, 8K…) through SDL's hardware renderer. It is DPI-aware on Windows, so
  nothing is stretched or blurred. Textures are painted at the tile size your screen needs
  (up to 384 px per tile), with half and quarter versions so zooming out stays clean.
- **Three cave themes** picked per map: *Jungle Temple* (moss, vines, ferns, torches),
  *Frost Cavern* (ice, icicles, glowing crystals, snow) and *Ember Dungeon* (basalt bricks,
  chains, glowing lava cracks).
- **Real lighting:** each frame a light map is drawn and multiplied into the scene. Torches,
  fires, crystals, lava, gems, the portal and impact flashes light their surroundings. On GPUs that support it, lights can also
  brighten above the base colours ("overbright"), which makes them glow.
- **Authored fantasy art:** a spiky-haired blue-gi player and a red-haired ponytail AI rival, each with front,
  back and side animation frames, separate tool poses and HD portraits. The boy's
  cream chest badge reads **S**, including mirrored left-facing poses; the girl's
  matching badge reads **A**. Her front poses and portraits look straight forward. There is no
  torch or staff in either character's hands. Highly faceted gems; sculpted boulders, illustrated
  tools, three themed vipers and guardian idols, carved portal gates and natural foliage.
- **Animated effects:** 24 fire frames, smoke and rising sparks, subtle looping snake
  deformation, authored hammer/hook/push actions, collection trails, impact flashes,
  ice and debris. Effects remain tied to the original event timeline.
- **Coherent interface:** bundled display and body fonts, bronze and teal menus,
  clear player cards, illustrated tools and a painted map overview.
- **Presentation only:** `world.py`, `ai.py`, `mapgen.py`, `config.py`, the search/planning
  modules are byte-for-byte identical to the supplied project. Audio changes are explicitly
  limited to sound design, music, playback and ambience.

## Action and impact update

- **Hammer:** authored wind-up, lift, strike and recovery poses for both characters in
  all four directions, with a material impact, fragments, dust and timed contact audio.
- **Push:** a braced character pose, weight shift, rolling stone, scrape and floor dust.
- **Hook:** a visible cast, forged hook head, curved braided tether, catch and tensioned
  return, synchronized with the character's pull pose and the target's movement.
- **Statues:** profile art faces the beam direction. The beam starts at the actual lens;
  a front view bridges turns, and the beam pauses while the statue turns.
- **Hazards:** authored fang strikes, attached burn flames and sparks, a 24-frame
  explosion followed by smoke, prismatic freeze/thaw effects, and heavier boulder impact
  with head-level chips, floor dust, fractures, recoil and a settling stone.
- **Mines:** compact flat top-down art with a faceted ruby, without a thick 3D side wall.
- **Cells:** subtle borders, a hover outline and numbered walking routes. Distances count
  walking moves in the current map; hazards can add damage or delays. Tools and future
  world changes are not included in the walking count.

## Sound

All 34 original effect categories have new sound designs. The edition includes **190
stereo samples across 36 families**, including local fire and beam loops, plus three
cave ambience beds. The background music now uses original 48-bar action-thriller
themes at 126–138 BPM: a driving drum groove, tight harmonic bass, rapid string
ostinatos, a recurring violin motif, low brass accents, tribal tom fills and
transition builds. The exploration stem carries the full groove from its opening;
the synchronized tension stem adds extra percussion and strings near hazards.

- Material layers distinguish leather boots, soft martial-arts shoes, dirt, stone, metal and rope.
- Hammer contact, hook catch and pull, snake bites, fire, explosions, freeze and boulder
  impacts follow the visual event timeline.
- Gems, portals, UI hover/click/error, tool selection, mines, death and victory/defeat
  cues are also replaced. Repeated actions choose from multiple variations.
- New effects ship as 44.1 kHz stereo PCM with controlled peak headroom. Local loops sit
  below transient effects; major impacts briefly lower music level.
- Sound-effects mute stops ongoing effect voices and local loops immediately.
- Material effects use deterministic layered synthesis in `sound_design.py`; musical
  stings and the new score use sample instruments. Build scripts, score MIDI, sample
  source and license notices are in `Production/` and `Licenses/`. No runtime sound
  downloads, composition tools or API keys are required. Music and effects retain the
  existing volume controls.
- Tracks: *Danger in the Ruins* (Jungle, 126 BPM / 91 seconds), *Icebound Pursuit*
  (Frost, 130 BPM / 89 seconds), and *Cinder Run* (Ember, 138 BPM / 83 seconds).
  Each has two synchronized stems.
- This revision changes the background music only. All sound-effect samples,
  ambience files, sound-design code and playback settings are byte-for-byte unchanged.
  `Previews/Diamond_Quest_Thriller_Music.mp3` is a music-only sampler of all three themes.

## Frost Cavern readability

- Open paths use smooth matte slate-blue stone. Solid blocked cells use dark raised
  rock with inset borders and heavy masonry faces. Diggable ground uses granular
  gray-brown frozen sediment. The three surfaces also differ in texture and shape.
- The ambient light, crystal lights, snow glare and local lighting are reduced.
- Hover labels identify the surface and movement distance. The large map legend
  includes open paths, diggable ground and solid blocked rock.

## Character and hazard refinement

- Independent 384-unit character masters, 512px action canvases, 384px portraits and
  1024px menu art retain detail when the camera zooms. Character LODs preserve at least
  two texture samples per display pixel where the source resolution permits.
- Viper bites use 42 baked motion-inbetween frames per biome, an anticipatory hiss,
  a fast visible fang contact, venom droplets, player recoil and recovery.
- Snake stun uses impact recoil, a slumped closed-eye posture, quiet breathing, an
  engraved ground sigil and a separate recovery sequence. The ownership label remains.
- Burn damage uses a 16-stage flame mantle drawn behind and in front of the actor,
  shake-off motion, rising embers, soot and a synchronized ignition-to-ember sound.
- All of these offsets and timelines are visual only. The game's snake removal, stun
  duration, turn order and damage remain unchanged.

## Rules

| | |
|---|---|
| Violet diamond | +10 points |
| Red diamond | +25 points |
| Starting life | 100 (0 = instant loss) |
| Snake bite | −15 life (the snake dies after biting) |
| Fire | −10 life when you step in, and −10 again every turn you stay. Fire never goes out and can't be hammered or hooked |
| Boulder hit | −40 life |
| Hammer | −5 life: stun a snake or break a cracked wall |
| Push | −8 life: shove an adjacent rock sideways |
| Hook | −6 life: pull a rock or snake up to 3 cells away (sideways, clear line) next to you |
| Lens beam | −40 life and you're frozen for your next 2 turns |
| Mine (optional) | each player has 3. Place one on a free cell next to you (it uses your tool for the turn, costs no life). Your rival loses 25 life when they step on it; your own mines don't hurt you. A rock landing on a mine sets it off |
| Exit | opens for a player holding 10 diamonds; whoever exits ends the game |

- Each turn: optionally **one tool** (it costs life, not a turn), then **move one cell**
  (no diagonals) or **wait**. Both players may share a cell.
- A stunned snake bites no one. The player who hammered it may stand in its cell until
  their next move, and must leave then. Anyone still in the cell when the stun ends is bitten,
  unless they hammered it themselves.
- **Lens statues:** the eye (top cell) shines a beam straight ahead along its row until a
  wall, cracked wall, soil, boulder or another statue stops it. Step into it, or let it swing
  onto you, and you freeze. Break a wall or dig the soil in its line and the beam reaches
  further. Hook the statue by its **body** (bottom cell), standing in the body's row so the
  beam can't touch you: it slides one cell toward you (or turns in place when it's next to
  you or has no room) and turns to face the other way. Its head blocks the hook.
- **Gravity is instant.** A boulder that loses what holds it up drops at once, as far as it
  goes (rolling off other boulders). Push or hook a rock over a gap and it's gone before you
  move, so you can walk on in the same turn. Dig out the soil a rock sits on (or take the
  diamond, or let the snake under it die) and it comes down on you: −40, and it rests on your
  head until you step out from under it. Don't step straight down from under it! Drop a rock
  onto your rival the same way.
- The game ends when someone exits, someone runs out of life, or every diamond has been
  collected. The highest points + life wins.
- The AI heads for the exit as soon as leaving would win, instead of hunting more diamonds.

## How the AI plays

The rival is built to be hard to beat. Every turn it:
1. plans a full route through the diamonds (beam search), leaving out the ones you will
   reach first;
2. checks the next step with minimax or MCTS, playing each line out to the end of the game
   (who reaches 10 diamonds first, who exits when, who steals whose diamond);
3. values life and points equally, so it uses tools whenever they save enough time;
4. walks out the moment leaving wins, and races you when you're the one who could leave;
5. mines your route and blocks you with rocks when that costs you more than it costs it;
   walks round to a statue's body and turns it when that clears its way, and turns one onto
   you when you're standing in its new line;
6. never digs under a boulder that would seal off its own way out, and never takes a route
   whose mines, fire and beams would add up to its own death;
7. goes for the knockout: every turn it checks whether a push or hook would drop a rock or
   swing a lens beam onto you, and does it when that hurts you more than it costs (or ends
   the game). It mines strategically: it predicts your next diamonds and your way to the exit
   and drops a mine on any chokepoint it passes where that forces a real detour on you (or a
   25-life toll where there is no way round);
8. when it is losing and you are about to open your exit, it races to the exit and mines the
   cells you must cross to leave (its own mines never hurt it).

It thinks for about a second per move.

You can watch it think with **B**.
- Mines are switched on or off in the setup screen.

All numbers live in `diamondquest/config.py`.

## Controls

| Key | Action |
|---|---|
| Arrows / WASD, or click a neighbouring cell | Move |
| H / P / K, then a direction | Hammer / Push / Hook |
| E, then a direction | Place a mine |
| Space | Wait |
| T or Enter | Type a command in English, e.g. `hammer left then go down twice` |
| L | Danger Sense (logic engine overlay) |
| V, then 1–7 (0 = compare all) | Algorithm Lab: watch each search algorithm explore |
| B | AI brain inspector (goal, minimax/MCTS stats, fuzzy memberships) |
| O | PDDL plan to the nearest diamond (files saved to `diamond_quest_output/`) |
| C | Camera: both players on screen / follow you / follow the AI |
| M | Big map |
| Tab (or the − on the minimap) | Hide / show the minimap (remembered between games) |
| N / X | Music on/off / sound effects on/off |
| F11 or Alt+Enter | Fullscreen |
| F3 | Frame-rate counter |
| Esc | Pause menu (settings, restart, quit) |

## Where each AI topic lives

| Topic | File | Role in the game |
|---|---|---|
| DFS, DLS, IDDFS, Bidirectional search | `search.py` | Algorithm Lab: animated search from you to a red diamond |
| Best-first, Greedy best-first, A*, heuristic function | `search.py` | A* drives the AI's movement; Manhattan distance heuristic; all shown in the Lab |
| Adversarial search, utility function, Minimax (alpha-beta) | `adversarial.py` | AI strategy: iterative-deepening alpha-beta over "go to diamond X / exit" moves; every leaf is played out to the end of the game, so the utility is the predicted final score difference |
| Monte Carlo Tree Search | `adversarial.py` | Alternative AI brain (choose "MCTS" in setup): UCT with full-game playouts |
| Route planning (beam search) | `adversarial.py` | The AI plans its whole diamond tour (an orienteering problem), skipping diamonds you'll reach first, and sticks to it unless something clearly better appears |
| Fuzzy inference: Mamdani, membership functions, fuzzification, defuzzification | `fuzzy.py` | AI risk appetite from its life and score lead |
| Constraint Satisfaction Problem | `csp.py`, `mapgen.py` | Fair map generation: start points, exit and red diamonds under 93 constraints |
| Inference engine for propositional logic | `logic.py` | Forward/backward chaining over Horn clauses: dangerous cells for the AI and Danger Sense |
| PDDL | `pddl/domain.pddl`, `pddl_planner.py` | Domain file, problem generator, parser, grounder and BFS planner |
| Parsing in NLP | `nlp.py` | Tokenizer + context-free grammar + recursive-descent parser for typed commands |

## Project layout

```
main.py                    entry point
diamondquest/
  config.py                balance numbers
  world.py                 rules engine (turns, tools, snakes, fire, gravity, scoring)
  mapgen.py, csp.py        fair random maps
  search.py, adversarial.py, fuzzy.py, logic.py, pddl_planner.py, nlp.py, ai.py   the AI
  biomes.py                cave themes
  paint.py                 procedural painting toolkit (noise, SDF shapes, lighting)
  art_terrain.py           floors, soil, walls, decorations
  art_objects.py           original procedural effects and fallback object helpers
  art_director.py          authored atlas sampling, character and creature frames, materials
  cartography.py           painted map overview, read-only world access
  cinematics.py            action, hazard, beam and impact choreography
  fx_art.py                action/effect atlas sampling and effect materials
  sound_design.py          new material-based sound synthesis
  sound_bank/              bundled stereo effects and ambience
  art/                     bundled PNG source atlases
  fonts/                   bundled fonts and licenses
  assets.py                parallel texture generation, disk cache, LODs
  gpu.py                   GPU renderer wrapper, UI primitives
  view.py                  world renderer, lighting, effects, camera
  hud.py, app.py           interface, screens, game loop
  audio.py                 synthesizers, effects, adaptive music, ambience
```

## Troubleshooting

- **Slow first start:** that's the one-time painting and composing; later launches use the cache.
- **Low frame rate:** check that your graphics drivers are installed (the Settings screen
  says "GPU accelerated" or "software rendering"). Press F3 to see the frame rate.
- **No sound:** check your system volume, then press N / X in game, or adjust the sliders
  in Settings.

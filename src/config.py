from llm_models import Providers, GroqModels, GoogleModels

number_of_enemies_per_bundle = 10
number_of_weapons_per_bundle = 10
number_of_levels_per_bundle = 4
provider_key = Providers.GROQ
model_key = GroqModels.OPENAI_GPT_OSS_120B

################################################################################
# Maps Description for teste
################################################################################

# The Cursed Dwarven Forge (Theme: Medieval/Fantasy)
prompts = [
    """
The map represents an ancient, abandoned dwarven stronghold buried deep beneath
a mountain, now overrun by goblins and ancient machinery. The architecture
should feature sharp, geometric angles with vast rectangular halls supported by
thick stone pillars, connected by long, rigid corridors carved directly into the
rock. The environment is dominated by elements of heat and stone, with rivers of
molten lava cutting through the floor layout, creating natural barriers that
force the player to find bridges or alternative routes.

The layout must include distinct functional areas such as massive foundries
containing anvils, cramped storage rooms filled with rusted weapons, and grand
throne rooms adorned with gold statues. The atmosphere is oppressive and
claustrophobic, yet majestic. While the main paths are paved and wide, there
should be secret, narrow mining tunnels that bypass main security doors, adding
a layer of non-linear exploration to the dungeon generation.
""",
    # The Derelict Starship (Theme: Sci-Fi/Space)
    """
The setting is a drifting, power-critical industrial spaceship lost in deep
space, infested with rogue security drones. The map structure is strictly
metallic and modular, consisting of interconnected sectors separated by heavy
blast doors and airlocks. The layout should reflect a functional design,
featuring clean, sterile medical bays, chaotic and cluttered cargo holds full of
shipping containers, and tight, maze-like maintenance crawlspaces that run
underneath the main flooring.

Lighting and hazards play a major role in the topology; some sections of the
ship have hull breaches exposing the vacuum of space, requiring the generation
of specific pathways to avoid suffocation. The generator should prioritize a
grid-based interior design with electrical conduits lining the walls, server
rooms with rows of data banks, and a central command bridge located at the
furthest point from the entry airlock, acting as the boss room.
""",
    # The Smuggler’s Grotto (Theme: Pirate/Nautical)
    """
This map depicts a hidden pirate cove located inside a massive, partially
flooded sea cave system. The terrain is a mix of natural wet rock formations and
man-made wooden structures. Rickety wooden walkways and suspension bridges
connect isolated rock islands over dark, deep water, creating a verticality
where falling means fighting aquatic monsters. The walls are irregular and damp,
covered in moss and glowing bioluminescent fungi.

Scattered throughout the cavern are wrecks of crashed ships that have been
repurposed into buildings, serving as shops or enemy barracks. The layout should
feel organic and somewhat chaotic, with secret alcoves hidden behind waterfalls
and large open areas where the underground sea meets a sandy shore. The
generation needs to balance dry land for combat and water tiles that hinder
movement, evoking the feeling of a treacherous hideout.
""",
    ## Neon Skyline Penthouse (Theme: Cyberpunk/Urban)
    """
The environment is the top floor of a futuristic mega-corporation tower in a
rainy, neon-lit metropolis. The map requires a sleek, modern architectural style
characterized by glass walls, chrome surfaces, and open-plan offices separated
only by holographic dividers. The aesthetic is high-tech and clean, but the
layout becomes complex due to security grids, laser barriers, and glass mazes
that confuse the line of sight while allowing visibility of enemies.

The level design should feature a stark contrast between the luxurious executive
suites—carpeted and spacious—and the sterile, cold server rooms that hum with
energy. Exterior balconies wrapping around the building offer risky shortcuts,
exposing the player to rain and sniper fire from adjacent skyscrapers. The
prompt demands a layout that emphasizes cover-based tactical positioning amidst
polished furniture and glowing computer terminals.
""",
    ## The Living Hive (Theme: Alien/Bio-Horror)
    """
The map takes place inside the belly of a colossal, planet-sized alien organism.
There are no straight lines or man-made materials here; everything is
biological, fleshy, and pulsating. The walls are made of muscle tissue and bone,
and the "doors" are sphincter-like valves that open when approached. The
corridors are tube-like veins that twist and turn unpredictably, leading to
spherical chambers filled with incubation eggs and acidic pools.

The generation must focus on organic, cellular shapes rather than rectangular
rooms. The ground is sticky and uneven, covered in creeping creep and slime that
slows movement. The layout should feel like a digestive system, with hazards
such as spikes erupting from the floor or walls that constrict the player. The
goal is to create a map that feels alive and hostile, completely rejecting
traditional architectural symmetry.
""",
]

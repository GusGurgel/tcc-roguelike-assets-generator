from typing import Type, TypeVar, List
from pydantic import BaseModel, ValidationError
from os.path import join
from math import floor
import time
import json

from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

from utils import *
from llm_models import get_model, Providers, GroqModels, GoogleModels
from models import *  # type: ignore
from vector_db import query_vector_store, StoreType, query_by_tileset_position
from db import *
from config import *


T = TypeVar("T", bound=BaseModel)

save_path = join(MAIN_PATH, "saves")

# Constante de fortificação para garantir a saída estruturada
JSON_CONSTRAINT_PROMPT = """
**CRITICAL JSON REQUIREMENT:**
You MUST output ONLY a valid JSON object that strictly adheres to the requested schema.
Do NOT include any conversational text, explanations, or markdown code blocks (like ```json) outside the pure JSON structure.
Your entire response must be parseable by a standard JSON parser.
"""


class AssetsGenerator:

    def __init__(self, theme_description) -> None:
        # A inicialização usa variáveis globais provider_key e model_key que podem ser injetadas
        self.model = get_model(provider_key, model_key)
        self.usage_callback = UsageMetadataCallbackHandler()

        # O prompt agora atua como um "Lead Game Designer" criando a documentação base.
        response = self.model.invoke(
            [
                HumanMessage(
                    f"""
Act as a Lead Game Designer and World Builder.
You are tasked with expanding a basic concept into a rich Roguelike Game Setting.

Input Concept: "{theme_description}"

**IMPORTANT LANGUAGE CONSTRAINT:**
Regardless of the language used in the "Input Concept", **you must write the expanded description entirely in English.**

Your goal is to write a cohesive "World Description" that will serve as the source of truth for generating assets later.
Please define:
1. **The Setting Name**: Give this world or dungeon a unique name.
2. **Visual Style & Atmosphere**: Describe the art direction (e.g., Gritty Industrial, Neon-Cyberpunk, Gothic Horror) and the mood (lighting, weather, ambient noise).
3. **The Lore**: Briefly explain the history. Why is this place dangerous? What happened here?
4. **The Core Conflict**: What is the corruption, curse, or enemy force controlling this place?

Do not list specific item stats yet. Focus on the narrative and sensory details to guide future generation.
"""
                )
            ],
            config={"callbacks": [self.usage_callback]},
        )

        self.raw_theme_description = theme_description
        self.theme_description: str = str(response.content)

    def _get_structured_model(self, schema_class: Type[T]):
        return self.model.with_structured_output(
            schema=schema_class.model_json_schema(), method="json_schema"
        )

    def _ask_llm_structured(self, schema_class: Type[T], messages: list) -> T:
        structured_llm = self._get_structured_model(schema_class)

        last_exception = None
        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            try:
                # Tenta invocar o modelo
                result = structured_llm.invoke(
                    messages,
                    config={"callbacks": [self.usage_callback]},
                )

                # Tenta validar o resultado com o Pydantic
                # Se o result já vier como dict (comum em structured output), o validate converte
                return schema_class.model_validate(result)

            except (ValidationError, ValueError, TypeError) as e:
                # Captura erros de validação do Pydantic ou erros de tipo
                last_exception = e
                print(
                    f"Tentativa {attempt}/{max_attempts} falhou ao validar o esquema. Erro: {e}"
                )
                # O loop continua para a próxima iteração automaticamente
            except Exception as e:
                # Captura outros erros inesperados (ex: erro de conexão com a API)
                last_exception = e
                print(f"Erro inesperado na tentativa {attempt}: {e}")

        # Se sair do loop, significa que falhou 5 vezes
        print("Todas as 5 tentativas de gerar a saída estruturada falharam.")
        if last_exception:
            raise last_exception
        else:
            raise Exception(
                "Falha desconhecida na geração estruturada após 5 tentativas."
            )

    def generate_player(self) -> Player:
        return self._ask_llm_structured(
            Player,
            [
                HumanMessage(
                    f"""
Act as a Narrative Designer and RPG System Creator.
Based on the rich world description below:
"{self.theme_description}"

Create the Main Protagonist (The Player Character) for this Roguelike.
Guidelines:
1. **Archetype**: Define a starting class/role that fits the setting (e.g., A Disgraced Knight, A Glitchy Android, A Cursed Cultist).
2. **Backstory & Motivation**: Why is this character entering this dangerous place? (Redemption, Greed, Revenge, Survival?).
3. **Visuals**: Describe their appearance, starting gear look, and distinct physical traits matching the visual style of the theme.

Generate the Player profile now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

    def generate_final_objective(self) -> FinalObjective:
        return self._ask_llm_structured(
            FinalObjective,
            [
                HumanMessage(
                    f"""
Act as a Lead Narrative Designer and Lore Architect.
Context: We are designing the climax of a Roguelike game based on this world setting:
"{self.theme_description}"

Your task is to design the **Final Objective**.
This is the ultimate item located at the very bottom of the dungeon that the player must retrieve and physically carry back to the entrance to win.

Guidelines:
1. **Type of Object**: The object must be tangible and portable (e.g., an artifact, a sacred scroll, a severed head, a crystallized soul). It cannot be a location or a giant structure.
2. **High Stakes Lore**: In the `back_history`, connect this object directly to the "Core Conflict" of the theme. Is it the source of the dungeon's corruption? The only cure for a plague? The key to a sealed god?
3. **Visual Distinction**: The `tile` description should sound legendary. It should visually stand out from regular loot.
4. **Motivation**: Clearly state why leaving it down there is not an option.

Generate the legendary Final Objective now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

    def generate_dungeon_levels(self) -> DungeonLevelList:
        return self._ask_llm_structured(
            DungeonLevelList,
            [
                HumanMessage(
                    f"""
Act as an expert Roguelike Level Designer. 
Your task is to generate a progression of {number_of_levels_per_bundle} dungeon levels based on the following theme:
"{self.theme_description}"

Guidelines for generation:
1. **Progression**: The levels must evolve. Depth 1 should be the easier, while Depth {number_of_levels_per_bundle} is the most dangerous.
2. **Atmosphere**: For each level, describe the environment, lighting, smells, and ambient sounds.
3. **Cohesion**: Ensure the transition between levels makes logical sense within the theme.
4. **Variety**: Avoid repeating the exact same descriptions.

Generate the {number_of_levels_per_bundle} levels now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

    def generate_weapons(self) -> WeaponList:
        return self._ask_llm_structured(
            WeaponList,
            [
                HumanMessage(
                    f"""
Act as a Creative Director for a Roguelike game.
Based on the theme specification: "{self.theme_description}"

Generate {number_of_weapons_per_bundle} unique weapons. 
Guidelines:
1. **Thematic Fit**: All weapons must strictly fit the technology/magic level and tone of the theme.
2. **Diversity**: Include a broad mix of types:
   - Melee (Daggers, Swords, Hammers, Polearms).
   - Ranged (Bows, Guns, Crossbows, Thrown).
   - Magic/Tech (Staffs, Wands, Experimental Devices).
3. **Rarity Spread (Balanced Economy)**:
   - 50% **Common weapons**: Rusty, improvised, or basic standard issue gear.
   - 30% **Rare weapons**: Specialized, high quality, superior craftsmanship, or enchanted.
   - 20% **Legendary weapons**: Artifacts, experimental prototypes, or named weapons with lore and unique properties.
4. **Descriptions**: Provide vivid descriptions focusing on the weapon's appearance and the specific "feeling" of wielding it.
5. **Range of Fields**: The fields rarity, weight and mana_cost need to be in the range [0, 10] (inclusive).

Generate the list of {number_of_weapons_per_bundle} weapons now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

    def generate_enemies(self) -> EnemyList:
        return self._ask_llm_structured(
            EnemyList,
            [
                HumanMessage(
                    f"""
Act as a Gameplay Balance Designer.
Using the theme: "{self.theme_description}"

Generate a Bestiary of {number_of_enemies_per_bundle} unique enemies distributed across the dungeon depths.
Guidelines:
1. **Archetypes**: Ensure a mix of:
   - *Skinny*: Weak, but really fast.
   - *Tanks*: Slow, high health.
2. **Variety**: Ensure variety on the thread of the enemies. Should exist 50% enemies with thread 1~5, 30% with thread 5~8 and 20% with thread 9~10
3. **Visuals**: Describe their appearance to match the gloomy/adventurous tone of the theme.
4. **Range of Fields**: The fields thread and weight need to be in the range [0, 10] (inclusive). 

Generate the {number_of_enemies_per_bundle} enemies now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

    def generate_asset_bundle(self) -> AssetBundle:
        start_time = time.time()

        asset_buddle_base = self._ask_llm_structured(
            AssetBundleBase,
            [
                HumanMessage(
                    f"""
Act as a Creative Director and Marketing Lead for a Game Studio.
Analyze the rich world description provided below:
"{self.theme_description}"

Your task is to craft the perfect Title (Name) for this Roguelike Asset Bundle.

Guidelines:
1. **Impact**: The name must be catchy, evocative, and marketable (e.g., "Echoes of the Void", "Neon Chrome", "The Iron Oath").
2. **Relevance**: It should capture the core of the theme.
3. **Format**: Use Title Case. Keep it concise (2 to 6 words). Avoid generic names like "Dungeon Pack 1".

Generate the title now.
{JSON_CONSTRAINT_PROMPT}
"""
                )
            ],
        )

        player = self.generate_player()
        dungeon_levels = self.generate_dungeon_levels()
        enemies = self.generate_enemies()
        weapons = self.generate_weapons()
        final_objective = self.generate_final_objective()

        player_with_texture = PlayerWithTexture(
            **player.model_dump(),
            tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                player.tile, "entities"
            ),
        )

        final_objective_with_texture = FinalObjectiveWithTexture(
            **final_objective.model_dump(),
            tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                final_objective.tile, "items"
            ),
        )

        dungeon_levels_with_texture_items: List[DungeonLevelWithTexture] = []
        for dungeon_level in dungeon_levels.items:
            dungeon_levels_with_texture_items.append(
                DungeonLevelWithTexture(
                    **dungeon_level.model_dump(),
                    wall_tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                        dungeon_level.wall_tile, "environments"
                    ),
                    floor_tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                        dungeon_level.floor_tile, "environments"
                    ),
                )
            )
        dungeon_levels_with_texture = DungeonLevelWithTextureList(
            items=dungeon_levels_with_texture_items
        )

        enemies_with_texture_items: List[EnemyWithTexture] = []
        for enemy in enemies.items:
            enemies_with_texture_items.append(
                EnemyWithTexture(
                    **enemy.model_dump(),
                    tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                        enemy.tile, "entities"
                    ),
                )
            )
        enemies_with_texture = EnemyWithTextureList(items=enemies_with_texture_items)

        weapons_with_texture_items: List[WeaponWithTexture] = []
        for weapon in weapons.items:
            weapons_with_texture_items.append(
                WeaponWithTexture(
                    **weapon.model_dump(),
                    tile_with_texture=AssetsGenerator.convert_tile_to_tile_with_texture(
                        weapon.tile, "items"
                    ),
                )
            )
        weapons_with_texture = WeaponWithTextureList(items=weapons_with_texture_items)

        total_time = time.time() - start_time

        return AssetBundle(
            **asset_buddle_base.model_dump(),
            raw_description=self.raw_theme_description,
            description=self.theme_description,
            player=player_with_texture,
            dungeon_levels=dungeon_levels_with_texture,
            enemies=enemies_with_texture,
            weapons=weapons_with_texture,
            final_objective=final_objective_with_texture,
            usage_metadata=self.usage_callback.usage_metadata,
            generation_time_seconds=floor(total_time),
        )

    @staticmethod
    def convert_tile_to_tile_with_texture(
        tile: Tile, store_type: StoreType
    ) -> TileWithTexture:
        texture_from_rag = query_vector_store(tile.description, store_type, 1)[0]

        position = Position(x=texture_from_rag["x"], y=texture_from_rag["y"])

        texture = Texture(
            tileset_position=position,
            tileset_description=texture_from_rag["description"],
        )

        return TileWithTexture(**tile.model_dump(), texture=texture)


def load_zombie_souls_asset_bundle() -> AssetBundle:
    return load_object_json(
        join(MAIN_PATH, "saves/", "zombie_asset_bundle.json"), AssetBundle
    )


if __name__ == "__main__":
    # Variáveis injetadas simulando um teste unitário local
    provider_key = Providers.GROQ
    model_key = GroqModels.OPENAI_GPT_OSS_20B
    prompt_index = 0

    asset_generator = AssetsGenerator(prompts[prompt_index])

    asset_bundle: AssetBundle = asset_generator.generate_asset_bundle()

    name = f"p_{prompt_index+1}_{model_key.replace('-', '_').replace('/', '_').lower().strip()}"

    asset_bundle.name = name

    with open(
        join(MAIN_PATH, "tests", f"{name}_bundle.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(asset_bundle.model_dump_json())

    insert_asset_bundle(asset_bundle, model_key)
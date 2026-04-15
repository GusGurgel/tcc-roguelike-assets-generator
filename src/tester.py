import os
import json
import uuid
import time

from os.path import join
from typing import List, Dict, Any, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError
from langchain.messages import HumanMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

# Importações dos seus módulos locais
from utils import MAIN_PATH
from llm_models import get_model, Providers, GroqModels, NvidiaModels
from vector_db import get_cosine_similarity

# Importação do gerador e injeção (patching) de globais
import asset_generator
from asset_generator import AssetsGenerator


T = TypeVar("T", bound=BaseModel)

# --- Pydantic Models para Saída Estruturada do Modelo Juiz ---

class CoherenceEvaluation(BaseModel):
    score: int = Field(
        description="A score from 0 to 100 quantifying the alignment, visual similarity, and thematic consistency between the textual intent and the generated game artifacts."
    )


class Evaluator:
    """
    Classe responsável por aplicar o modelo Juiz e garantir que os outputs 
    estruturados sejam validados corretamente via Pydantic com lógica de retentativas.
    """
    def __init__(self, provider, model_name) -> None:
        self.model = get_model(provider, model_name)
        self.usage_callback = UsageMetadataCallbackHandler()

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


def remove_texture_fields(data: Any) -> Any:
    """
    Função recursiva para remover todas as chaves que terminam com '_with_texture'.
    Isso elimina as duplicatas geradas pelas classes WithTexture (que já contêm a classe base),
    reduzindo o tamanho do JSON e economizando tokens no LLM Juiz.
    """
    if isinstance(data, dict):
        return {
            k: remove_texture_fields(v)
            for k, v in data.items()
            if not k.endswith('_with_texture')
        }
    elif isinstance(data, list):
        return [remove_texture_fields(item) for item in data]
    else:
        return data


def run_evaluation_pipeline(
    tested_provider, 
    tested_model_name: str, 
    judge_provider, 
    judge_model_name: str, 
    test_inputs: List[Dict[str, str]]
) -> List[Dict]:
    """
    Função principal de teste que recebe a LLM testada, a LLM juíza e um vetor de inputs.
    Realiza o teste de fidelidade semântica e fidelidade de reconstrução.
    """
    
    # Prepara a pasta de destino
    tests_dir = join(MAIN_PATH, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    
    # Instancia a classe Avaliadora que usa as funções seguras
    print(f"Instanciando Judge Model via Evaluator: {judge_model_name}...")
    evaluator = Evaluator(judge_provider, judge_model_name)
    
    results = []
    
    for t_input in test_inputs:
        prompt_text = t_input.get("prompt", "")
        prompt_name = t_input.get("prompt_name", "unnamed_prompt")
        prompt_index = t_input.get("prompt_index", 0)
        
        print(f"\n[{prompt_name}] Iniciando teste com o Tested Model: {tested_model_name}...")
        
        # 1. Injeta os parâmetros de LLM testada diretamente no módulo asset_generator
        asset_generator.provider_key = tested_provider
        asset_generator.model_key = tested_model_name
        
        # 2. Geração do Asset Bundle
        try:
            generator = AssetsGenerator(prompt_text)
            bundle = generator.generate_asset_bundle()
        except Exception as e:
            print(f"[{prompt_name}] Falha ao gerar asset bundle com o Tested Model. Erro: {e}")
            continue

        # Extrai a descrição refinada e isolada gerada pelo modelo testado
        refined_description = bundle.description
        
        # 3. Prepara o JSON do Asset Bundle para o Juiz
        bundle_dict = bundle.model_dump()
        
        # Removemos as descrições originais para o teste cego da reconstrução
        bundle_dict.pop("description", None)
        bundle_dict.pop("raw_description", None)
        
        # Limpa os campos de textura duplicados para economizar tokens
        clean_bundle_dict = remove_texture_fields(bundle_dict)
        
        stripped_bundle_json = json.dumps(clean_bundle_dict, indent=2)
        
        coherence_score = -1
        reconstruction_score = -1.0
        
        # ------------------------------------------------------------
        # MÉTRICA 1: Fidelidade Semântica (Coherence Metric)
        # ------------------------------------------------------------
        print(f"[{prompt_name}] Avaliando Coerência Semântica...")
        coherence_prompt = f"""
Act as an expert game evaluator.
You are tasked with evaluating the coherence between a game's intended thematic description and its generated procedural assets.

=== REFINED THEME DESCRIPTION (INTENT) ===
{refined_description}

=== GENERATED ASSETS (JSON) ===
{stripped_bundle_json}

Review the alignment between the textual intent and the final game artifacts.
Focus on visual similarity and thematic consistency across entities, environments, weapons, and enemies.
Assign a score from 0 to 100, where 100 means perfect alignment and coherence.
"""
        try:
            # Uso da função segura _ask_llm_structured com retentativas
            coherence_eval = evaluator._ask_llm_structured(
                CoherenceEvaluation, 
                [HumanMessage(coherence_prompt)]
            )
            coherence_score = coherence_eval.score
        except Exception as e:
            print(f"[{prompt_name}] Erro definitivo na avaliação de Coerência: {e}")

        # ------------------------------------------------------------
        # MÉTRICA 2: Fidelidade de Reconstrução (Reconstruction Metric)
        # ------------------------------------------------------------
        print(f"[{prompt_name}] Avaliando Reconstrução Semântica...")
        reconstruction_prompt = f"""
Act as a Lead Game Designer and Narrative Architect.
You are given the JSON data of procedurally generated game assets for a Roguelike game.
Your task is to infer and reconstruct the game's theme, setting, visual style, atmosphere, lore, and core conflict solely based on the provided assets.

=== GENERATED ASSETS (JSON) ===
{stripped_bundle_json}

Write a cohesive "World Description" that conveys the thematic intent of these assets entirely in English.
Do not list the JSON fields or stats; write only the narrative setting description.
"""
        try:
            # Como a reconstrução gera apenas um texto livre, usamos a invocação direta do modelo
            reconstruction_response = evaluator.model.invoke(
                [HumanMessage(reconstruction_prompt)],
                config={"callbacks": [evaluator.usage_callback]}
            )
            reconstructed_text = str(reconstruction_response.content)
            
            # Calcula a similaridade do cosseno entre o texto refinado original e o texto reconstruído
            reconstruction_score = get_cosine_similarity(refined_description, reconstructed_text)
        except Exception as e:
            print(f"[{prompt_name}] Erro na avaliação de Reconstrução: {e}")


        # ------------------------------------------------------------
        # SALVAR RESULTADOS SEPARADOS (AVALIAÇÃO E BUNDLE)
        # ------------------------------------------------------------
        
        # Gerar identificadores seguros e únicos
        safe_tested_name = tested_model_name.replace("/", "_").replace("-", "_")
        unique_id = uuid.uuid4().hex[:6]
        
        eval_filename = f"eval_{prompt_name}_{safe_tested_name}_{unique_id}.json"
        bundle_filename = f"bundle_{prompt_name}_{safe_tested_name}_{unique_id}.json"
        
        eval_file_path = join(tests_dir, eval_filename)
        bundle_file_path = join(tests_dir, bundle_filename)
        
        # 1. Salvar o Asset Bundle isoladamente no seu próprio arquivo
        # Converte para dict, salva como JSON indentado (ou poderia usar bundle.model_dump_json(indent=4))
        with open(bundle_file_path, "w", encoding="utf-8") as f:
            f.write(bundle.model_dump_json(indent=4))
        
        # 2. Salvar as métricas com o nome do arquivo do bundle atrelado
        final_result_data = {
            "tested_model_name": tested_model_name,
            "judge_model_name": judge_model_name,
            "semantic_coherence_metric": coherence_score,
            "semantic_reconstruction_metric": float(reconstruction_score),
            "prompt": prompt_text,
            "prompt_name": prompt_name,
            "prompt_index": prompt_index,
            "asset_bundle_file": bundle_filename  # Referência ao arquivo gerado
        }
        
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(final_result_data, f, ensure_ascii=False, indent=4)
            
        print(f"[{prompt_name}] Teste concluído!")
        print(f" -> Avaliação salva em: {eval_filename}")
        print(f" -> Asset Bundle salvo em: {bundle_filename}")
        
        results.append(final_result_data)
        
        # Aguarda 2 minutos (para evitar Rate Limits pesados das APIs free)
        print("Aguardando 2 minutos antes do próximo teste...\n")
        time.sleep(60 * 2)
        
    return results


if __name__ == "__main__":

    # 15 Prompts extremamente variados para testar robustez no TCC
    test_inputs = [
        {
            "prompt": "A submerged neon-gothic underwater city filled with mutated fish-people cultists.",
            "prompt_name": "neon_gothic_underwater",
            "prompt_index": 1
        },
        {
            "prompt": "An abandoned orbital space station overgrown by aggressive, bioluminescent plant life and hive-mind fungi.",
            "prompt_name": "overgrown_space_station",
            "prompt_index": 2
        },
        {
            "prompt": "A clockwork purgatory inside a colossal ticking pocket watch, guarded by rusted brass automatons and ticking gear-wraiths.",
            "prompt_name": "clockwork_purgatory",
            "prompt_index": 3
        },
        {
            "prompt": "A desolate, sun-scorched desert canyon littered with the colossal bones of ancient titans, inhabited by scavengers and blood-magic shamans.",
            "prompt_name": "titan_bone_desert",
            "prompt_index": 4
        },
        {
            "prompt": "A frozen medieval citadel trapped in eternal blizzard, where cursed frost-knights and ice-gargoyles protect a shattered frozen throne.",
            "prompt_name": "frozen_citadel",
            "prompt_index": 5
        },
        # {
        #     "prompt": "A twisted, flesh-and-bone labyrinth inside the stomach of a sleeping eldritch god, swarming with parasitic worms and twisted digestive demons.",
        #     "prompt_name": "eldritch_flesh_labyrinth",
        #     "prompt_index": 6
        # },
        # {
        #     "prompt": "A cyberpunk mega-slum built on top of a massive toxic landfill, controlled by cyborg syndicates and rogue surgical drones.",
        #     "prompt_name": "toxic_cyberpunk_slum",
        #     "prompt_index": 7
        # },
        # {
        #     "prompt": "An ethereal floating library in the astral plane, slowly crumbling into the void, haunted by ink-elementals and corrupted scholars.",
        #     "prompt_name": "astral_floating_library",
        #     "prompt_index": 8
        # },
        # {
        #     "prompt": "A Victorian-era train hurtling infinitely through a realm of thick fog, infested with shapeshifting vampires and spectral passengers.",
        #     "prompt_name": "victorian_ghost_train",
        #     "prompt_index": 9
        # },
        # {
        #     "prompt": "A subterranean dwarven mining complex entirely corrupted by liquid gold and animated greed-spirits, featuring molten rivers and gemstone golems.",
        #     "prompt_name": "corrupted_gold_mines",
        #     "prompt_index": 10
        # },
        # {
        #     "prompt": "A post-apocalyptic amusement park in a radioactive wasteland, patrolled by deranged animatronics and mutated clowns.",
        #     "prompt_name": "radioactive_amusement_park",
        #     "prompt_index": 11
        # },
        # {
        #     "prompt": "A hyper-advanced alien mothership made of living crystalline structures, defended by light-bending templars and prismatic sentinels.",
        #     "prompt_name": "crystalline_alien_mothership",
        #     "prompt_index": 12
        # },
        # {
        #     "prompt": "A cursed feudal Japanese village trapped in an eternal blood-moon night, stalked by vengeful yokai and undead samurai.",
        #     "prompt_name": "blood_moon_yokai_village",
        #     "prompt_index": 13
        # },
        # {
        #     "prompt": "A dreamscape realm made of floating staircases and melting clocks, populated by faceless manifestations of forgotten memories and nightmare weavers.",
        #     "prompt_name": "surreal_dreamscape",
        #     "prompt_index": 14
        # },
        # {
        #     "prompt": "A sprawling trench-warfare wasteland locked in World War I era technology, but twisted by necromancy, featuring zombie infantry and chemical-mutated behemoths.",
        #     "prompt_name": "dieselpunk_necromancy_trenches",
        #     "prompt_index": 15
        # }
    ]
    
    # Definição dos Modelos Testado e Juiz
    tested_prov = Providers.GROQ
    tested_model = GroqModels.OPENAI_GPT_OSS_20B
    
    judge_prov = Providers.GROQ
    judge_model = GroqModels.MOONSHOTAI_KIMI_K2_INSTRUCT
    
    print("Iniciando bateria de testes...\n")
    run_evaluation_pipeline(
        tested_provider=tested_prov,
        tested_model_name=tested_model,
        judge_provider=judge_prov,
        judge_model_name=judge_model,
        test_inputs=test_inputs 
    )
    print("\nTodos os testes finalizados com sucesso!")
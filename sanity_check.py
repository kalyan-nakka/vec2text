import vec2text
import sys
from typing import List
from vec2text.models.model_utils import device
import torch

inversion_model  = None
corrector_model = None
prompt_index = 0
# 0 for V2T_CLIPv1_4
# 1 for V2T_CLIPv2_1
# 2 for V2T_CLIPv1_0

test_prompts = [
    "A serene Japanese garden with a red wooden bridge over a koi pond, cherry blossoms falling, soft morning light filtering through maple trees",
    "A cyberpunk street market at night with neon signs, holographic advertisements, street vendors selling ramen, rain-slicked pavement reflecting purple and blue lights",
    "An ancient library with towering bookshelves reaching into darkness, a single beam of sunlight illuminating floating dust particles, leather-bound books and wooden ladders",
    "A cozy coffee shop interior on a rainy day, steaming cup of cappuccino on a wooden table, blurred people in background, warm yellow lighting, water droplets on window",
    "A majestic snow leopard standing on a rocky mountain cliff at sunset, its spotted fur catching golden light, distant Himalayan peaks in the background",
    "An abandoned space station floating in deep space, broken solar panels, Earth visible in the distance, stars and nebulae creating a purple and blue backdrop",
    "A medieval blacksmith workshop with a glowing forge, sparks flying as a hammer strikes an anvil, iron tools hanging on stone walls, muscular figure silhouetted against orange flames",
    "A whimsical underwater coral reef scene with tropical fish, sea turtle swimming past, sunlight rays penetrating blue water, colorful anemones and brain coral formations",
    "A Victorian-era detective's office at night, desk covered with case files and magnifying glass, pipe resting in ashtray, shadows cast by a green-shaded lamp, rain visible through window",
    "A futuristic city skyline at dawn with flying cars, massive holographic billboards, glass skyscrapers with vertical gardens, orange and pink sky reflecting off chrome surfaces"
]


if len(sys.argv) > 1:
    if sys.argv[1] == "J":
        inversion_model = vec2text.models.InversionModel.from_pretrained("AusmitM/SD14_100K_J_Inv")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("AusmitM/SD14_100K_J_Corr")
    elif sys.argv[1] == "A":    
        inversion_model = vec2text.models.InversionModel.from_pretrained("AusmitM/SD14_100K_A_Inv")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("AusmitM/SD14_100K_A_Corr")
    elif sys.argv[1] == "250K":    
        inversion_model = vec2text.models.InversionModel.from_pretrained("AusmitM/SD14_250K_A_Inv")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("AusmitM/SD14_250K_A_Corr")


    prompt_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
else:
    print("Usage: python sanity_check.py <model_version> <prompt_index>")
    print("model_version: J for SD14_100K_J, A for SD14_100K_A")
    print("prompt_index: index of the prompt to use from the predefined list (0-9)")
    sys.exit(1)



corrector = vec2text.load_corrector(inversion_model, corrector_model)
# corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

# do one prommpt at a time to save VRAM
def generate_embedding(strings: List[str])-> List[str]:
    inputs = corrector.embedder_tokenizer(
        strings,
        return_tensors="pt",
        max_length=77, #changed from 128 to 77
        truncation=True,
        padding="max_length",
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        frozen_embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return frozen_embeddings
    

frozen_embeddings = generate_embedding([test_prompts[prompt_index]])
num_steps=20 #20
sequence_beam_width=8 #8

result=vec2text.invert_embeddings(
        embeddings=frozen_embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )



# print(f"\n{'='*60}")
# print(f"ORIGINAL:      {test_prompts[prompt_index]}")
# print(f"RECONSTRUCTED: {result}")
# print(f"Result type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
# print(f"{'='*60}\n")

print(f"ORIGINAL PROMPT: {test_prompts[prompt_index]}")
print(f"INVERTED PROMPT: {result[0] if result else ''}")
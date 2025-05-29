import json
import os
import random
from datasets import Dataset

PRETRAIN_PHRASES = [
    "What is happening in the given music <sound>?",
    "Describe the sound. <sound>",
    "Describe the music. <sound>",
    "<sound> Provide a description of the music.",
    "<sound> Provide a description of the sound.",
    "Can you interpret <sound>?",
    "Please explain what's happening in <sound>",
    "What does <sound> represent?",
    "Could you describe <sound> for me?",
    "What's the content of <sound>?",
    "Can you depict <sound>?",
    "What is <sound>?",
    "In the music clip, <sound>, what is happening?",
    "Provide a description of the music. <sound>",
    "Provide a description of the sound. <sound>",
    "Provide a caption for the sound. <sound>",
    "Provide a caption for the music. <sound>",
]

def convert_json_to_dataset(input_file, output_folder, train_ratio=0.8):
    with open(input_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    os.makedirs(output_folder, exist_ok=True)
    cache_path = os.path.join(output_folder, "gpt-cache.jsonl")
    cache = open(cache_path, "a")

    # Shuffle the data
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    def gen(entries):
        for idx, entry in enumerate(entries):
            sound_location = entry["location"]
            main_caption = entry["main_caption"]
            alt_caption = entry["alt_caption"]
            
            # Randomly select a pretrain phrase for user content
            user_content = random.choice(PRETRAIN_PHRASES)

            # Construct the main example
            example_1 = {
                "id": f"{2*idx + 1:07}",
                "sounds": [sound_location],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": main_caption}
                ]
            }

            cache.write(json.dumps(example_1) + "\n")
            yield example_1

            # Construct the alt example
            example_2 = {
                "id": f"{2*idx+2:07}",
                "sounds": [sound_location],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": alt_caption}
                ]
            }

            cache.write(json.dumps(example_2) + "\n")
            yield example_2

    train_ds = Dataset.from_generator(
        gen,
        num_proc=1,  # Set num_proc to adjust parallel processing
        gen_kwargs={"entries": train_data},
    )
    train_ds.save_to_disk(os.path.join(output_folder, "train"))

    val_ds = Dataset.from_generator(
        gen,
        num_proc=1,  # Set num_proc to adjust parallel processing
        gen_kwargs={"entries": val_data},
    )
    val_ds.save_to_disk(os.path.join(output_folder, "val"))

    cache.close()

if __name__ == "__main__":
    input_file = "/Users/anuradhachopra/Downloads/MusicBench_train.json"  # Change this to your input JSON file path
    output_folder = "/Users/anuradhachopra/data/musicbench_multitoken"    # Change this to your desired output folder path

    convert_json_to_dataset(input_file, output_folder, train_ratio=0.8)

import os
import ujson
import tqdm


def calculate_samples():
    base_path = '/mnt/8T/high_modality/multimodal/mimiciv/admissions_multimodal'
    jsons = os.listdir(base_path)

    total_samples = 0
    ihm_48_pos = 0
    ihm_48_neg = 0
    survived = 0
    for json_path in tqdm.tqdm(jsons):
        if not json_path.endswith('.json'):
            continue
        with open(os.path.join(base_path, json_path), 'r') as f:
            data = ujson.load(f)
        total_samples += 1
        ihm_48_pos += data['48_ihm']
        ihm_48_neg += 1 - data['48_ihm']
        survived += data['survived']

    print(f"Total samples: {total_samples}")
    print(f"IHM 48 positive: {ihm_48_pos}")
    print(f"IHM 48 negative: {ihm_48_neg}")
    print(f"Survived: {survived}")



if __name__ == '__main__':
    calculate_samples()
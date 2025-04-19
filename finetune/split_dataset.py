import json
import random

def main():

    input_filename = "../data/processed/clean_data.json"
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_examples = len(data)
    print(f"Total exemple în {input_filename}: {total_examples}")

    random.shuffle(data)

    desired_train = 1474
    desired_test = 184
    desired_valid = 185
    desired_total = desired_train + desired_test + desired_valid

    if total_examples < desired_total:
        print("Totalul de exemple este mai mic decat 1843. Se ajustează marimile subseturilor.")
        desired_train = int(total_examples * 0.63)
        desired_test = int(total_examples * 0.185)
        desired_valid = total_examples - desired_train - desired_test
        print(f"Dimensiuni ajustate: train={desired_train}, test={desired_test}, valid={desired_valid}")
    else:
        print(f"Folosim marimile: train={desired_train}, test={desired_test}, valid={desired_valid}")

    train_data = data[:desired_train]
    test_data = data[desired_train:desired_train+desired_test]
    valid_data = data[desired_train+desired_test:desired_train+desired_test+desired_valid]

    with open("../data/processed/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open("../data/processed/test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    with open("../data/processed/validate_data.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=4, ensure_ascii=False)

    print("Fisierele 'train_data.json', 'test_data.json' și 'validate_data.json' au fost salvate cu succes.")

if __name__ == "__main__":
    main()
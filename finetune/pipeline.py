from finetune.download_dataset import main as dataset_download
from finetune.split_dataset import main as dataset_split
from finetune.train import main as model_finetuning

def main():
    print("Pasul 1: Descarc datele de pe HuggingFace...")
    dataset_download()

    print("Pasul 2: Splituiesc datele in train/test/valid...")
    dataset_split()

    print("Pasul 3: Antrenez modelul...")
    model_finetuning()

    print("Totul a fost executat cu succes.")

if __name__ == "__main__":
    main()

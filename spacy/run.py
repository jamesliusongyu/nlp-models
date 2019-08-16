import spacy
import config

def main():
    output_dir = config.SPACY_MODELS
    TEST_DATA = config.TEST_DATA

    # test the model
    spacy_model = spacy.load(output_dir)
    for text, _ in TEST_DATA:
        doc = spacy_model(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

if __name__ == "__main__":
    main()

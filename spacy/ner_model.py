from typing import Any
from typing import List
from typing import Tuple

import click
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

import helpers
import config


def clean_data() -> List[Tuple]:
    for country in config.ALL_DATA:
        # Read country data; Doccano format
        data = helpers.load_input_file(country)

        # Clean data to fit Spacy format
        spacy_training_data = [
            (
                row['text'],
                {'entities': [(ln[0], ln[1], ln[2].upper()) for ln in [tuple(l) for l in row['labels']]]}
            )
            for row in data]

    return spacy_training_data


@click.command()
@click.option('--model', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--output_dir', default=None, type=click.Path(exists=True, dir_okay=False))
def main(model: str, output_dir: str) -> None:
    TRAIN_DATA = clean_data()
    n_iter = config.N_ITER

    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    output_dir = config.SPACY_MODELS
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


if __name__ == "__main__":
    main()

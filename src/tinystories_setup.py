"""Restore Part II's story-to-batch teaching sequence inside Part III.

Reuse the original figures and explanations. List-allocation/append microsteps
and repeated MLP route maps stay in Notebook 5, not the classroom sequence.
"""
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/wordlm'))
from slow_walkthrough import STAGES

SETUP_STEPS = (
    'data', 'story-complete', 'story-excerpts', 'split', 'sentence',
    'tokenization-intro', 'tokenization-choices', 'tokenization-rules',
    'tokenize', 'vocabulary', 'special', 'boundaries',
    'story-indices', 'one-pair', 'pair-first', 'pair-second',
    'windows-first', 'windows-last', 'pairs-tensors',
    'counts-story', 'counts', 'context', 'batch', 'batch-ids', 'batches',
)

# Only adjust references to steps/models that are no longer on the main route.
COPY = {
    'sentence': 'We will trace this complete authored example using a small vocabulary of 10 items and four input slots. The trained experiment later uses 4,000 items and 64 slots.',
    'tokenization-rules': 'All three benchmark models use these same word-and-punctuation rules. Build the vocabulary from training stories only; map a missing item to UNK.',
    'vocabulary': 'Six ordinary tokens plus four special tokens give C = 10 in this teaching example. IDs name vocabulary items; they are not embedding coordinates.',
    'special': '“blue” is absent from this toy vocabulary, so its ID is 3, UNK. BOS and EOS mark the whole story; PAD fills unused input slots.',
    'boundaries': 'One BOS starts the whole story and one EOS ends it, even if it contains several sentences. Full stops remain ordinary tokens inside the story.',
    'pair-second': 'Move one token forward. The observed “lily” joins the input, and “found” becomes the next target. Training uses the story’s words, not the model’s guesses.',
    'windows-first': 'Repeat that operation for each target. These are the first four input–target pairs, with IDs translated back into tokens.',
    'windows-last': 'A full window keeps the previous four tokens and drops older ones. The last target is EOS; windows never cross into another story.',
    'pairs-tensors': 'Store all seven input rows in all_X [7, 4] and their seven answers in all_y [7]. These are integer token IDs, not learned vectors yet.',
    'batch-ids': 'X contains two examples, each with four token IDs. y contains their two next-token IDs. Embedding lookup transforms X; the loss still uses y as IDs.',
    'batches': 'Seven examples with B = 2 give batches of 2, 2, 2 and 1 in a sequential pass. The saved experiment instead samples 512 windows per update, for 6,000 updates.',
}


def setup_frames(stage):
    sources = {s['id']: s for s in STAGES}
    result = []
    for key in SETUP_STEPS:
        source = sources[key]
        figure = (ROOT / 'figures/wordlm-pipeline/steps' / (key + '.svg')).read_text()
        body = COPY.get(key, '. '.join(source['body'].split('. ')[:2]).rstrip('.') + '.')
        companion = '<p>' + escape(source['body'].replace('Both models', 'All three models')) + '</p>'
        companion += f'<p><a href="notebooks/wordlm/05_training_and_inference_maps.html#{key}">Original worked example and executable code</a></p>'
        if key in {'data', 'story-complete', 'story-excerpts'}:
            companion += '<p>TinyStories by Ronen Eldan and Yuanzhi Li. <a href="https://huggingface.co/datasets/roneneldan/TinyStories">Source dataset</a>, <a href="https://cdla.dev/sharing-1-0/">CDLA-Sharing-1.0</a>. Source revision, row IDs and unchanged texts are recorded in <a href="notebooks/wordlm/story_examples.json">story_examples.json</a>. […] marks omitted text.</p>'
        if key.startswith('tokenization-'):
            companion += '<p><a href="https://huggingface.co/docs/transformers/tokenizer_summary">Hugging Face tokenizer overview</a>. The subword split is illustrative, not a fitted BPE output. The executable English tokenizer is in <a href="notebooks/wordlm/wordlm.py">wordlm.py</a>.</p>'
        if source.get('check'):
            question, answer = source['check']
            companion += f'<details><summary>{escape(question)}</summary><p>{escape(answer)}</p></details>'
        notes = source['body'] + ' This figure is reused from the Part II TinyStories lab.'
        if key == 'data':
            notes += ' Aim for 10–15 minutes through the TinyStories finale: 5–6 minutes on stories and batches, 4–5 on the multi-head model and generation, 2–3 on results and the app. The successive frames reveal small steps; avoid treating each as a separate mini-lecture.'
        result.append(stage('s03-data-' + key, source['title'], figure, body,
                            notes=notes, companion=companion))
    return result

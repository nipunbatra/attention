"""Shared lesson steps and numeric figures for the notebook and Part II slides.

Each figure is rendered from the variables produced by that step's actual code.
Run build_slow_lesson.py to execute the notebook and export the teaching pages.
"""
from __future__ import annotations
from html import escape
from pathlib import Path
import json
import textwrap

from pipeline_maps import pipeline_svg

ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads((ROOT / 'lesson_evidence.json').read_text())
STORY_EXAMPLES = json.loads((ROOT / 'story_examples.json').read_text())
STAGES = []


def step(key, title, chapter, body, code, *, kind='mlp', focus=(), check=None):
    STAGES.append(dict(id=key, title=title, chapter=chapter, body=body,
                       code=textwrap.dedent(code).strip(), kind=kind,
                       focus=focus, check=check))


step('data', 'The data: complete short stories', '1. Data and tokens',
     'The saved experiment uses 6,000 TinyStories documents. Each document is a complete story. This is an excerpt from source row 12400.', '''
sample = evidence['sample']
print(sample['text_excerpt'])
print('Source row:', sample['row_idx'])
''', focus=('stories',))

step('story-complete', 'Read one complete story', '1. Data and tokens',
     'One document is one complete story, not one sentence. This is the shortest story in our saved 6,000-document subset; many others are longer.', '''
short_story = story_examples['examples'][0]
short_words = len(short_story['text'].split())
short_tokens = len(tokenize(short_story['text']))
assert (short_words, short_tokens) == (52, 60)
print(short_story['text'])
print(f'Full story: {short_words} words; {short_tokens} tokens.')
''', focus=('stories',))

step('story-excerpts', 'Longer stories have several paragraphs', '1. Data and tokens',
     'These beginnings and endings come from two other documents in the same subset. […] marks omitted text; the lengths count each whole story. Words are whitespace-separated items; tokens also separate punctuation and exclude start/end markers.', '''
long_stories = story_examples['examples'][1:]
story_lengths = []
for story in long_stories:
    counts = (len(story['text'].split()), len(tokenize(story['text'])))
    assert counts == (story['word_count'], story['token_count'])
    story_lengths.append(counts)
    print(f"Source row {story['row_idx']}: {counts[0]} words; {counts[1]} tokens")
    print(story['text'] + '\\n')  # full texts here; excerpts in the figure
train_lengths = evidence['audit']['story_length_tokens']['train']
print('Training-story token lengths:', train_lengths)
''', focus=('stories',), check=('Is a 300-token story a single 64-token training input?', 'No. A story is a document. Later, we turn it into many next-token examples, each using at most the previous 64 tokens in the saved experiment.'))

step('split', 'Stories stay together when we split the data', '1. Data and tokens',
     'The split happens before overlapping windows are made. Training fits parameters and the vocabulary. Validation chooses settings and checkpoints. Test measures the frozen choice.', '''
audit = evidence['audit']
print(audit['documents'])
assert sum(audit['documents'].values()) == 6000
assert not any(audit['document_overlap'].values())
''', focus=('split',))

step('sentence', 'A sentence we can trace completely', '1. Data and tokens',
     'We use this authored sentence for the arithmetic. Its small vocabulary and randomly initialized models are separate from the measured TinyStories experiment.', '''
sentence = 'Lily found a red ball.'
print(sentence)
''', focus=('stories',))

step('tokenization-intro', 'Tokenization', '1. Data and tokens',
     'A token is one item the model reads or predicts. The tokenizer chooses these items before we assign integer IDs and look up learned embeddings.', '''
tokenization_text = 'redder!'
print('The same text for three tokenization choices:', tokenization_text)
''', focus=('tokenize',))

step('tokenization-choices', 'Words, characters or pieces of words', '1. Data and tokens',
     'Whole-word vocabularies need entries for many word forms, while characters produce longer sequences. Subword methods such as byte pair encoding (BPE) learn reusable pieces from training text to balance these concerns.', '''
tokenization_choices = {
    'Word + punctuation': tokenize(tokenization_text),
    'Character': list(tokenization_text),
    'Subword (illustrative)': ['red', 'der', '!'],
}
tokenization_counts = {name: len(parts) for name, parts in tokenization_choices.items()}
assert list(tokenization_counts.values()) == [2, 7, 3]
for name, parts in tokenization_choices.items():
    assert ''.join(parts) == tokenization_text
    print(name, parts, 'tokens:', len(parts))
# The subword split is hand-chosen, not the output of a trained BPE tokenizer.
''', focus=('tokenize',), check=('Does a context window of four tokens always hold four words?', 'No. It holds four items produced by that tokenizer. A word can take several subword or character tokens, and punctuation can use a token too.'))

step('tokenization-rules', 'The tokenizer used in this notebook', '1. Data and tokens',
     'Words plus punctuation keep our calculations easy to inspect. Both models use the same rules and vocabulary during training and generation. We build the vocabulary from training stories only and map missing words to the unknown-token marker, UNK.', '''
tokenizer_probe = "Lily can't find 12 balls!"
probe_tokens = tokenize(tokenizer_probe)
assert probe_tokens == ['lily', "can't", 'find', '12', 'balls', '!']
assert tokenize('LILY   found!') == ['lily', 'found', '!']
print(tokenizer_probe)
print(probe_tokens)
# This English teaching tokenizer drops case and spacing.
# Its word pattern is ASCII-based; it is not a general multilingual tokenizer.
''', focus=('tokenize','ids'))

step('tokenize', 'Lowercase words and separate punctuation', '1. Data and tokens',
     'This tokenizer normalizes Unicode, lowercases text and keeps punctuation as tokens. Five words plus the full stop give six tokens. A tokenizer decides what one prediction unit is.', '''
pieces = tokenize(sentence)
print(pieces)
assert pieces == ['lily', 'found', 'a', 'red', 'ball', '.']
assert len(pieces) == 6
''', focus=('tokenize',))

step('vocabulary', 'Every vocabulary item gets an integer ID', '1. Data and tokens',
     'Our demonstration has six ordinary tokens and four special tokens, so C=10. IDs are arbitrary labels. The benchmark uses a different, frequency-ranked vocabulary with C=4,000.', '''
words = list(SPECIAL_TOKENS) + sorted(set(pieces))
vocab = Vocabulary(words, {t:i for i,t in enumerate(words)}, {}, 10, 1)
C = len(words)
assert C == 10
print(vocab.stoi)
''', focus=('tokenize','ids'))

step('special', 'Four special tokens have different jobs', '1. Data and tokens',
     'BOS marks the start of each story. EOS is an observed stopping target. PAD fills unused input slots. UNK represents a word missing from the fitted vocabulary.', '''
unknown = vocab.encode_tokens(tokenize('blue'), boundaries=False)
assert unknown == [vocab.unk_id]
print('blue maps to', words[unknown[0]])
''', focus=('ids','windows'))

step('boundaries', 'Add the story boundaries once', '1. Data and tokens',
     'There are now eight IDs: BOS, six ordinary tokens and EOS. We can ask for the next ID after each prefix. BOS is input context; it is not one of the seven targets.', '''
ids = vocab.encode_tokens(pieces)
print(ids)
assert ids == [1, 8, 7, 5, 9, 6, 4, 2]
''', focus=('ids',), check=('How many ordinary tokens are there?', 'Six. The full stop counts as one token; BOS and EOS are additional boundary tokens.'))

step('story-indices', 'A position in the story is different from a token ID', '2. Windows and batches',
     'Python positions start at 0: position 4 contains red, whose vocabulary ID is 9. With a four-token window, this example reads positions 0 through 3 and predicts the token at position 4.', '''
w = 4
target_position = 4
print(list(enumerate(ids)))
''', focus=('windows',))

step('one-pair', 'One input window and its observed target', '2. Windows and batches',
     'context_ids is the input for one example, often called x, and target_id is its single observed answer, often called y. The slice ids[0:4] stops before position 4, so red is outside this input.', '''
context_ids = ids[target_position-w:target_position]
target_id = ids[target_position]
assert context_ids == [1, 8, 7, 5] and target_id == 9
print('context_ids:', context_ids, 'target_id:', target_id)
''', focus=('windows',))

step('pair-lists', 'Two empty lists will collect the training pairs', '2. Windows and batches',
     'We inspected the example with red as its target, but have not stored any pairs yet. To collect every example in order, start at position 1, where lily follows BOS.', '''
contexts = []
targets = []
assert len(contexts) == len(targets) == 0
''', focus=('windows',))

step('pair-first', 'The first pair has only BOS as history', '2. Windows and batches',
     'At position t=1, the visible prefix is [1], meaning BOS. Three PAD IDs fill the unused slots, giving context_ids=[0, 0, 0, 1] and target_id=8 for lily.', '''
t = 1
visible = ids[max(0, t-w):t]
context_ids = [vocab.pad_id] * (w-len(visible)) + visible
target_id = ids[t]
assert context_ids == [0, 0, 0, 1] and target_id == 8
''', focus=('windows',))

step('pair-append', 'One append stores the input, the other stores its target', '2. Windows and batches',
     'contexts.append(context_ids) adds the whole four-ID list as one row. targets.append(target_id) adds one answer at the same row index, so contexts[0] and targets[0] belong together.', '''
contexts.append(context_ids)
targets.append(target_id)
assert contexts == [[0, 0, 0, 1]] and targets == [8]
print('contexts =', contexts)
print('targets =', targets)
''', focus=('windows',))

step('pair-second', 'The next pair includes the previous observed target', '2. Windows and batches',
     'At t=2, BOS and lily form the known prefix, and the observed next word is found. Computing this pair changes context_ids and target_id, while the stored lists still contain only the first pair.', '''
t = 2
visible = ids[max(0, t-w):t]
context_ids = [vocab.pad_id] * (w-len(visible)) + visible
target_id = ids[t]
assert context_ids == [0, 0, 1, 8] and target_id == 7
assert contexts == [[0, 0, 0, 1]] and targets == [8]
''', focus=('windows',))

step('pair-append-second', 'The second pair becomes row 1 in both lists', '2. Windows and batches',
     'The same two append calls add the new pair after the first one. Each row in contexts still lines up with exactly one entry in targets.', '''
contexts.append(context_ids)
targets.append(target_id)
assert contexts == [[0, 0, 0, 1], [0, 0, 1, 8]]
assert targets == [8, 7]
print('contexts =', contexts)
print('targets =', targets)
''', focus=('windows',))

step('pairs-loop', 'The loop repeats those steps for the remaining positions', '2. Windows and batches',
     'Positions 1 and 2 are already stored, so this loop continues at 3 and stops before len(ids)=8. Each pass builds a fresh input list and appends it with the observed target, producing seven paired rows in total.', '''
for t in range(3, len(ids)):
    visible = ids[max(0, t-w):t]
    context_ids = [vocab.pad_id] * (w-len(visible)) + visible
    target_id = ids[t]
    contexts.append(context_ids)
    targets.append(target_id)
assert len(contexts) == len(targets) == 7
assert contexts[3] == ids[0:4] and targets[3] == ids[4]
''', focus=('windows',))

step('windows-first', 'Early prefixes need padding', '2. Windows and batches',
     'These are the first four stored rows, with the IDs translated back to tokens. Row 3 is the red-target pair we inspected first, now stored in contexts[3] and targets[3].', '''
for row in range(4):
    print(row, contexts[row], targets[row])
assert contexts[3] == [1, 8, 7, 5] and targets[3] == 9
''', focus=('windows',))

step('windows-last', 'Later prefixes drop their oldest tokens', '2. Windows and batches',
     'The remaining three rows use full windows, keeping only the four tokens immediately before each target. The last target is EOS, and no window crosses into a different story.', '''
for row in range(4, 7):
    print(row, contexts[row], targets[row])
assert targets[-1] == vocab.eos_id
''', focus=('windows',))

step('pairs-tensors', 'The paired lists become two integer tensors', '2. Windows and batches',
     'The conversion preserves every token ID and row pairing. all_X has shape [7, 4], all_y has shape [7], and torch.long means integer IDs.', '''
all_X = torch.tensor(contexts, dtype=torch.long)
all_y = torch.tensor(targets, dtype=torch.long)
N = len(all_y)
assert N == 7 and all_X.shape == (7, 4)
assert all_X.tolist() == contexts and all_y.tolist() == targets
assert not all_y.eq(vocab.pad_id).any()
''', focus=('windows',), check=('Have these token IDs become embeddings yet?', 'No. This step only organizes the IDs into tensors. The embedding layer will later look up a learned vector for each ID.'))

step('counts-story', 'Six ordinary tokens give seven training examples', '2. Windows and batches',
     'Each ordinary token is a target once, and EOS supplies one more target. BOS starts the history, while PAD fills empty input slots, so neither adds a target.', '''
ordinary_tokens = len(pieces)
examples_in_story = ordinary_tokens + 1
assert ordinary_tokens == 6 and examples_in_story == N == 7
''', focus=('windows',))

step('counts-train', 'The same count across all training stories', '2. Windows and batches',
     'The saved training split has 964,338 ordinary tokens across 4,822 complete stories. Adding one EOS target per story gives 969,160 training examples.', '''
train_tokens = 964_338
train_stories = 4_822
train_examples = train_tokens + train_stories
assert train_tokens == audit['oov']['train']['tokens']
assert train_stories == audit['documents']['train']
assert train_examples == 969160
''', focus=('stories','windows'))

step('counts', 'Training, validation and test use the same counting rule', '2. Windows and batches',
     'For each split, add its ordinary-token count and its story count. These are counts of supervised examples, not optimizer steps.', '''
window_counts = {
    split: audit['oov'][split]['tokens'] + count
    for split, count in audit['documents'].items()
}
assert window_counts['train'] == 969160
assert window_counts['train'] == train_examples
print(window_counts)
''', focus=('stories','windows'), check=('Does w=8 create more targets than w=4 here?', 'No. With this padding rule, both give seven targets. A wider window changes the visible input history, not the target count.'))

step('context', 'Context length controls what is visible', '2. Windows and batches',
     'History is everything already known. The context window is the suffix read for this prediction. Increasing w can recover an older clue, but also changes model size or computation.', '''
history = ids[:6]  # BOS lily found a red ball
contexts_by_width = {width: history[-width:] for width in [2, 4, 6]}
print(contexts_by_width)
''', focus=('windows',))

step('batch', 'A real batch with B=2', '2. Windows and batches',
     'Each batch row pairs four input token IDs in X with one observed next-token ID in y. The text columns decode the IDs, and the targets come from the story.', '''
selected = torch.tensor([2, 3])
X, y = all_X[selected], all_y[selected]
B = X.shape[0]
assert X.tolist() == [[0, 1, 8, 7], [1, 8, 7, 5]]
assert y.tolist() == [5, 9] and B == 2
''', focus=('windows',))

step('batch-ids', 'X and y contain integer token IDs', '2. Windows and batches',
     'Tokenization and vocabulary lookup are complete: X and y contain integers, with no embedding coordinates yet. Next, X goes through embedding lookup, while y stays as target IDs for the loss.', '''
print(X.shape, X.dtype)
print(y.shape, y.dtype)
assert X.shape == (2, 4) and y.shape == (2,)
assert X.dtype == y.dtype == torch.long
''', focus=('windows',), check=('Are the four numbers in each row of X embedding coordinates?', 'No. They are IDs for four separate token slots. Embedding lookup later replaces each input ID with a learned vector. X.shape describes the size of the ID tensor, not its contents.'))

step('batches', 'Seven examples do not mean seven optimizer steps', '2. Windows and batches',
     'A simple loader with B=2 and drop_last=False makes four batches in one pass: 2, 2, 2 and 1 example. The real benchmark instead samples B=512 windows per step with replacement.', '''
batch_sizes = [len(all_y[start:start+B]) for start in range(0, N, B)]
assert batch_sizes == [2, 2, 2, 1]
print('One sequential pass:', batch_sizes)
print('Benchmark target presentations:', 6000 * 512)
''', focus=('windows','optimizer'))

step('shapes', 'Concrete dimensions for the worked batch', '2. Windows and batches',
     'The batch and context axes describe the data. The representation widths are model choices. The vocabulary size is the number of output classes, including special tokens.', '''
d, h, d_k, d_v = 4, 8, 3, 2
torch.manual_seed(11)
mlp = FixedWindowMLP(C, w, d, h, vocab.pad_id)
torch.manual_seed(11)
attention = CausalAttentionLM(C, w, d, d_k, d_v, h, vocab.pad_id)
print(dict(B=B, w=w, C=C, d=d, h=h, d_k=d_k, d_v=d_v))
''', check=('Must d_k equal d_v?', 'No. Queries and keys need the same width for their dot product. Values may have a different width. Here d_k=3 and d_v=2.'))

step('mlp-map', 'The full MLP training map', '3. The MLP forward pass',
     'We have prepared X and y. Follow X through the network. The observed target bypasses the network and joins the logits at the loss. The purple loop updates every learned layer.', '''
print('X:', tuple(X.shape), 'y:', tuple(y.shape))
''')

step('lookup', 'An ID selects one embedding row', '3. The MLP forward pass',
     'The embedding table has C=10 rows and d=4 columns. ID 7, found, selects one four-number row. The four coordinates below are learned features with no assigned word meanings.', '''
embedding_table = mlp.token_embedding.weight
found_id = vocab.stoi['found']
found_vector = embedding_table[found_id]
assert found_vector.shape == (4,)
''', focus=('embedding',))

step('embedding-batch', 'The whole batch becomes a 2 × 4 × 4 tensor', '3. The MLP forward pass',
     'Each of the two examples has four token slots. Each slot receives four embedding coordinates. PAD uses the fixed zero row. The same token ID selects the same row wherever it occurs.', '''
E_mlp = mlp.token_embedding(X)
assert E_mlp.shape == (2, 4, 4)
assert E_mlp[0, 0].eq(0).all()
assert torch.equal(E_mlp[0, 2], E_mlp[1, 1])  # lily
''', focus=('embedding',))

step('flatten', 'Flatten the context axis, keep the batch axis', '3. The MLP forward pass',
     'Each example becomes one ordered row of 4×4=16 numbers. We never join example 0 to example 1. Swapping token positions changes which input connections receive each embedding.', '''
flat = E_mlp.flatten(start_dim=1)
assert flat.shape == (2, 16)
assert torch.equal(flat[1, :4], E_mlp[1, 0])
''', focus=('flatten',))

step('hidden-affine', 'One hidden neuron combines all sixteen inputs', '3. The MLP forward pass',
     'In row-vector notation the hidden matrix is 16×8. PyTorch stores Linear weights as 8×16. A neuron multiplies all sixteen input coordinates by its weights, sums them and adds one bias.', '''
pre_hidden = mlp.hidden_layer(flat)
terms = flat[1] * mlp.hidden_layer.weight[0]
manual_hidden = terms.sum() + mlp.hidden_layer.bias[0]
assert torch.allclose(manual_hidden, pre_hidden[1, 0])
''', focus=('hidden',))

step('relu', 'ReLU keeps positive activations', '3. The MLP forward pass',
     'ReLU acts separately on each hidden coordinate: max(0,z). It keeps the tensor shape at 2×8. The nonlinearity lets the network learn more than a single affine mapping.', '''
hidden_mlp = torch.relu(pre_hidden)
assert hidden_mlp.shape == (2, 8)
assert hidden_mlp.ge(0).all()
''', focus=('hidden',))

step('vocab-head', 'Eight hidden numbers produce ten word scores', '3. The MLP forward pass',
     'Every vocabulary item gets a logit, including words that are not the target. The output matrix is 8×10 in row-vector notation. Logits can be negative and need not sum to one.', '''
logits_mlp = mlp.vocab_head(hidden_mlp)
assert logits_mlp.shape == (2, 10)
assert torch.allclose(logits_mlp, mlp(X))
''', focus=('logits',))

step('attention-map', 'The full attention training map', '4. The attention forward pass',
     'The data, targets and loss stay the same. Attention changes how context reaches the hidden prediction layer. This one-layer implementation computes only the final query for each window.', '''
assert X.shape == (2, 4) and y.shape == (2,)
print('Both architectures predict the same two targets:', y.tolist())
''', kind='attention')

step('positions', 'Add a position vector to each token vector', '4. The attention forward pass',
     'The position table has four rows, one for each context slot. Token and position vectors both have width four, so addition preserves the width. PAD keys will be masked even though position addition can make their input row nonzero.', '''
positions = torch.arange(w)
token_rows = attention.token_embedding(X)
position_rows = attention.position_embedding(positions)
E = token_rows + position_rows[None, :, :]
assert E.shape == (2, 4, 4)
''', kind='attention', focus=('embedding',))

step('query', 'The final known token supplies one query', '4. The attention forward pass',
     'For example 1 the final known token is a. Its four-number input row maps to three query coordinates. The unseen target red does not take part in this multiplication.', '''
q = attention.W_Q(E[:, -1:, :])
query_terms = E[1, -1] * attention.W_Q.weight[0]
assert q.shape == (2, 1, 3)
assert torch.allclose(query_terms.sum(), q[1, 0, 0])
''', kind='attention', focus=('qkv',))

step('keys', 'Each source has a matching key', '4. The attention forward pass',
     'All four source rows map to keys of width three. Query and key widths match because we will take their dot products. The same W_K applies at every source position and in both examples.', '''
K = attention.W_K(E)
assert K.shape == (2, 4, 3)
print('Row-vector W_K shape:', tuple(attention.W_K.weight.T.shape))
''', kind='attention', focus=('qkv','scores'))

step('values', 'Each source also has information to send', '4. The attention forward pass',
     'W_V maps each source to a two-number value. These coordinates carry information into the weighted message. They do not enter the query-key dot product.', '''
V = attention.W_V(E)
assert V.shape == (2, 4, 2)
print('Row-vector W_V shape:', tuple(attention.W_V.weight.T.shape))
''', kind='attention', focus=('qkv','message'))

step('scores', 'Compare the query with all four keys', '4. The attention forward pass',
     'Each score is a three-term dot product divided by √3. The result has one row of four source scores per example. These are source-match scores, not vocabulary logits.', '''
raw_scores = q @ K.transpose(-2, -1)
scores = raw_scores / math.sqrt(d_k)
dot_terms = q[1, 0] * K[1, 2]
assert torch.allclose(dot_terms.sum(), raw_scores[1, 0, 2])
assert scores.shape == (2, 1, 4)
''', kind='attention', focus=('scores',))

step('mask', 'Padding must receive zero attention weight', '4. The attention forward pass',
     'Example 0 has PAD in its first slot, so that score becomes −∞. All slots of example 1 are real tokens. Since these are final queries, their windows contain no future columns.', '''
pad_mask = X[:, None, :].eq(vocab.pad_id)
masked_scores = scores.masked_fill(pad_mask, float('-inf'))
assert torch.isneginf(masked_scores[0, 0, 0])
''', kind='attention', focus=('weights',))

step('attention-softmax', 'Softmax turns four source scores into weights', '4. The attention forward pass',
     'Subtract the largest score, exponentiate and divide by the row sum. This softmax runs over context slots. Later, a different softmax will run over vocabulary items.', '''
source_exp = (masked_scores - masked_scores.amax(-1, keepdim=True)).exp()
A = source_exp / source_exp.sum(-1, keepdim=True)
assert torch.allclose(A, masked_scores.softmax(-1))
assert A[0, 0, 0] == 0 and torch.allclose(A.sum(-1), torch.ones(2, 1))
''', kind='attention', focus=('weights',))

step('mix', 'Each weight scales a complete value row', '4. The attention forward pass',
     'For example 1, multiply each two-number value by its source weight and add the four contributions. The result is one two-number message. Weighting does not change the value width.', '''
contributions = A[1, 0, :, None] * V[1]
message = A @ V
assert message.shape == (2, 1, 2)
assert torch.allclose(contributions.sum(0), message[1, 0])
''', kind='attention', focus=('message',))

step('output-map', 'Project two message coordinates into four', '4. The attention forward pass',
     'The message has width two, but the original input has width four. The learned W_O is 2×4 in row-vector notation. Every output coordinate can combine both message coordinates.', '''
update = attention.W_O(message).squeeze(1)
assert update.shape == (2, 4)
assert torch.allclose(message[1, 0] @ attention.W_O.weight.T, update[1])
''', kind='attention', focus=('projection',))

step('residual', 'Add the update to the original final row', '4. The attention forward pass',
     'The residual addition now combines two vectors with the same width. The updated row enters an eight-unit ReLU layer and then the ten-class vocabulary head, just as in the MLP path.', '''
final = E[:, -1, :] + update
hidden_att = torch.relu(attention.hidden_layer(final))
logits_att = attention.vocab_head(hidden_att)
assert torch.allclose(logits_att, attention(X), atol=1e-6)
assert torch.allclose(logits_att, attention.forward_details(X)['logits'], atol=1e-6)
''', kind='attention', focus=('residual','readout','hidden','logits'), check=('Is the two-number message already a prediction?', 'No. W_O maps it to four coordinates, the residual adds the original row, and the hidden vocabulary head produces ten logits.'))

step('word-softmax', 'Ten logits become ten next-token probabilities', '5. Loss and learning',
     'For the MLP example with target red, apply vocabulary softmax. The denominator includes all ten classes. These values come from the seeded model before training. Even a correct guess here can occur by chance.', '''
word_exp = (logits_mlp[1] - logits_mlp[1].max()).exp()
p = word_exp / word_exp.sum()
assert torch.allclose(p, logits_mlp[1].softmax(-1))
assert torch.allclose(p.sum(), torch.tensor(1.0))
''', focus=('logits','loss'))

step('target', 'The text gives the target; the model gives a guess', '5. Loss and learning',
     'Argmax returns the largest-probability vocabulary ID. The target remains red because the corpus contains red after this prefix. Training does not replace that observed target with the model’s guess.', '''
guess_id = int(p.argmax())
target_id = int(y[1])
print('Guess:', words[guess_id], 'target:', words[target_id])
print('Probability of target:', float(p[target_id].detach()))
''', focus=('logits','loss'))

step('loss', 'Two targets give two losses, then one batch mean', '5. Loss and learning',
     'For each example, cross-entropy is −log of the probability assigned to its observed target. PyTorch accepts raw logits and performs log-softmax internally. The optimizer uses the mean across B=2 examples.', '''
losses = F.cross_entropy(logits_mlp, y, reduction='none')
loss = losses.mean()
manual_loss = -logits_mlp.log_softmax(-1)[torch.arange(B), y]
assert torch.allclose(losses, manual_loss)
''', focus=('loss',))

step('gradient', 'Backward computes a direction for every parameter', '5. Loss and learning',
     'For vocabulary weight W[j,r], the batch-mean gradient is the mean of (p_r − 1[y=r]) × hidden_j. Here we inspect the weight feeding the red logit from hidden unit 0.', '''
mlp.zero_grad(set_to_none=True)
loss.backward()
manual_grad = ((logits_mlp.softmax(-1)[:, 9] - y.eq(9).float()) * hidden_mlp[:, 0]).mean()
grad = mlp.vocab_head.weight.grad[9, 0]
assert torch.allclose(grad, manual_grad)
''', focus=('backward',))

step('update', 'One optimizer step changes the stored weights', '5. Loss and learning',
     'For this arithmetic demonstration we use SGD with learning rate 0.1: new weight = old weight − 0.1×gradient. The measured TinyStories runs use validation-tuned AdamW instead.', '''
old_weight = mlp.vocab_head.weight[9, 0].detach().clone()
before_loss = float(loss.detach())
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)
optimizer.step()
new_weight = mlp.vocab_head.weight[9, 0].detach().clone()
after_loss = float(F.cross_entropy(mlp(X), y).detach())
assert torch.allclose(new_weight, old_weight - 0.1*grad)
''', focus=('optimizer','parameters'))

step('training-loop', 'Many batches, with validation between checkpoints', '5. Loss and learning',
     'The actual trainer samples 512 training windows per step, computes the loss, backpropagates and updates parameters. Validation selects the best saved checkpoint. The test set does not choose a checkpoint.', '''
benchmark = evidence['benchmark']
print('Maximum steps:', 6000, 'batch size:', 512)
print('Maximum target presentations:', 6000*512)
print('Checkpoint rule: best validation loss')
''', kind='attention', focus=('loss','backward','optimizer','parameters'))

step('evaluation', 'Held-out scoring and free-running generation', '6. Inference',
     'Held-out scoring uses observed X,y pairs and records loss. Generation has a prompt but no observed next target. Both freeze parameters. Neither calls backward or optimizer.step.', '''
mlp.eval()
with torch.inference_mode():
    frozen_loss = F.cross_entropy(mlp(X), y)
print('Illustrative scoring loss:', float(frozen_loss))
''', focus=('loss',))

step('mlp-inference', 'The complete MLP generation loop', '6. Inference',
     'Load the same vocabulary and trained parameters. Prepare the latest window, predict, choose a token and append it. Stop on EOS or the generation budget. The only changing state is the history.', '''
print('Generation uses B=1; the parameters stay fixed.')
''')

step('attention-inference', 'Attention uses the same generation loop', '6. Inference',
     'The new final token supplies the next query. This notebook recomputes the current window and has no KV cache. Position indices describe the slots in that current window.', '''
attention.eval()
print('Final query only; all available keys and values; no KV cache.')
''', kind='attention')

step('prompt', 'Prepare a prompt using the same rules', '6. Inference',
     'Use the training tokenizer and vocabulary on the unfinished prompt. Prepend BOS once, and leave EOS out because generation has not finished.', '''
prompt = 'Lily found'
prompt_ids = vocab.encode_tokens(tokenize(prompt), boundaries=False)
history = [vocab.bos_id] + prompt_ids
assert history == [1, 8, 7]
''', kind='attention', focus=('prompt','tokenize'))

step('prompt-window', 'The prompt fills the same four input slots', '6. Inference',
     'Keep the most recent w=4 IDs from the history. This prompt has only three IDs including BOS, so one PAD ID fills the unused slot on the left.', '''
kept = history[-w:]
context_ids = [vocab.pad_id] * (w-len(kept)) + kept
inference_X = torch.tensor([context_ids])
assert inference_X.tolist() == [[0, 1, 8, 7]]
''', kind='attention', focus=('windows',))

step('generation-logits', 'A frozen model scores the next token', '6. Inference',
     'The MLP reads the prepared window and returns ten vocabulary scores. inference_mode disables gradient tracking for this forward pass, and no optimizer updates the parameters.', '''
with torch.inference_mode():
    next_logits = mlp(inference_X)[0]
assert next_logits.shape == (C,)
''', focus=('logits',))

step('generation-probabilities', 'Special input tokens are excluded from generation', '6. Inference',
     'Set the scores of PAD, BOS and UNK to negative infinity before softmax. Their probabilities become zero, while EOS remains available as a stopping token.', '''
blocked_ids = [vocab.pad_id, vocab.bos_id, vocab.unk_id]
with torch.inference_mode():
    next_logits[blocked_ids] = float('-inf')
    next_p = next_logits.softmax(-1)
assert next_p[blocked_ids].eq(0).all()
assert torch.allclose(next_p.sum(), torch.tensor(1.0))
''', focus=('logits',))

step('decode', 'Greedy decoding and sampling make different choices', '6. Inference',
     'Greedy chooses the ID with the highest probability, while sampling uses the cumulative probabilities (CDF). The fixed draw 0.8 selects the first CDF entry at least 0.8, making this example reproducible.', '''
greedy_id = int(next_p.argmax())
cdf = next_p.cumsum(0)
sample_id = int(torch.searchsorted(cdf, torch.tensor(0.8)))
''', focus=('logits',))

step('generation-append', 'A selected token extends the history unless it is EOS', '6. Inference',
     'Use the greedy ID for this demonstration, and check whether it is EOS before changing the history. If it is EOS, stop generation, otherwise append it and prepare the next window.', '''
history_before_choice = history.copy()
chosen = greedy_id
if chosen != vocab.eos_id:
    history.append(chosen)
assert history_before_choice == [1, 8, 7]
''', focus=('windows',))

step('append', 'The chosen token becomes input on the next step', '6. Inference',
     'Repeat window preparation, scoring and token selection, stopping on EOS or after four steps. This table repeats generation from the original prompt, with all parameters frozen.', '''
history = history_before_choice.copy()
frozen = {n:p.detach().clone() for n,p in mlp.named_parameters()}
generation_trace = []
with torch.inference_mode():
    for generation_step in range(4):
        kept = history[-w:]
        x_next = torch.tensor([[vocab.pad_id]*(w-len(kept)) + kept])
        z_next = mlp(x_next)[0]
        z_next[[0, 1, 3]] = float('-inf')
        chosen = int(z_next.argmax())
        generation_trace.append((x_next[0].tolist(), chosen))
        if chosen == vocab.eos_id: break
        history.append(chosen)
assert all(torch.equal(p, frozen[n]) for n,p in mlp.named_parameters())
''', focus=('windows',))

step('benchmark', 'The trained TinyStories comparison', '7. The real experiment',
     'The saved experiment uses the same documents, vocabulary, targets and 64-token context for both models. Each model has validation-tuned optimizer settings. These are three-seed test results, separate from our ten-token arithmetic example.', '''
comparison = benchmark['aggregate']
for name in ['mlp', 'attention']:
    stats = comparison[name]['test_perplexity']
    print(name, stats['mean'], '+/-', stats['sample_std'])
''', kind='attention')

step('next', 'The same steps in the runnable notebooks', '7. The real experiment',
     'This notebook executes the small calculation behind every figure. Notebook 1 loads the corpus and trains the MLP. Notebook 3 trains attention. Notebook 4 checks the saved comparison and studies representations.', '''
assert B == 2 and w == 4 and C == 10
print('Toy example: 7 targets; worked batch: 2 examples.')
print('Real experiment: 969,160 training windows; context 64; vocabulary 4,000.')
''')


BLUE, PURPLE, ORANGE, TEAL, RED, GREEN = '#245EDB','#8B2CDE','#AA4E08','#0F766E','#BE123C','#147737'
INK, MUTED, LINE, PAPER = '#14171F','#4A5160','#D9DFE9','#F7F8FA'


class Figure:
    def __init__(self, title, width=1160, height=310):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}" style="font-family:Avenir Next,Segoe UI,sans-serif;background:{PAPER}"><title>{escape(title)}</title>']

    def text(self, x, y, value, color=INK, size=26, weight=500, anchor='start', mono=False):
        family = ' font-family="ui-monospace,SFMono-Regular,monospace"' if mono else ''
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{family}>{escape(str(value))}</text>')

    def line(self, x1,y1,x2,y2,color=LINE):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="2"/>')

    def rect(self,x,y,w,h,fill='#fff',stroke=LINE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>')

    def table(self, headers, rows, *, x=20,y=15,width=1120,row_h=42, widths=None, colors=None, size=24):
        widths = widths or [width/len(headers)]*len(headers)
        cols = [x]
        for w in widths: cols.append(cols[-1]+w)
        self.rect(x,y,sum(widths),(len(rows)+1)*row_h)
        for j, head in enumerate(headers):
            self.text(cols[j]+12,y+row_h*.69,head,MUTED,size-1,600)
        self.line(x,y+row_h,x+sum(widths),y+row_h)
        for i,row in enumerate(rows):
            for j,value in enumerate(row):
                color=colors[j] if colors else INK
                self.text(cols[j]+12,y+(i+1)*row_h+row_h*.69,value,color,size,500)
            if i<len(rows)-1:self.line(x,y+(i+2)*row_h,x+sum(widths),y+(i+2)*row_h)

    def finish(self): return ''.join(self.parts)+'</svg>'


def num(value):
    import math
    v=float(value.detach() if hasattr(value, 'detach') else value)
    return '−∞' if v == -math.inf else f'{v:.3f}'


def vector(values): return '[ '+', '.join(num(v) for v in values)+' ]'


def render_figure(stage, ns):
    """All numeric text comes from tensors calculated by the matching code cell."""
    k=stage['id'];f=Figure(stage['title']);words=ns.get('words',[])
    decode=lambda ids:' '.join(words[int(i)].replace('<','').replace('>','') for i in ids)
    X=ns.get('X'); row_tokens=lambda b:[words[int(i)] for i in X[b]]
    if k in {'mlp-map','attention-map','mlp-inference','attention-inference'}:
        return pipeline_svg(stage['kind'],'inference' if 'inference' in k else 'training')
    if k=='data':
        f.text(25,50,'TinyStories · source row 12400',BLUE,27,600)
        f.text(25,115,'“Once upon a time, there lived a little bunny.”',size=35)
        f.text(25,183,'One excerpt from a complete story; 6,000 documents in this subset.',size=27)
        f.text(25,250,'First, look at whole stories. Then we will build training pairs.',MUTED,25)
    elif k=='story-complete':
        f=Figure(stage['title'],height=350)
        story=ns['short_story']
        f.text(25,35,f"TinyStories · source row {story['row_idx']} · complete text",BLUE,26,600)
        f.text(25,77,f"{ns['short_words']} words · {ns['short_tokens']} tokens",TEAL,27,600)
        for i,line in enumerate(textwrap.wrap(story['text'],width=76)):
            f.text(25,135+i*42,line,size=29)
        f.text(25,332,'Everything above is one document, from the opening to “The end.”',MUTED,24)
    elif k=='story-excerpts':
        f=Figure(stage['title'],height=390)
        f.line(580,12,580,337)
        for j,(story,(word_count,token_count)) in enumerate(zip(ns['long_stories'],ns['story_lengths'])):
            x=25+590*j
            f.text(x,33,f"Source row {story['row_idx']}",BLUE,26,600)
            f.text(x,72,f'{word_count} words · {token_count} tokens',TEAL,26,600)
            # Exact opening/closing sentences; bracketed ellipsis marks omissions.
            text=story['text'];opening=text.split('. ',1)[0]+'.'
            ending=text.rsplit('. ',1)[1]
            lines=textwrap.wrap(opening,width=39)+['[…]']+textwrap.wrap(ending,width=39)
            for i,line in enumerate(lines):
                f.text(x,120+i*33,line,MUTED if line=='[…]' else INK,26)
            paragraphs=len(text.split('\n\n'))
            f.text(x,329,f'Full story: {paragraphs} paragraphs',MUTED,23)
        stats=ns['train_lengths']
        f.text(25,378,f"Training stories: {stats['min']}–{stats['max']} tokens; median {stats['median']} tokens.",MUTED,24)
    elif k=='split':
        f.table(['Split','Whole stories','Used for'],[(s.capitalize(),f"{ns['audit']['documents'][s]:,}",use) for s,use in [('train','Parameter and vocabulary learning'),('validation','Settings and checkpoint selection'),('test','Final held-out measurement')]],widths=[220,260,640],row_h=65)
    elif k=='sentence':
        f.text(30,100,'Lily found a red ball.',BLUE,50,600)
        f.text(30,190,'A complete authored example, including its final punctuation.',MUTED,29)
    elif k=='tokenization-intro':
        f.text(25,60,'Does “next token” always mean “next word”?',size=36,weight=600)
        f.text(25,150,ns['tokenization_text'],BLUE,52,600)
        f.text(25,235,'The same text can become different sequences of tokens.',MUTED,28)
    elif k=='tokenization-choices':
        f=Figure(stage['title'],height=350)
        rows=[(name, ' '.join('['+piece+']' for piece in parts),len(parts))
              for name,parts in ns['tokenization_choices'].items()]
        f.table(['Choice','Tokens for '+ns['tokenization_text'],'Count'],rows,
                widths=[310,620,190],row_h=58,colors=[INK,BLUE,TEAL],size=26)
        f.text(25,287,'Context length counts tokens. The same width can cover different amounts of text.',size=25)
        f.text(25,332,'The subword split is illustrative. Actual splits depend on the tokenizer.',MUTED,24)
    elif k=='tokenization-rules':
        f=Figure(stage['title'],height=350)
        f.text(25,42,'Text: '+ns['tokenizer_probe'],size=31)
        f.text(25,100,' '.join('['+piece+']' for piece in ns['probe_tokens']),BLUE,29)
        f.line(25,126,1135,126)
        f.text(25,176,'Lily and lily share a token. Spaces separate words but are not tokens here.',size=26)
        f.text(25,225,"The apostrophe stays inside can't. The digits in 12 stay together.",size=26)
        f.text(25,274,'The exclamation mark is a separate token, just like the full stop in our story.',size=26)
        f.text(25,332,'English teaching tokenizer. Original case and spacing are lost.',MUTED,24)
    elif k=='tokenize':
        f.text(25,50,'Text: Lily found a red ball.',size=32)
        f.text(25,110,'Tokens:',BLUE,27,600)
        for j,t in enumerate(ns['pieces']):
            f.rect(160+j*155,75,140,70,stroke=BLUE);f.text(230+j*155,120,t,BLUE,30,anchor='middle')
        f.text(25,220,'5 word tokens + 1 punctuation token = 6 ordinary tokens',size=32)
    elif k=='vocabulary':
        for group in range(2):
            rows=[(i,words[i]) for i in range(group*5,(group+1)*5)]
            f.table(['ID','Vocabulary item'],rows,x=20+570*group,widths=[120,410],row_h=46,colors=[BLUE,INK])
    elif k=='special':
        f.table(['ID','Token','Role'],[(0,'PAD','Fill unused input slots'),(1,'BOS','First context token of a story'),(2,'EOS','Predict that the story ends'),(3,'UNK','Represent a missing vocabulary item')],widths=[100,170,850],row_h=55)
    elif k=='boundaries':
        for j,i in enumerate(ns['ids']):
            x=20+j*140;f.rect(x,55,128,120,stroke=BLUE)
            f.text(x+64,95,words[i],BLUE,25,600,'middle');f.text(x+64,148,i,BLUE,32,500,'middle')
        f.text(25,250,'8 IDs total. Predict each ID after BOS: 7 supervised targets.',size=30)
    elif k=='story-indices':
        f.table(['Story position']+list(range(len(ns['ids']))),
                [('Token',*[words[i] for i in ns['ids']]),('Token ID',*ns['ids'])],
                widths=[216]+[113]*8,row_h=60,size=26)
        f.text(25,257,'Input: positions 0, 1, 2, 3',BLUE,30,600)
        f.text(640,257,'Target: position 4, ID 9 (red)',RED,30,600)
    elif k=='one-pair':
        f.text(25,45,'context_ids: one input x',BLUE,30,600)
        f.text(25,120,decode(ns['context_ids']),BLUE,38,600)
        f.text(25,180,str(ns['context_ids']),BLUE,30,mono=True)
        f.text(730,45,'target_id: one answer y',RED,30,600)
        f.text(730,120,words[ns['target_id']],RED,42,600)
        f.text(730,180,'ID '+str(ns['target_id']),RED,30)
        f.text(25,270,'ids[0:4] contains four IDs',BLUE,29)
        f.text(730,270,'ids[4] is one ID',RED,29)
    elif k=='pair-lists':
        f.text(25,65,'contexts = []',BLUE,38,600,mono=True)
        f.text(25,130,'Will hold one input list per example',BLUE,29)
        f.text(650,65,'targets = []',RED,38,600,mono=True)
        f.text(650,130,'Will hold one answer per example',RED,29)
        f.text(25,245,'Stored so far: 0 input rows and 0 targets',size=31)
    elif k in {'pair-first','pair-second'}:
        f=Figure(stage['title'],height=225)
        visible=ns['visible'];padding=ns['w']-len(visible)
        f.text(25,40,f"t = {ns['t']}    visible = {visible}    ({decode(visible)})",BLUE,29,600)
        known='known token ID' if len(visible)==1 else 'known token IDs'
        f.text(25,98,f"{padding} PAD IDs + {len(visible)} {known} = {ns['w']} input slots",size=28)
        f.text(25,159,'context_ids = '+str(ns['context_ids']),BLUE,30,mono=True)
        f.text(750,159,'target_id = '+str(ns['target_id']),RED,30,mono=True)
        f.text(25,213,decode(ns['context_ids']),BLUE,27)
        f.text(750,213,words[ns['target_id']],RED,27)
    elif k in {'pair-append','pair-append-second'}:
        f.text(25,42,'contexts: a list of input lists',BLUE,29,600)
        f.text(735,42,'targets: a list of IDs',RED,29,600)
        # Literal nested lists expose what append adds, including the outer brackets.
        f.text(25,100,'[',BLUE,31,mono=True)
        f.text(735,100,str(ns['targets']),RED,31,mono=True)
        for row,context in enumerate(ns['contexts']):
            y=148+row*51
            f.text(60,y,str(context)+(',' if row<len(ns['contexts'])-1 else ''),BLUE,31,mono=True)
            f.text(460,y,f'row {row}',MUTED,26)
            f.text(735,y,f"targets[{row}] = {ns['targets'][row]} ({words[ns['targets'][row]]})",RED,27)
        f.text(25,148+len(ns['contexts'])*51,']',BLUE,31,mono=True)
        count=len(ns['contexts']);plural='' if count==1 else 's'
        f.text(25,296,f"{count} input row{plural}, {count} target{plural}. Same index means the same example.",size=27)
    elif k=='pairs-loop':
        f=Figure(stage['title'],height=135)
        f.table(['Already stored','This loop adds','Total pairs'],
                [('t = 1, 2','t = 3, 4, 5, 6, 7',len(ns['targets']))],
                widths=[360,500,260],row_h=53,size=29)
    elif k in {'windows-first','windows-last'}:
        start,end=(0,4) if k=='windows-first' else (4,7)
        rows=[(j,decode(ns['contexts'][j]),words[ns['targets'][j]]) for j in range(start,end)]
        f.table(['Stored row','Input: exactly four slots','Next target'],rows,widths=[170,730,220],row_h=53,colors=[MUTED,BLUE,RED])
        if k=='windows-last':f.text(25,285,'Every ordinary token plus EOS is predicted once: N = 6 + 1 = 7.',size=27)
    elif k=='pairs-tensors':
        f=Figure(stage['title'],height=360)
        rows=[(r,str(ns['all_X'][r].tolist()),int(ns['all_y'][r]),words[int(ns['all_y'][r])])
              for r in range(ns['N'])]
        f.table(['Row','contexts[r] = all_X[r]','targets[r] = all_y[r]','Word'],rows,
                widths=[100,480,340,200],row_h=39,size=26,colors=[MUTED,BLUE,RED,RED])
        f.text(25,350,'all_X: 7 rows × 4 IDs',BLUE,28,600)
        f.text(610,350,'all_y: 7 targets     N = 7',RED,28,600)
    elif k in {'counts-story','counts-train'}:
        if k=='counts-story':
            a,b,total=ns['ordinary_tokens'],1,ns['examples_in_story']
            labels=['Ordinary tokens','EOS target','Training examples']
        else:
            a,b,total=ns['train_tokens'],ns['train_stories'],ns['train_examples']
            labels=['Ordinary tokens','One EOS per story','Training examples']
        for x,label,value in zip([25,420,820],labels,[a,b,total]):
            f.text(x,65,label,MUTED,28,600)
            f.text(x,155,f'{value:,}',TEAL if x==820 else BLUE,52,600)
        f.text(325,155,'+',size=48)
        f.text(720,155,'=',size=48)
        f.text(25,265,'Every ordinary token and each story ending supplies one target.',size=30)
    elif k=='counts':
        rows=[(s.capitalize(),f"{ns['audit']['oov'][s]['tokens']:,}",f"{ns['audit']['documents'][s]:,}",f"{ns['window_counts'][s]:,}") for s in ['train','validation','test']]
        f.table(['Split','Ordinary tokens','One EOS / story','Total targets'],rows,widths=[190,290,310,330],row_h=57)
        f.text(25,295,'For one story: 6 + 1 = 7. Across training stories: 964,338 + 4,822 = 969,160.',MUTED,25)
    elif k=='context':
        f.table(['w','Visible suffix','Target count / story'],[(i,decode(ns['contexts_by_width'][i]),7) for i in [2,4,6]],widths=[100,760,260],row_h=67,colors=[INK,BLUE,RED])
    elif k=='batch':
        f.table(['Data row','Batch row','X: input IDs','Input tokens','y: ID','Target token'],
                [(int(ns['selected'][i]),i,str(X[i].tolist()),' '.join(row_tokens(i)),int(ns['y'][i]),words[int(ns['y'][i])]) for i in range(len(X))],
                widths=[125,125,250,340,100,180],row_h=65,colors=[MUTED,INK,BLUE,BLUE,RED,RED])
        f.text(25,250,'Dataset rows 2 and 3 become batch rows 0 and 1.',MUTED,27)
        f.text(25,300,'PAD fills an empty slot. BOS marks the start of the story.',MUTED,27)
    elif k=='batch-ids':
        f=Figure(stage['title'],height=340)
        f.text(25,40,'X: input token IDs',BLUE,30,600)
        f.table(['Batch row','Slot 1','Slot 2','Slot 3','Slot 4'],
                [(i,*X[i].tolist()) for i in range(len(X))],
                y=65,widths=[160,125,125,125,125],row_h=52,colors=[MUTED,BLUE,BLUE,BLUE,BLUE],size=27)
        f.text(25,275,'X.shape = (2, 4)',BLUE,31,600)
        f.text(25,320,'2 examples × 4 input slots',MUTED,27)
        f.text(775,40,'y: next-token IDs',RED,30,600)
        f.text(775,163,str(ns['y'].tolist()),RED,44,600,mono=True)
        f.text(775,275,'y.shape = (2,)',RED,31,600)
        f.text(775,320,'2 targets, one per example',MUTED,27)
    elif k=='batches':
        f.table(['Sequential batch','Example indices','Actual B'],[(1,'0, 1',2),(2,'2, 3',2),(3,'4, 5',2),(4,'6',1)],widths=[300,470,350],row_h=54)
    elif k=='shapes':
        f.table(['Symbol','Value','Meaning'],[('B / w','2 / 4','examples per batch / context slots'),('C / d','10 / 4','vocabulary items / embedding coordinates'),('h','8','hidden units in the prediction head'),('dₖ / dᵥ','3 / 2','matching coordinates / value coordinates')],widths=[200,190,730],row_h=54)
    elif k=='lookup':
        rows=[(words[i],i,*[num(v) for v in ns['embedding_table'][i]]) for i in [0,7,8,9]]
        f.table(['Token','ID','coord. 1','coord. 2','coord. 3','coord. 4'],rows,widths=[200,100,205,205,205,205],row_h=55,colors=[INK,BLUE,BLUE,BLUE,BLUE,BLUE])
    elif k=='embedding-batch':
        for b in range(2):
            f.text(20+580*b,30,f'Example {b}: {decode(X[b])}',BLUE,24,600)
            f.table(['Token','c1','c2','c3','c4'],[(t,*[num(v) for v in ns['E_mlp'][b,j]]) for j,t in enumerate(row_tokens(b))],x=20+580*b,y=50,widths=[150,95,95,95,95],row_h=45,size=21)
    elif k=='flatten':
        f.text(25,40,'Example 1: concatenate in slot order',BLUE,28,600)
        for j in range(4):
            f.text(25,105+j*49,f'{row_tokens(1)[j]:>6}  {vector(ns["E_mlp"][1,j])}',BLUE,28,mono=True)
        f.text(735,135,'One row:',size=30);f.text(735,190,'4 × 4 = 16 inputs',BLUE,33,600)
        f.text(735,250,'Batch shape [2, 16]',MUTED,26)
    elif k=='hidden-affine':
        f.text(25,40,'Hidden neuron 0, example 1',TEAL,29,600)
        f.table(['Input group','Sum of four weighted inputs'],[(f'Slot {j+1}: {row_tokens(1)[j]}',num(ns['terms'][j*4:(j+1)*4].sum())) for j in range(4)],y=65,widths=[590,530],row_h=40)
        f.text(25,300,f"Sum {num(ns['terms'].sum())} + bias {num(ns['mlp'].hidden_layer.bias[0])} = {num(ns['manual_hidden'])}",TEAL,27,600)
    elif k=='relu':
        for group in range(2):
            f.table(['Unit','Before ReLU','After ReLU'],[(j,num(ns['pre_hidden'][1,j]),num(ns['hidden_mlp'][1,j])) for j in range(group*4,(group+1)*4)],x=20+570*group,widths=[95,220,215],row_h=53,colors=[INK,MUTED,TEAL])
    elif k=='vocab-head':
        for group in range(2):
            f.table(['ID / word','Logit, example 1'],[(f'{i}  {words[i]}',num(ns['logits_mlp'][1,i])) for i in range(group*5,(group+1)*5)],x=20+570*group,widths=[280,250],row_h=46,colors=[INK,RED])
    elif k=='positions':
        f.table(['Final slot of example 1','c1','c2','c3','c4'],[('Token: a',*[num(v) for v in ns['token_rows'][1,-1]]),('Position: slot 4',*[num(v) for v in ns['position_rows'][-1]]),('Sum E',*[num(v) for v in ns['E'][1,-1]])],widths=[400,180,180,180,180],row_h=65)
    elif k=='query':
        f.text(25,45,'Final input E[1, −1]: '+vector(ns['E'][1,-1]),BLUE,28)
        f.text(25,105,'W_Q is 4 × 3 in row-vector notation',PURPLE,28)
        f.text(25,163,'First query coordinate: sum of four products',size=27)
        f.text(25,213,' + '.join(num(v) for v in ns['query_terms'])+' = '+num(ns['q'][1,0,0]),PURPLE,28)
        f.text(25,282,'Query q: '+vector(ns['q'][1,0]),PURPLE,32,600)
    elif k in {'keys','values'}:
        tensor=ns['K' if k=='keys' else 'V'];width=tensor.shape[-1];color=ORANGE if k=='keys' else TEAL
        f.table(['Source in example 1']+[f'coord. {j+1}' for j in range(width)],[(t,*[num(v) for v in tensor[1,j]]) for j,t in enumerate(row_tokens(1))],widths=[400]+[720/width]*width,row_h=55,colors=[INK]+[color]*width)
    elif k=='scores':
        f.text(25,40,'One match, source found: '+ ' + '.join(num(v) for v in ns['dot_terms']),RED,27)
        f.text(25,88,f"Dot product {num(ns['raw_scores'][1,0,2])} ÷ √3 = {num(ns['scores'][1,0,2])}",RED,30,600)
        f.table(['Source','BOS','lily','found','a'],[('Raw q·k',*[num(v) for v in ns['raw_scores'][1,0]]),('Scaled',*[num(v) for v in ns['scores'][1,0]])],y=140,widths=[260,215,215,215,215],row_h=50)
    elif k=='mask':
        f.table(['Example 0','PAD','BOS','lily','found'],[('Before mask',*[num(v) for v in ns['scores'][0,0]]),('After mask',*[num(v) for v in ns['masked_scores'][0,0]])],widths=[280,210,210,210,210],row_h=65,colors=[INK,RED,RED,RED,RED])
        f.text(25,280,'exp(−∞) = 0, so the padded source contributes zero weight.',RED,30)
    elif k=='attention-softmax':
        f.table(['Example 0','PAD','BOS','lily','found'],[('Shifted exponentials',*[num(v) for v in ns['source_exp'][0,0]]),('Normalized A',*[num(v) for v in ns['A'][0,0]])],widths=[330,198,198,198,196],row_h=65)
        f.text(25,275,f"Divide by {num(ns['source_exp'][0,0].sum())}; the normalized weights sum to 1.",RED,29)
    elif k=='mix':
        f.table(['Source','Weight','Value 1','Value 2','Contribution 1','Contribution 2'],[(t,num(ns['A'][1,0,j]),*[num(v) for v in ns['V'][1,j]],*[num(v) for v in ns['contributions'][j]]) for j,t in enumerate(row_tokens(1))],widths=[155,140,175,175,235,240],row_h=46,size=22)
        f.text(25,295,'Add the four contribution rows: message = '+vector(ns['message'][1,0]),TEAL,29,600)
    elif k=='output-map':
        f.text(25,40,'Message: '+vector(ns['message'][1,0]),TEAL,30)
        f.table(['W_O: value → model','c1','c2','c3','c4'],[(f'Value coordinate {j+1}',*[num(v) for v in ns['attention'].W_O.weight.T[j]]) for j in range(2)],y=80,widths=[400,180,180,180,180],row_h=52)
        f.text(25,292,'Update: '+vector(ns['update'][1]),GREEN,29,600)
    elif k=='residual':
        f.table(['Example 1','c1','c2','c3','c4'],[('Original final row',*[num(v) for v in ns['E'][1,-1]]),('Attention update',*[num(v) for v in ns['update'][1]]),('Updated final row',*[num(v) for v in ns['final'][1]])],widths=[400,180,180,180,180],row_h=55)
        f.text(25,295,'4 updated coordinates → 8 ReLU units → 10 vocabulary logits',GREEN,29,600)
    elif k in {'word-softmax','decode'}:
        probs=ns['p' if k=='word-softmax' else 'next_p'];cdf=ns.get('cdf')
        for group in range(2):
            rows=[]
            for i in range(group*5,(group+1)*5):
                rows.append((words[i],f'{float(probs[i].detach())*100:.2f}%',num(cdf[i]) if k=='decode' else num(ns['word_exp'][i])))
            f.table(['Word','Probability','CDF' if k=='decode' else 'exp(z − max)'],rows,x=20+570*group,widths=[160,185,185],row_h=40,size=22,colors=[INK,GREEN,MUTED])
        if k=='decode':
            f.text(25,297,'Greedy: '+words[ns['greedy_id']]+'; fixed sampling draw 0.8: '+words[ns['sample_id']],GREEN,26,600)
        else:
            f.text(25,297,'For each word: exp(z − max) ÷ '+num(ns['word_exp'].sum())+' = probability.',GREEN,26,600)
    elif k=='target':
        f.text(25,65,'Known input: BOS lily found a',BLUE,33)
        f.text(25,140,'Model’s largest probability: '+words[ns['guess_id']],GREEN,34,600)
        f.text(25,210,'Observed next token: red (ID 9)',RED,34,600)
        f.text(25,280,f"p(red) = {float(ns['p'][9].detach()):.4f}; the loss measures this observed target.",MUTED,27)
    elif k=='loss':
        probs=ns['logits_mlp'].softmax(-1)
        f.table(['Example','Target','Target probability','−log(probability)'],[(b,words[int(ns['y'][b])],num(probs[b,ns['y'][b]]),num(ns['losses'][b])) for b in range(2)],widths=[170,180,370,400],row_h=65)
        f.text(25,280,f"Batch loss = ({num(ns['losses'][0])} + {num(ns['losses'][1])}) / 2 = {num(ns['loss'])}",RED,30,600)
    elif k=='gradient':
        f.text(25,55,'Weight: hidden unit 0 → red logit',PURPLE,32,600)
        f.text(25,125,'For each example: (p(red) − target-is-red) × hidden[0]',size=29)
        f.text(25,198,'Average the two contributions because B = 2.',size=29)
        f.text(25,274,'Autograd = manual gradient = '+num(ns['grad']),PURPLE,35,600)
    elif k=='update':
        f.text(25,65,f"{num(ns['old_weight'])} − 0.1 × ({num(ns['grad'])}) = {num(ns['new_weight'])}",PURPLE,39,600)
        f.text(25,145,'old weight             learning rate × gradient          new weight',MUTED,24)
        f.text(25,225,f"On this batch: loss {ns['before_loss']:.3f} → {ns['after_loss']:.3f}",RED,32)
        f.text(25,285,'The step updates all trainable parameters, not just this one weight.',MUTED,26)
    elif k=='training-loop':
        f.table(['Stage','Data / action'],[('Training step','512 sampled windows; forward, loss, backward, AdamW'),('Validation','Evaluate frozen weights; save lower validation loss'),('Finish','Restore best validation checkpoint; score test once')],widths=[250,870],row_h=70,size=24)
    elif k=='evaluation':
        f.table(['Mode','Input','Observed target?','Parameter updates?'],[('Training','Training prefix','Yes','Yes'),('Held-out scoring','Test prefix','Yes','No'),('Generation','Growing prompt','No','No')],widths=[240,330,285,265],row_h=67,size=23)
    elif k=='prompt':
        f.text(25,50,'Prompt: '+ns['prompt'],size=35)
        f.text(25,125,'Token IDs: '+str(ns['prompt_ids']),BLUE,34)
        f.text(25,200,'With BOS: '+str(ns['history'])+'  (BOS lily found)',BLUE,32)
        f.text(25,275,'EOS is absent because the prompt is unfinished.',MUTED,29)
    elif k=='prompt-window':
        f.text(25,50,'Prompt: Lily found',size=35)
        f.text(25,120,'History: BOS lily found',BLUE,33)
        f.text(25,190,'Model input: [PAD BOS lily found] = [0, 1, 8, 7]',BLUE,31)
        f.text(25,265,'B = 1; w = 4; no target supplied; parameters are frozen.',MUTED,29)
    elif k=='generation-logits':
        for group in range(2):
            rows=[]
            for i in range(group*5,(group+1)*5):
                rows.append((f'{i}  {words[i]}',num(ns['next_logits'][i])))
            f.table(['ID / token','Logit'],
                    rows,x=20+570*group,widths=[280,250],row_h=46,
                    colors=[INK,RED])
    elif k=='generation-probabilities':
        f=Figure(stage['title'],height=225)
        selected=[ns['vocab'].pad_id,ns['vocab'].bos_id,ns['vocab'].unk_id,ns['vocab'].eos_id]
        f.table(['Token']+[words[i] for i in selected],
                [('Masked score',*[num(ns['next_logits'][i]) for i in selected]),
                 ('Probability',*[f"{float(ns['next_p'][i])*100:.2f}%" for i in selected])],
                widths=[260,215,215,215,215],row_h=54,size=28)
        f.text(25,212,'Softmax still uses all 10 vocabulary entries.',MUTED,27)
    elif k=='generation-append':
        before=ns['history_before_choice'];after=ns['history']
        f.text(25,50,'Chosen ID: '+str(ns['chosen'])+' ('+words[ns['chosen']]+')',GREEN,34,600)
        f.text(25,125,'Before: '+decode(before),BLUE,32)
        f.text(25,195,'After: '+decode(after),BLUE,32)
        f.text(25,270,'EOS: stop.' if ns['chosen']==ns['vocab'].eos_id else 'Continue with the newest four IDs as the next input.',MUTED,29)
    elif k=='append':
        f.table(['Step','Prepared input','Chosen token'],[(j+1,decode(x),words[token]) for j,(x,token) in enumerate(ns['generation_trace'])],widths=[140,760,220],row_h=53)
        if len(ns['generation_trace'])<4:f.text(25,280,'EOS ended this trace early.',MUTED,26)
    elif k=='benchmark':
        f.table(['Model','Test perplexity','Parameters'],[(name,f"{ns['comparison'][name]['test_perplexity']['mean']:.2f} ± {ns['comparison'][name]['test_perplexity']['sample_std']:.2f}",f"{EVIDENCE['benchmark']['parameter_counts'][name]:,}") for name in ['mlp','attention']],widths=[340,430,350],row_h=70)
        f.text(25,290,'39.1% lower perplexity for attention in this saved experiment.',GREEN,29,600)
    elif k=='next':
        f.table(['Companion','Purpose'],[('Notebook 1','Corpus preparation and MLP training'),('Notebook 3','Attention operations and training'),('Notebook 4','Measured comparison and representations'),('Notebook 5','Every calculation and figure in this walkthrough')],widths=[290,830],row_h=54)
    else:raise KeyError(k)
    return f.finish()


def show_figure(key, namespace):
    from IPython.display import SVG, HTML, display
    stage=next(s for s in STAGES if s['id']==key)
    display(SVG(render_figure(stage, namespace)))
    if stage['focus']:
        mode='inference' if stage['chapter'].startswith('6.') and key!='evaluation' else 'training'
        svg=pipeline_svg(stage['kind'],mode,stage['focus'])
        display(HTML('<details><summary>Locate this step on the complete map</summary>'+svg+'</details>'))


def initial_namespace():
    import torch, math
    import torch.nn.functional as F
    from wordlm import tokenize, SPECIAL_TOKENS, Vocabulary, FixedWindowMLP, CausalAttentionLM
    return dict(torch=torch, math=math, F=F, tokenize=tokenize,
                SPECIAL_TOKENS=SPECIAL_TOKENS, Vocabulary=Vocabulary,
                FixedWindowMLP=FixedWindowMLP, CausalAttentionLM=CausalAttentionLM,
                evidence=EVIDENCE, story_examples=STORY_EXAMPLES)

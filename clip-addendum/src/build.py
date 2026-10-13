"""CLIP addendum: applications and practical builds in the course visual style."""
from pathlib import Path
import json,sys,re
from html import escape
ROOT=Path(__file__).resolve().parents[1]
from reveal_markup import step,sequence
MEDIA=json.loads((ROOT/'media.json').read_text())
PIN='f9b8106ffb8308b4f48b76149d270f7c2e6b4918'
REPO=f'https://github.com/ARahim3/DigUp/tree/{PIN}'
SEARCH=f'https://github.com/ARahim3/DigUp/blob/{PIN}/Sources/DigUpKit/Searcher.swift'
MODEL='https://huggingface.co/google/embeddinggemma-2'
SPEECH='https://eg2.haku.sh/'
METHOD='https://x.com/TslShahir/status/2109660369429954631'
WER='https://x.com/TslShahir/status/2109732426591478238'
slides=[]
def link(url,label):return f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'
def arrow():return '<span class="arrow" aria-hidden="true">→</span>'
def flow(*items):return '<div class="flow">'+arrow().join(items)+'</div>'
def node(title,sub='',color='blue'):
 return f'<div class="node {color}"><b>{title}</b>'+ (f'<small>{sub}</small>' if sub else '')+'</div>'
def video(key):
 m=MEDIA[key]
 return f'''<figure class="demo"><div class="video-frame"><video controls playsinline preload="none" poster="{m['poster']}" aria-label="{m['author']} demonstration"><source src="{escape(m['video'])}" type="video/mp4"></video><img class="print-poster" src="{m['poster']}" alt="Frame at {m['poster_timestamp_seconds']} seconds from {m['author']}’s demo"></div><figcaption>{link(m['post'],'Watch the original X demo')} · {m['author']} · {round(m['duration'])} s <span class="online">· video needs internet</span></figcaption><p class="video-fallback" hidden>Video unavailable here. Use the original X demo link above.</p></figure>'''
def slide(id,title,body,take,notes,sources,titles=None,scope=''):
 n=len(slides)+1
 if titles:body=sequence('<div class="builds">'+body+'</div>',titles)
 slides.append(f'''<section class="slide" id="{id}" aria-labelledby="title-{id}"><header><p class="eyebrow">CLIP ADDENDUM · {'DIGUP' if n<=5 else 'SPEECH FROM EMBEDDINGS'}</p><h1 id="title-{id}">{title}</h1><p class="scope">{scope}</p></header><div class="visual">{body}</div><p class="takeaway" data-reveal-step="{len(titles) if titles else 1}">{take}</p><footer><span>{sources}</span><span>{n} / 9</span></footer><details class="notes"><summary>Teaching notes & sources</summary>{notes}</details></section>''')

slide('digup-demo','Can we find the moment, not just the file?',
 '<div class="split demo-split">'+video('digup')+'<div class="demo-prompts"><p class="lead">What do you remember?</p>'+step('<p><q>zebra in a video</q><small>Open the matching moment.</small></p>',2)+step('<p><q>where they talk about sleep</q><small>Open the audio segment.</small></p>',3)+step('<p><q>the clause about pets in the lease</q><small>Open the PDF page.</small></p>',4)+'</div></div>',
 'DigUp turns a shared embedding space into local file search.',
 '<p>Play Abdur Rahim’s 60-second demo first. Ask: what must be stored to jump to the right moment? These are the developer’s example queries, not results reproduced in class. The app is a native Apple Silicon Mac application using llama.cpp and Metal, not WebGPU. Our browser lab is a separate implementation of related ideas.</p><p>CLIP connected image and text representations. EmbeddingGemma 2 also supports audio and video inputs; it is a different pretrained model, not an extra modality added to our CLIP checkpoint.</p>',
 link(MEDIA['digup']['post'],'Abdur Rahim · X demo')+' · '+link(REPO,'DigUp source'),
 ['Describe what you remember','Find a video moment','Find an audio segment','Find a page'], 'AUTHOR DEMO · DigUp · 10 October 2026')

inputs='''<div class="index-inputs"><div><b class="blue">Photo</b><span>the picture</span></div><div><b class="purple">PDF</b><span>page / passage</span></div><div><b class="teal">Recording</b><span>overlapping 30 s windows</span></div><div><b class="amber">Video</b><span>sampled frames + audio</span></div></div>'''
store='''<div class="stored"><p class="label teal">SQLite index</p><table><thead><tr><th>Piece</th><th>Vector</th><th>Location</th></tr></thead><tbody><tr><td>video frame</td><td>[… 768 values …]</td><td>file + time</td></tr><tr><td>PDF page</td><td>[… 768 values …]</td><td>file + page</td></tr><tr><td>audio window</td><td>[… 768 values …]</td><td>file + time</td></tr></tbody></table></div>'''
slide('build-index','The index stores pieces, vectors and locations',
 '<div class="index-flow">'+inputs+step('<div>'+arrow()+node('EmbeddingGemma 2','frozen model<br>pool → project → normalize')+'</div>',2)+step('<div>'+arrow()+store+'</div>',3)+'</div>'+step('<div class="index-bottom"><b>First pass:</b> read the chosen folders. <span>After an edit: refresh the affected entries.</span></div>',4),
 'An embedding says what a piece is about. Its metadata says where it came from.',
 '<p>Diagram is a simplified source-grounded view. DigUp stores L2-normalized 768-dimensional vectors as float16 in SQLite, along with segment kind and location. Photos use visual content; screenshots also get OCR for keyword search. PDFs can supply page images and text passages. Audio uses overlapping 30-second windows; video uses sparse frames, shot selection and audio windows.</p><p>Indexing runs the pretrained model; it does not train it on your folder. Files must be indexed before semantic search can retrieve them. The app incrementally handles changes. A miss may be caused by unreadable or unsampled content, not just the embedding.</p>',
 link(REPO,'README & readers')+' · '+link(f'https://github.com/ARahim3/DigUp/blob/{PIN}/Sources/DigUpKit/IndexStore.swift','Stored schema'),
 ['Split files into searchable pieces','Encode each piece','Keep the vector and its source location','Refresh the index when files change'], 'SOURCE-GROUNDED SCHEMATIC · vectors are not model outputs shown here')

query='<div class="query"><q>zebra in a video</q><span>→ text encoder → normalize</span><b>q = [0.6, 0.8, 0]</b></div>'
matrix='''<table class="numbers"><thead><tr><th>Stored piece</th><th>coordinate 1</th><th>coordinate 2</th><th>coordinate 3</th><th>q · v</th></tr></thead><tbody><tr><th>wildlife.mp4 · 00:24</th><td>0.8</td><td>0.6</td><td>0</td><td class="score" data-reveal-step="3">0.96</td></tr><tr><th>lease.pdf · page 2</th><td>0</td><td>0.6</td><td>0.8</td><td class="score" data-reveal-step="3">0.48</td></tr><tr><th>notes.txt · passage 3</th><td>0</td><td>0</td><td>1</td><td class="score" data-reveal-step="3">0.00</td></tr></tbody></table>'''
slide('query-scores','The dot product you already know does the search',
 query+step(matrix,2)+step('<p class="calculation">0.6 × 0.8 + 0.8 × 0.6 + 0 × 0 = <b>0.96</b></p>',3)+step('<p class="plain-note">In the app: an N × 768 matrix times a 768-dimensional query → N cosine scores.</p>',4),
 'The file vectors are already stored. Only the new query needs encoding.',
 '<p>Every number and filename on this slide is illustrative. All four displayed vectors have norm 1. Multiplication is the same as CLIP’s unit-vector comparison, now over stored pieces. The real app uses 768 dimensions and scores the stored matrix in blocks. These three toy cosine scores are not DigUp’s final ranking scores: the next slide adds its documented calibration, keyword matching and file grouping. A cosine is a similarity score, not a probability.</p>',
 link(SEARCH+'#L294','DigUp matrix–vector search'),
 ['Encode the query once','Read the stored candidate vectors','Compute one cosine for each piece','Scale the same operation to the full index'], 'ILLUSTRATIVE · 3D unit vectors for arithmetic; DigUp uses 768D')

slide('hybrid-search','Meaning helps. Exact words still matter.',
 '<p class="lead hybrid-query">Find the invoice containing <b>4815162342</b></p><div class="split hybrid"><div><p class="label blue">Semantic route</p>'+node('Query embedding → cosine scores','Find related content')+step('<p>Calibrate across modalities;<br>keep each file’s best segment.</p>',2)+'</div>'+step('<div><p class="label purple">Word route</p>'+node('Filename + text + OCR','Look for the actual identifier','purple')+'<p>An embedding need not preserve<br>every digit of an invoice number.</p></div>',3)+'</div>'+step('<div class="merge">Semantic evidence + word coverage + file-type hint <span>→ ranked files</span></div>',4),
 'DigUp combines learned similarity with ordinary search rules.',
 '<p>The pinned Searcher.swift computes cosine similarity, subtracts modality means and divides centered scores by a shared scale with a floor. It selects the best semantic segment per file. FTS5 searches file/folder names, document text and OCR. Files qualify through semantic salience or enough query-word coverage.</p><p>For ordinary non-code queries, score = max(z,0) + 2 × coverage^1.5 + a file-kind bonus when applicable. Lookup-like queries halve the semantic weight and double the word weight; code uses a smaller word weight. This is a concise description of that revision, not a universal formula for embedding search. Exact IDs and dates are useful tests of the lexical route.</p>',
 link(SEARCH+'#L174','Hybrid ranking')+' · '+link(SEARCH+'#L315','Modality calibration'),
 ['Try an exact identifier','Score semantic matches','Also search the literal words','Combine the evidence'], 'IMPLEMENTATION DETAIL · DigUp revision f9b8106 · no measured ranking shown')

slide('open-source-piece','A search result needs a route back to its source',
 '<div class="route-table"><div class="route-head"><span>Your query</span><span>Winning indexed piece</span><span>Open here</span></div>'+step('<div><q>pets in the lease</q><span>PDF page / text passage</span><b class="purple">PDF page</b></div>',1)+step('<div><q>zebra in a video</q><span>sampled video frame</span><b class="amber">timestamp</b></div>',2)+step('<div><q>code: retry with backoff</q><span>source-code segment</span><b class="blue">file + line</b></div>',3)+'</div>'+step('<p class="class-task"><b>Try this with our slides:</b> query → ranked excerpts → lecture + slide link.<br>'+link('https://nipunbatra.github.io/attention/embeddinggemma/#documents','Open the course-search lab')+'</p>',4),
 'Retrieval returns evidence you can inspect. A generated answer would be an extra step.',
 '<p>These are schematic routes based on documented DigUp behavior. Source-code search is opt-in and has its own index; explicit identifier queries also use definition lookup. Semantic matches generally open the best segment; exact word hits may determine a more precise location. Sparse video frames can miss a brief event; index completeness and source metadata are part of retrieval quality.</p><p>Class question: if we change a lecture slide, what must be refreshed? Answer: its extracted content and embedding, while preserving the correct lecture/slide link. Our existing course-search lab illustrates the same pattern; it is a separate app, not a DigUp integration. No generative answer model is needed to show cited excerpts.</p>',
 link(SEARCH+'#L193','Result location selection')+' · '+link(REPO,'Code-search behavior'),
 ['Open the PDF page','Open the video moment','Open the code location','Apply the same pattern to course slides'], 'SCHEMATIC · example queries from the project; course exercise is an analogy')

slide('speech-demo','Can an embedding model help transcribe speech?',
 '<div class="split demo-split">'+video('speech')+'<div class="speech-metrics"><p class="label">AUTHOR-REPORTED</p><p class="metric">~13 s<small>to process 1 hour of audio<br>base M3 MacBook Air</small></p>'+step('<p class="metric smaller">5.18%<small>word error rate<br>LibriSpeech test-clean</small></p>',2)+step('<p class="plain-note">EmbeddingGemma stays frozen.<br>A separate reader is trained.</p>',3)+'</div></div>',
 'A shared space is useful, but a transcript still needs a sequence of word decisions.',
 '<p>Play Shahir’s 26-second video. The project site reports 12.6 seconds and calls it 288× real time; rounded figures need not divide to that exact factor, so this slide uses “about 13 seconds.” These are the developer’s measurements, not a course benchmark. Author replies report 5.18% WER on LibriSpeech test-clean and about 11.5–11.8% on the harder test set. The app uses the Apple Neural Engine and is not a WebGPU demo.</p><p>Ask whether finding a recording “about sleep” is enough to recover its exact words. Retrieval can tolerate paraphrase; transcription must preserve lexical detail and time order.</p>',
 link(MEDIA['speech']['post'],'Shahir · X demo')+' · '+link(SPEECH,'Method & timings')+' · '+link(WER,'WER report'),
 ['Watch the speech demo','Read speed alongside accuracy','Identify the extra trained component'], 'AUTHOR DEMO · 12 October 2026 · benchmark not independently reproduced')

readerflow=flow(node('Audio slices','ordered through time','teal'),step(node('Audio encoder','EmbeddingGemma 2<br>frozen','blue'),2),step(node('Word reader','trained mapping','purple'),3),step(node('Aligned vectors','one decision at a time','teal'),4))
slide('speech-reader','Keep the audio order; learn how to read it',
 readerflow+'<div class="timeline">'+''.join(f'<div data-reveal-step="{1 if i==0 else 4}"><span class="time-label">{label}</span><div class="time-cells">'+''.join(f'<span>{x}</span>' for x in vals)+'</div></div>' for i,(label,vals) in enumerate([('Audio slices',['t₁','t₂','t₃','…']),('Reader outputs',['r₁','r₂','r₃','…'])]))+'</div>'+step('<p class="plain-note wide">A single pooled vector is useful for retrieval.<br>Exact words require decisions across time.</p>',4),
 'The author describes a reader over audio slices, not inversion of one whole-recording vector.',
 '<p>The public description says the audio encoder is unchanged, a small reader maps short audio slices into the model’s text space, nearest words are selected, and a tiny model adds punctuation/capitalization. This is a high-level diagram of those stated components. The exact slice duration, reader layers, alignment procedure and training loss are not specified in the public notes inspected on 13 October 2026.</p><p>Do not infer that an arbitrary 768-dimensional pooled embedding uniquely determines a transcript. Similar meanings can have different wording. We do not label this reader as CTC or an autoregressive decoder without implementation evidence.</p>',
 link(SPEECH,'Author’s method')+' · '+link(METHOD,'Method thread'),
 ['Keep a time-ordered audio sequence','Reuse the frozen audio encoder','Train a reader for the new task','Produce vectors for word decisions'], 'SOURCE-GROUNDED SCHEMATIC · internal layers and slice timing not disclosed')

slide('word-lookup','Compare a reader output with candidate words',
 '<div class="word-intro">'+node('This audio slice','reader output r = [0.6, 0.8]','teal')+arrow()+node('Stored text-space word vectors','~229,000 English words in the author’s app','purple')+'</div>'+step('''<table class="numbers words"><thead><tr><th>Candidate</th><th>Toy word vector</th><th>Illustrative cosine r · e<sub>w</sub></th></tr></thead><tbody><tr><th>cat</th><td>[0.8, 0.6]</td><td data-reveal-step="3" class="score">0.6 × 0.8 + 0.8 × 0.6 = <b>0.96</b></td></tr><tr><th>cap</th><td>[1, 0]</td><td data-reveal-step="3">0.60</td></tr><tr><th>dog</th><td>[−0.6, −0.8]</td><td data-reveal-step="3">−1.00</td></tr></tbody></table>''',2)+step('<p class="word-choice">arg max<sub>w</sub> (r · e<sub>w</sub>) → <b>cat</b> <span>→ assemble words over time → punctuation</span></p>',4),
 'The comparison looks familiar. The learned reader makes it useful for recognizing words.',
 '<p>All 2D vectors, word scores and the winning word on this slide are invented for teaching. Every vector is unit length, so a dot product equals cosine. The documented project selects the nearest of about 229,000 English words in the model’s text space. The exact distance function is not publicly specified; cosine/argmax here illustrates nearest-vector lookup, not a reproduced implementation or output.</p><p>The table contains candidate word representations, not the token-embedding matrix of a language model. The public notes do not specify how repeated slice predictions are aligned or merged. A tiny additional model restores punctuation and capitals.</p>',
 link(SPEECH,'Nearest-word method'),
 ['Map one audio slice into the text space','Compare with word vectors','Work out the toy dot products','Choose a word and continue through time'], 'ILLUSTRATIVE · 2D arithmetic, invented vectors; exact lookup metric not disclosed')

slide('what-is-trained','The embedding model is reused; the reader learns a new job',
 '<div class="training-line">'+node('Audio + transcript','supervised training examples','teal')+arrow()+node('Frozen audio encoder','weights stay fixed')+arrow()+node('Train the reader','transcript supervision','purple')+'</div>'+step('<div class="training-update"><span>Training feedback</span><b>↳ update the reader</b><small>The public notes do not specify its loss or alignment algorithm.</small></div>',2)+step('<div class="summary-pair"><div><p class="label blue">DIGUP</p><p>Encode files → store vectors<br>→ retrieve a source location</p></div><div><p class="label purple">SPEECH DEMO</p><p>Audio features → trained reader<br>→ word sequence</p></div></div>',3)+step('<p class="closing-check">What would you test next? <b>Names, accents, noise and long recordings.</b></p>',4),
 'Shared representations enable new applications. The task still determines the readout and the evaluation.',
 '<p>The author reports training the reader in under two hours on eight H100 GPUs using openly licensed English speech; fast on-device inference does not mean this training ran on the laptop. The pretrained embedding model was not originally trained as this speech recognizer. The public method leaves the exact objective unspecified, so the diagram intentionally shows only the supported direction of learning.</p><p>Evaluate transcription using WER = (substitutions + deletions + insertions) / reference words, and runtime on the same audio and hardware. A good retrieval result and a good transcript require different evidence. The author acknowledges higher errors on their own speech. Download links lead to independent third-party Mac apps; the lecture itself does not install anything.</p>',
 link(METHOD,'Training report')+' · '+link(SPEECH,'Speech app')+' · '+link(REPO,'DigUp'),
 ['Pair the audio with a transcript','Train the new reader','Compare the two applications','Choose the next evaluation'], 'AUTHOR METHOD · high-level training view; no invented loss or reproduced benchmark')

from practical_slides import add
add(slide,step,node,flow,arrow,link,MODEL,REPO)
order=['digup-demo','build-index','index-recipe','query-scores','hybrid-search','lookup-recipe','open-source-piece','speech-demo','speech-reader','word-lookup','training-data','reader-shapes','reader-loss','reader-code','full-transcription','what-is-trained']
by_id={re.search(r'<section class="slide" id="([^"]+)"',s)[1]:s for s in slides}
slides=[by_id[id] for id in order]
for j,s in enumerate(slides):
 s=re.sub(r'<span>\d+ / 9</span>',f'<span>{j+1} / {len(slides)}</span>',s)
 if order[j] in ['index-recipe','lookup-recipe','training-data','reader-shapes','reader-loss','reader-code','full-transcription']:
  s=re.sub(r'CLIP ADDENDUM · [^<]+','CLIP ADDENDUM · PRACTICAL BUILD',s,count=1)
 slides[j]=s

nav='''<nav class="toolbar"><a class="brand" href="https://nipunbatra.github.io/attention/clip/">CLIP · Addendum</a><button id="mode">Present</button><button id="previous" aria-label="Previous step">←</button><span id="counter" aria-live="polite"></span><button id="next" aria-label="Next step">→</button><button id="all">Show all</button><button id="notes-toggle">Notes</button><a href="clip-addendum.pdf">PDF</a><a href="SOURCES.md">Sources</a></nav>'''
html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CLIP addendum · DigUp and speech from embeddings</title><style>'+ (ROOT/'src/style.css').read_text()+'</style></head><body>'+nav+'<main>'+''.join(slides)+'</main><script>'+(ROOT/'src/deck.js').read_text()+'</script></body></html>'
(ROOT/'index.html').write_text(html)
print(f'Built {len(slides)} slides')

"""Every figure must be backed by the calculation presented beside it."""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slow_walkthrough import STAGES, STORY_EXAMPLES, ROUTE_CHECKPOINTS, initial_namespace, render_figure, map_mode


def test_all_lesson_steps_execute_and_render():
    ns=initial_namespace()
    assert len(STAGES)==88
    assert len({s['id'] for s in STAGES})==88
    for s in STAGES:
        exec(compile(s['code'],s['id'],'exec'),ns)
        root=ET.fromstring(render_figure(s,ns))
        assert root.attrib['role']=='img'
        assert root.attrib['aria-label']
    assert ns['all_X'].shape==(7,4)
    assert ns['all_y'].tolist()==[8,7,5,9,6,4,2]
    assert ns['X'].tolist()==[[0,1,8,7],[1,8,7,5]]
    assert ns['window_counts']=={'train':969160,'validation':120220,'test':120393}
    assert ns['A'][0,0,0]==0
    assert ns['new_weight']!=ns['old_weight']
    assert ns['after_loss']<ns['before_loss']
    assert ns['comparison']['attention']['test_perplexity']['mean']<ns['comparison']['mlp']['test_perplexity']['mean']


def test_real_story_examples_have_reproducible_lengths_and_provenance():
    import hashlib
    from wordlm import tokenize
    calculations=[s for s in STAGES if not s.get('map_checkpoint')]
    assert [s['id'] for s in calculations[:4]] == ['data','story-complete','story-excerpts','split']
    assert STORY_EXAMPLES['license']=='CDLA-Sharing-1.0'
    assert STORY_EXAMPLES['revision']=='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
    assert STORY_EXAMPLES['subset_documents']==6000
    assert [(s['row_idx'],s['word_count'],s['token_count']) for s in STORY_EXAMPLES['examples']]==[(1992490,52,60),(12400,107,133),(588307,248,300)]
    for story in STORY_EXAMPLES['examples']:
        assert hashlib.sha256(story['text'].encode()).hexdigest()==story['sha256']
        assert len(story['text'].split())==story['word_count']
        assert len(tokenize(story['text']))==story['token_count']
    ns=initial_namespace()
    for s in calculations[:3]:
        exec(s['code'],ns)
        svg=render_figure(s,ns)
        if s['id']=='story-complete':
            shown=' '.join(t.text for t in ET.fromstring(svg).findall('{http://www.w3.org/2000/svg}text')[2:-1])
            assert shown==STORY_EXAMPLES['examples'][0]['text']
        elif s['id']=='story-excerpts':
            assert svg.count('[…]')==2
            assert '107 words · 133 tokens' in svg
            assert '248 words · 300 tokens' in svg
            assert '60–942 tokens; median 176' in svg


def test_tokenization_detour_matches_the_actual_rules_without_changing_the_toy():
    keys=[s['id'] for s in STAGES]
    start=keys.index('sentence')
    assert keys[start:start+6]==['sentence','tokenization-intro','tokenization-choices','tokenization-rules','route-tokenize','tokenize']
    ns=initial_namespace()
    for stage in STAGES:
        exec(stage['code'],ns)
        if stage['id']=='vocabulary': break
    assert ns['tokenization_choices']=={
        'Word + punctuation':['redder','!'],
        'Character':list('redder!'),
        'Subword (illustrative)':['red','der','!'],
    }
    assert list(ns['tokenization_counts'].values())==[2,7,3]
    assert ns['probe_tokens']==['lily',"can't",'find','12','balls','!']
    assert ns['pieces']==['lily','found','a','red','ball','.']
    assert ns['C']==10 and ns['vocab'].stoi['red']==9
    choices=next(s for s in STAGES if s['id']=='tokenization-choices')
    assert 'illustrative' in render_figure(choices,ns)


def test_unknown_token_example_explains_the_lookup_and_boundary_flag():
    from build_slow_lesson import SLIDE_CODE
    ns = initial_namespace()
    for stage in STAGES:
        exec(stage['code'], ns)
        if stage['id'] == 'special':
            break
    assert 'blue' not in ns['vocab'].stoi
    assert ns['token_ids'] == [3]
    assert ns['words'][ns['token_ids'][0]] == '<UNK>'
    assert ns['with_boundaries'] == [1, 3, 2]
    assert "['blue']" in SLIDE_CODE['special']
    assert 'print(token_ids)  # [3]: UNK' in SLIDE_CODE['special']
    assert SLIDE_CODE['special'] in stage['code']
    # The slide exporter keeps the first two sentences. Both must explain the call.
    shown = '. '.join(stage['body'].split('. ')[:2])
    assert 'missing from our toy vocabulary' in shown
    assert '[3], the UNK ID' in shown
    assert 'boundaries=False' in shown and 'do not add BOS or EOS' in shown


def test_notebook_stages_are_complete_and_linked():
    from build_slow_lesson import notebook_cells
    import nbformat as nbf
    cells=notebook_cells(nbf.v4.new_markdown_cell,nbf.v4.new_code_cell)
    assert [c.metadata['lesson_stage'] for c in cells if 'lesson_stage' in c.metadata]==[s['id'] for s in STAGES]
    for index,s in enumerate(STAGES):
        assert f'#s19/{index+5}/0' in cells[2+index*2].source
        assert f"show_figure('{s['id']}'" in cells[3+index*2].source


def test_pairs_grow_in_lockstep_before_tensor_conversion():
    from build_slow_lesson import SLIDE_CODE
    ns=initial_namespace()
    expected_inputs=[[0,0,0,1],[0,0,1,8],[0,1,8,7],[1,8,7,5],
                     [8,7,5,9],[7,5,9,6],[5,9,6,4]]
    expected_targets=[8,7,5,9,6,4,2]
    checkpoints={'pair-lists':0,'pair-first':0,'pair-append':1,
                 'pair-second':1,'pair-append-second':2,'pairs-loop':7}
    for stage in STAGES:
        exec(stage['code'],ns)
        key=stage['id']
        if key=='one-pair':
            assert ns['context_ids']==[1,8,7,5] and ns['target_id']==9
            assert 'contexts' not in ns
        if key in checkpoints:
            count=checkpoints[key]
            assert ns['contexts']==expected_inputs[:count]
            assert ns['targets']==expected_targets[:count]
            assert 'all_X' not in ns
        if key=='pairs-tensors':
            assert ns['all_X'].tolist()==ns['contexts']==expected_inputs
            assert ns['all_y'].tolist()==ns['targets']==expected_targets
            assert ns['all_X'].dtype==ns['all_y'].dtype==ns['torch'].long
            assert len({id(row) for row in ns['contexts']})==7
            for row,context in enumerate(ns['contexts']):
                t=row+1
                visible=ns['ids'][max(0,t-4):t]
                assert context==[0]*(4-len(visible))+visible
                assert ns['targets'][row]==ns['ids'][t]
        if key=='batch':
            assert ns['X'].tolist()==expected_inputs[2:4]
            assert ns['y'].tolist()==expected_targets[2:4]
            break
    for key in ['pair-append','pair-append-second','pairs-loop']:
        assert 'contexts.append(context_ids)' in SLIDE_CODE[key]
        assert 'targets.append(target_id)' in SLIDE_CODE[key]
    for key in ['one-pair','pair-first','pair-second']:
        assert 'context_ids =' in SLIDE_CODE[key]
        assert 'target_id =' in SLIDE_CODE[key]
    assert all('one_x' not in s['code'] and 'one_y' not in s['code'] for s in STAGES)


def test_batch_ids_decode_to_tokens_before_any_embedding_lookup():
    ns=initial_namespace()
    for stage in STAGES:
        exec(stage['code'],ns)
        if stage['id']=='batch':
            svg=ET.fromstring(render_figure(stage,ns))
            labels=[t.text for t in svg.findall('{http://www.w3.org/2000/svg}text')]
            assert labels[:6]==['Data row','Batch row','X: input IDs','Input tokens','y: ID','Target token']
            assert labels[6:12]==['2','0','[0, 1, 8, 7]','<PAD> <BOS> lily found','5','a']
            assert labels[12:18]==['3','1','[1, 8, 7, 5]','<BOS> lily found a','9','red']
            assert 'Meaning' not in labels
        if stage['id']=='batch-ids':
            assert ns['X'].dtype==ns['y'].dtype==ns['torch'].long
            assert ns['X'].shape==(2,4) and ns['y'].shape==(2,)
            assert 'E_mlp' not in ns and 'mlp' not in ns
            text=' '.join(ET.fromstring(render_figure(stage,ns)).itertext())
            assert 'X.shape = (2, 4)' in text and 'y.shape = (2,)' in text
            assert 'y stays as target IDs for the loss' in stage['body']
            break


def test_batch_count_explanation_states_the_update_assumption():
    stage=next(s for s in STAGES if s['id']=='batches')
    assert stage['title']=='Seven examples, four batches'
    assert 'B=2 with drop_last=False' in stage['body']
    assert 'With one update per batch' in stage['body']
    assert '512 windows per update with replacement' in stage['body']


def test_dimensions_have_one_symbol_value_and_definition_per_row():
    ns=initial_namespace()
    for stage in STAGES:
        exec(stage['code'],ns)
        if stage['id']=='shapes': break
    def labels():
        return [t.text for t in ET.fromstring(render_figure(stage,ns))
                .findall('{http://www.w3.org/2000/svg}text')]
    shown=labels()
    assert shown[:4]==['Data dimensions','Symbol','Value','What it counts']
    assert shown[4:13]==[
        'B','2','examples per batch',
        'w','4','input slots per example',
        'C','10','vocabulary items',
    ]
    assert shown[13:17]==['Model dimensions','Symbol','Value','What it counts']
    assert shown[17:]==[
        'd','4','embedding coordinates',
        'h','8','prediction-head hidden units',
        'dₖ','3','query and key coordinates',
        'dᵥ','2','value coordinates',
    ]
    assert not any('/' in text for text in shown)
    assert 'C includes the special tokens' in stage['body']
    # The displayed values must come from the executed example, not fixed labels.
    for key,value,index in [('B',5,5),('w',6,8),('C',12,11),
                            ('d',7,18),('h',9,21),('d_k',4,24),('d_v',6,27)]:
        ns[key]=value
        assert labels()[index]==str(value)


def test_map_checkpoints_precede_examples_without_changing_state():
    from pipeline_maps import graph
    by_id={s['id']:i for i,s in enumerate(STAGES)}
    for target,(key,title,focus,body) in ROUTE_CHECKPOINTS.items():
        assert by_id[key]+1==by_id[target]
        stage=STAGES[by_id[key]]
        assert stage['map_checkpoint'] and stage['code']==''
        svg=ET.fromstring(render_figure(stage,{}))
        assert set(svg.attrib['data-focus'].split())==set(focus)
        nodes=svg.findall("{http://www.w3.org/2000/svg}g[@data-stage]")
        assert {n.attrib['data-stage'] for n in nodes if n.attrib['data-active']=='true'}==set(focus)
        assert len(nodes)==len(graph(stage['kind'],map_mode(stage))['nodes'])
    assert map_mode(STAGES[by_id['route-prompt']])=='inference'
    assert STAGES[by_id['route-prompt']]['kind']=='mlp'


def test_lookup_flow_selects_a_whole_row_for_each_input_id():
    ns=initial_namespace()
    for stage in STAGES:
        exec(stage['code'],ns)
        if stage['id']=='lookup-flow': break
    assert ns['selected_rows'].shape==(4,4)
    assert ns['torch'].equal(ns['selected_rows'],ns['embedding_table'][ns['X'][1]])
    assert ns['torch'].equal(ns['selected_rows'][2],ns['found_vector'])
    text=' '.join(ET.fromstring(render_figure(stage,ns)).itertext())
    assert 'T: 10 rows × 4 coordinates' in text
    assert 'X [2,4] indexes T [10,4] to produce E [2,4,4]' in text

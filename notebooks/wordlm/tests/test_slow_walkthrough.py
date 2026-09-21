"""Every figure must be backed by the calculation presented beside it."""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slow_walkthrough import STAGES, initial_namespace, render_figure


def test_all_lesson_steps_execute_and_render():
    ns=initial_namespace()
    assert len(STAGES)==47
    assert len({s['id'] for s in STAGES})==47
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


def test_notebook_stages_are_complete_and_linked():
    from build_slow_lesson import notebook_cells
    import nbformat as nbf
    cells=notebook_cells(nbf.v4.new_markdown_cell,nbf.v4.new_code_cell)
    assert [c.metadata['lesson_stage'] for c in cells if 'lesson_stage' in c.metadata]==[s['id'] for s in STAGES]
    for index,s in enumerate(STAGES):
        assert f'#s19/{index+5}/0' in cells[2+index*2].source
        assert f"show_figure('{s['id']}'" in cells[3+index*2].source

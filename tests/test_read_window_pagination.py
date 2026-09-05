from __future__ import annotations

import io
import os
import re
import sys

import pytest

from harness.tools.read_window import read_window, embedded_window_source
from harness.tools.file_read import FileReadTool
from bench._streaming_harbor_read import HarborFileReadTool
from tests.test_file_publication_integration import tool as harbor_tool

pytestmark=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux pinned-file fixtures')


def records(output):
    return [(int(line.split('\t',1)[0]),line.split('\t',1)[1])
            for line in output.splitlines() if re.match(r'^\d+\t',line)]


@pytest.fixture(params=['host','embedded'])
def window(request):
    if request.param=='host':return read_window
    namespace={};exec(embedded_window_source(),namespace);return namespace['read_window']


@pytest.mark.parametrize('cap',[50,60,70,80,99,100,101,150,200])
def test_complete_records_and_footer_never_skip_or_split_content(window,cap):
    lines=[f'line{i:02d}-'+('é🌲' if i%2 else 'x')*3 for i in range(1,20)]
    offset=1;observed=[]
    for _ in range(len(lines)+1):
        source=iter((s,False) for s in lines)
        result=window(lambda:next(source,None),offset=offset,limit=10,
                      max_line_bytes=1000,max_output_chars=cap)
        if not result['success']:
            assert observed==[] or result['metadata']['retry_offset']==offset
            assert result['metadata']['next_offset'] is None
            assert result['metadata']['output_limit_too_small']
            break
        page=records(result['output']);meta=result['metadata']
        assert len(result['output'])<=cap and len(page)==meta['lines_returned']
        assert page==[(i,lines[i-1]) for i in range(offset,offset+len(page))]
        observed.extend(page)
        if not meta['has_more']:
            assert observed==list(enumerate(lines,1));break
        assert page and meta['end_line']==page[-1][0]
        assert meta['next_offset']==page[-1][0]+1 and meta['next_offset']>offset
        assert result['output'].endswith(f"offset={meta['next_offset']} to continue)")
        offset=meta['next_offset']
    else:pytest.fail('pagination did not terminate')


@pytest.mark.parametrize('cap',[1,2,12,40,80])
def test_unrepresentable_first_record_is_an_actionable_failure(window,cap):
    source=iter([('x'*100,False),('second',False)])
    result=window(lambda:next(source,None),offset=1,limit=10,max_line_bytes=200,max_output_chars=cap)
    assert not result['success'] and result['output']==''
    assert result['metadata']['next_offset'] is None and result['metadata']['retry_offset']==1
    assert result['metadata']['required_output_chars']>cap
    assert 'Increase max_output_chars' in result['error']


def test_exact_cap_at_eof_needs_no_footer(window):
    source=iter([('abc',False)])
    result=window(lambda:next(source,None),offset=1,limit=1,max_line_bytes=10,max_output_chars=5)
    assert result['success'] and result['output']=='1\tabc'
    assert result['metadata']['total_lines_known'] and result['metadata']['total_lines']==1
    assert not result['metadata']['has_more']


@pytest.mark.parametrize('lines,offset',[([],1),(['a','b'],10)])
def test_empty_or_beyond_eof(window,lines,offset):
    source=iter((s,False) for s in lines)
    result=window(lambda:next(source,None),offset=offset,limit=5,max_line_bytes=10,max_output_chars=1)
    assert result['success'] and result['output']==''
    assert result['metadata']['total_lines']==len(lines)
    assert result['metadata']['lines_returned']==0 and result['metadata']['next_offset'] is None


@pytest.mark.parametrize('engine',['local','harbor'])
def test_public_pagination_reconstructs_every_short_line(tmp_path,engine):
    path=tmp_path/'lines';lines=[f'line{i}-'+'x'*20 for i in range(1,7)]
    path.write_text('\n'.join(lines)+'\n')
    reader=FileReadTool(max_output_chars=80) if engine=='local' else harbor_tool(HarborFileReadTool)
    reader.max_output_chars=80
    seen=[];offset=1
    for _ in range(10):
        result=reader.execute(str(path),offset=offset,limit=10)
        assert result.success,result.error
        page=records(result.output);seen.extend(page)
        assert len(result.output)<=80 and page
        marker=re.search(r'^\.\.\. \(more lines, use offset=(\d+) to continue\)$',result.output,re.M)
        if not marker:break
        next_offset=int(marker.group(1));assert next_offset==page[-1][0]+1 and next_offset>offset
        if engine=='local':
            assert result.metadata['lines_returned']==len(page)
            assert result.metadata['next_offset']==next_offset
        offset=next_offset
    assert seen==list(enumerate(lines,1))


@pytest.mark.parametrize('engine',['local','harbor'])
def test_public_tiny_cap_never_returns_successful_same_offset(tmp_path,engine):
    path=tmp_path/'long';path.write_text('x'*100+'\nnext\n')
    reader=FileReadTool(max_output_chars=80) if engine=='local' else harbor_tool(HarborFileReadTool)
    reader.max_output_chars=80
    result=reader.execute(str(path),limit=10)
    assert not result.success and result.output==''
    assert result.metadata['output_limit_too_small'] and result.metadata['next_offset'] is None


@pytest.mark.parametrize('engine',['local','harbor'])
@pytest.mark.parametrize('content',[b'a'*9000+b'\x00\n',b'a'*9000+b'\xff\n',b'%PDF-1.7'])
def test_binary_and_decoder_failures_still_propagate(tmp_path,engine,content):
    path=tmp_path/'content';path.write_bytes(content)
    reader=FileReadTool(max_output_chars=80) if engine=='local' else harbor_tool(HarborFileReadTool)
    result=reader.execute(str(path),limit=1)
    assert not result.success and result.output==''
    assert result.metadata.get('binary_file_unsupported') or result.metadata.get('text_decode_error')


@pytest.mark.parametrize('engine',['local','harbor'])
def test_fifo_read_rejects_without_waiting(tmp_path,engine):
    path=tmp_path/'pipe';os.mkfifo(path)
    reader=FileReadTool() if engine=='local' else harbor_tool(HarborFileReadTool)
    result=reader.execute(str(path))
    assert not result.success

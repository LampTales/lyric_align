"""Compare saved -1.5 and -2.0 runs in one four-track lyric player."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from urllib.parse import quote

from lyric_align import AlignmentArtifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    base = args.baseline.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    songs, comparisons, changes = [], [], []
    variants = {'japanese': '旧日语模型 · 阈值 −1.5', 'japanese2': '旧日语模型 · 阈值 −2.0',
                'nextfire': 'NextFire · 阈值 −1.5', 'nextfire2': 'NextFire · 阈值 −2.0'}
    keys = ('text', 'start_ms', 'end_ms', 'status', 'alignment_status', 'coverage', 'ctc_score',
            'timing_source', 'ctc_window', 'warnings')
    diagnostics = Counter()
    for item in manifest['songs']:
        name = item['song']
        versions = {}
        for key in variants:
            newer = key.endswith('2')
            profile = key.removesuffix('2')
            value = json.loads(((out if newer else base) / profile / name / 'alignment.json').read_text())
            AlignmentArtifact.from_dict(value).validate()
            assert value['stages']['ctc']['status'] == 'done', (name, key)
            versions[key] = value
        anchor = versions['japanese']
        for key, artifact in versions.items():
            assert artifact['timing'] == anchor['timing'], (name, key, 'offset differs')
            assert [(l['source_index'], l['text'], l['start_ms'], l['end_ms']) for l in artifact['lines']] == [(l['source_index'], l['text'], l['start_ms'], l['end_ms']) for l in anchor['lines']]
        for profile in ('japanese', 'nextfire'):
            old, new = versions[profile]['lines'], versions[profile+'2']['lines']
            eligible = sum(bool(l['reading']) and l['status'] != 'non_sung' for l in old)
            promoted = []
            for a,b in zip(old,new):
                assert a['ctc_score'] == b['ctc_score'] and a['coverage'] == b['coverage'], (name, profile, a['source_index'], 'score changed')
                assert a['alignment_status'] != 'ctc' or b['alignment_status'] == 'ctc', 'unexpected loss'
                if a['alignment_status'] != 'ctc' and b['alignment_status'] == 'ctc':
                    repaired = any('zero-duration' in w for w in b['warnings'])
                    diagnostics[profile+'_newly_accepted'] += 1
                    diagnostics[profile+'_newly_accepted_with_token_repair'] += repaired
                    changed = dict(song=name,profile=profile,source_index=b['source_index'],text=b['text'],
                                   score=b['ctc_score'],coverage=b['coverage'],previous_timing_source=a['timing_source'],
                                   timing_source=b['timing_source'],token_repaired=repaired,warnings=b['warnings'])
                    changes.append(changed);promoted.append(b['source_index'])
            ca=sum(l['alignment_status']=='ctc' for l in old)
            cb=sum(l['alignment_status']=='ctc' for l in new)
            comparisons.append(dict(song=name,profile=profile,eligible=eligible,before_accepted=ca,after_accepted=cb,
                                    before_fallback=eligible-ca,after_fallback=eligible-cb,newly_accepted=len(promoted),
                                    promoted_source_indices=promoted,offset_ms=anchor['timing']['global_offset_ms']))
        song_dir=out/'nextfire'/name
        audio=next(song_dir.glob('audio.*'));vocals=next((song_dir/'stems').glob('vocals.*'))
        lines=[]
        for i,a in enumerate(anchor['lines']):
            row={k:a[k] for k in ('text','start_ms','end_ms')}
            for key in variants:
                l=versions[key]['lines'][i]
                # The player needs token text/times, not nested source maps.
                value={k:l.get(k) for k in keys}
                value.update({field:[{k:u[k] for k in ('text','start_ms','end_ms')} for u in l[field]]
                              for field in ('tokens','display_units')})
                row[key]=value
            row['newly_accepted']=any(row[p]['alignment_status']!='ctc' and row[p+'2']['alignment_status']=='ctc' for p in ('japanese','nextfire'))
            lines.append(row)
        songs.append(dict(name=name,audio=quote(str(audio.relative_to(out)),safe='/'),
                          vocals=quote(str(vocals.relative_to(out)),safe='/'),lines=lines,
                          offset_ms=anchor['timing']['global_offset_ms'],offset_status=anchor['timing']['offset_status']))
    template=Path(__file__).with_name('review_template.html').read_text()
    template=template.replace('<title>NextFire 歌词对照</title>','<title>阈值 −1.5 / −2.0 对照试听</title>')
    template=template.replace('<h1>NextFire / 当前日语模型</h1>','<h1>阈值 −1.5 / −2.0 对照试听</h1>')
    start=template.index('<p>');end=template.index('</p>',start)+4
    template=template[:start]+'<p>7 首完整重跑，offset 与边界检查开启。每句展示两模型各自的 −1.5 与 −2.0 结果。点击“下一新增 CTC 句”试听由整行均分改为模型时间的片段。</p>'+template[end:]
    template=template.replace("['japanese','nextfire']",json.dumps(list(variants)))
    template=template.replace("(profile==='japanese'?'当前日语模型':'NextFire')",'('+json.dumps(variants,ensure_ascii=False)+')[profile]')
    template=template.replace("result.alignment_status+' · 覆盖率 '","(result.alignment_status || result.status)+' · score '+result.ctc_score+' · '+result.timing_source+' · 覆盖率 '")
    template=template.replace("result.alignment_status!=='ctc'?' · 最终字符使用回退时间':''", "result.alignment_status==='fallback'?' · 最终字符使用整行插值':''")
    template=template.replace("summary.textContent='目标与诊断';", "summary.textContent='目标与诊断';")
    template=template.replace("score:row[p].ctc_score,warnings:","score:row[p].ctc_score,timing_source:row[p].timing_source,warnings:")
    template=template.replace("`${i+1}. ${row.text}`", "`${i+1}. ${row.newly_accepted?'【新增 CTC】 ':''}${row.text}`")
    template=template.replace('<button id="next">下一句</button>','<button id="next">下一句</button><button id="next-promoted">下一新增 CTC 句</button>')
    template=template.replace("$('view').onchange=draw;", """$('view').onchange=draw;
$('next-promoted').onclick=()=>{const rows=song().lines;let i=rows.findIndex(r=>r.newly_accepted&&r.start_ms>audio.currentTime*1000+100);if(i<0)i=rows.findIndex(r=>r.newly_accepted);if(i>=0)seekLine(i)};""")
    template=template.replace('同一人声、同一歌词窗口，关闭全曲 offset。','同一人声、同一歌词窗口，开启全曲 offset。')
    data=json.dumps(songs,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (out/'review.html').write_text(template.replace('__REVIEW_DATA__',data),encoding='utf-8')
    (out/'threshold_comparison.json').write_text(json.dumps(dict(songs=comparisons,diagnostics=diagnostics),ensure_ascii=False,indent=2))
    with (out/'newly_accepted.csv').open('w',encoding='utf-8-sig',newline='') as f:
        if changes:
            w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader()
            for row in changes:w.writerow({**row,'warnings':json.dumps(row['warnings'],ensure_ascii=False)})
    report=['# 阈值 -2.0 重跑比较','','7 首完整歌曲，新旧模型共 14 次推理；offset 与边界检查保持开启。与基线代码和模型相同，唯一配置变更为 ctc_score_threshold=-2.0。',
            '','已逐行核对原始分数、coverage、时间窗口以及 offset 诊断均与 -1.5 相同，没有原先通过而此次回退的行。',
            '','| 歌曲 | 可对齐行 | 旧模型回退：-1.5 → -2 | NextFire 回退：-1.5 → -2 |', '|---|---:|---:|---:|']
    for a,b in zip(comparisons[::2],comparisons[1::2]):
        report.append(f"| {a['song']} | {a['eligible']} | {a['before_fallback']} → {a['after_fallback']} | {b['before_fallback']} → {b['after_fallback']} |")
    report+=['','新增通过并不等于准确率提升。请重点试听标记【新增 CTC】的句子；tokens 是保存的模型 token 时间，可能经过零时长修复，非未经处理的原始输出。', '', '新增 CTC 与修复统计：', '', '```json',json.dumps(diagnostics,ensure_ascii=False,indent=2),'```']
    (out/'THRESHOLD_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(songs=len(songs),lines=sum(len(s['lines']) for s in songs),diagnostics=diagnostics),ensure_ascii=False))


if __name__=='__main__':
    main()

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Create complete CSV, Markdown and an offline filterable table from the suite."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = {}
    for path in sorted(directory.glob("[0-9]*.json")):
        result = json.loads(path.read_text())
        records[result["id"]] = result
    rows = []
    for case in manifest["requested_cases"]:
        result = records.get(case["id"], {})
        backends = result.get("backends", {})
        ts = backends.get("ts-auto", {})
        trt = backends.get("trtllm-gen", {})
        row = dict(case)
        row.update(
            ts_status=ts.get("status", "not_run"),
            trt_status=trt.get("status", "not_run"),
            ts_us=ts.get("median_us"),
            trt_us=trt.get("median_us"),
            gap_percent=None,
            ts_over_trt=None,
            complete=result.get("complete", False),
            ts_error=ts.get("error", ""),
            trt_error=trt.get("error", ""),
            family=ts.get("family", ""),
            tile_q=ts.get("tile_q"),
            splits=ts.get("split_count"),
            v_partitions=ts.get("head_dim_ctas"),
            direct=ts.get("direct_inputs"),
            valid_topk_min=result.get("valid_topk_min"),
            valid_topk_max=result.get("valid_topk_max"),
            fixture_id=result.get("fixture_id", ""),
        )
        if row["complete"] and row["ts_us"] is not None and row["trt_us"] is not None:
            row["ts_over_trt"] = row["ts_us"] / row["trt_us"]
            row["gap_percent"] = 100 * (row["ts_over_trt"] - 1)
        rows.append(row)
    matched = [r for r in rows if r["ts_over_trt"] is not None]

    def stats(items):
        ratios = [r["ts_over_trt"] for r in items]
        return dict(
            comparable=len(items),
            ts_faster=sum(x < 1 - 1e-6 for x in ratios),
            tied=sum(abs(x - 1) <= 1e-6 for x in ratios),
            ts_slower=sum(x > 1 + 1e-6 for x in ratios),
            within_5_percent=sum(x <= 1.05 for x in ratios),
            median_ts_over_trt=statistics.median(ratios) if ratios else None,
            geometric_mean_ts_over_trt=math.exp(statistics.mean(map(math.log, ratios)))
            if ratios
            else None,
            worst_ts_over_trt=max(ratios) if ratios else None,
        )

    summary = dict(
        requested=len(rows),
        completed=sum(r["complete"] for r in rows),
        failed=[
            {
                k: r[k]
                for k in (
                    "id",
                    "phase",
                    "batch",
                    "queries",
                    "heads",
                    "topk",
                    "dtype",
                    "ts_error",
                    "trt_error",
                )
            }
            for r in rows
            if "failed" in (r["ts_status"], r["trt_status"])
        ],
        overall=stats(matched),
        groups={
            f"{phase}/{dtype}": stats(
                [r for r in matched if r["phase"] == phase and r["dtype"] == dtype]
            )
            for phase in ("prefill", "decode")
            for dtype in ("bf16", "fp8")
        },
        source=manifest,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    introduction = (
        "# Fixed-32K sparse MLA results\n\n"
        f"Completed **{summary['completed']}/{len(rows)}** cases; "
        f"**{len(matched)}** paired comparisons; **{len(summary['failed'])}** cases with backend failures. "
        "Times are medians in microseconds. Positive gap means Prims-TS is slower.\n\n"
        "All cases use identical inputs per backend pair, full-output FP64 checking, "
        "cold 4xL2 eviction before every invocation, and CUDA Graphs. "
        f"Sampling: {manifest['replays']} replays × {manifest['samples_per_replay']} samples. "
        f"GPU: {manifest['gpu']}; source: `{manifest['revision'][:12]}`.\n\n"
        f"Raw context: {manifest['raw_kv_tokens']}; primary compression ratio: "
        f"{manifest['compression_ratio']}; causal suffix: {manifest['causal']}. "
        "K is in addition to 128 SWA slots. Native D512, one KV head, BF16 output. "
        "These are model-shape proxies, not end-to-end model runs.\n\n"
        "[Filterable offline table](results.html) · [CSV](results.csv) · [Summary](summary.json)\n\n"
        "| Phase/type | Paired | TS faster | Tied | TS slower | Within +5% | Median TS/TRT | Worst TS/TRT |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for group, s in summary["groups"].items():
        median = f"{s['median_ts_over_trt']:.3f}" if s["comparable"] else "—"
        worst = f"{s['worst_ts_over_trt']:.3f}" if s["comparable"] else "—"
        introduction += f"|{group}|{s['comparable']}|{s['ts_faster']}|{s['tied']}|{s['ts_slower']}|{s['within_5_percent']}|{median}|{worst}|\n"

    def table(items):
        text = "| ID | Phase | Type | B | Q | H | K | TS µs | TRT µs | Gap | TS profile |\n"
        text += "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        for r in items:
            ts_time = f"{r['ts_us']:.3f}" if r["ts_us"] is not None else r["ts_status"]
            trt_time = (
                f"{r['trt_us']:.3f}" if r["trt_us"] is not None else r["trt_status"]
            )
            gap = f"{r['gap_percent']:+.2f}%" if r["gap_percent"] is not None else "—"
            profile = (
                f"{r['family']} M{r['tile_q']}/S{r['splits']}/V{r['v_partitions']}"
            )
            text += f"|{r['id']}|{r['phase']}|{r['dtype']}|{r['batch']}|{r['queries']}|{r['heads']}|{r['topk']}|{ts_time}|{trt_time}|{gap}|{profile}|\n"
        return text

    slow = sorted(
        [r for r in matched if r["gap_percent"] > 1e-4], key=lambda r: -r["gap_percent"]
    )
    (output / "results.md").write_text(
        introduction + "\n## All requested cases\n\n" + table(rows)
    )
    (output / "slower_than_trt.md").write_text(
        introduction + "\n## All measured slowdowns\n\n" + table(slow)
    )

    payload = json.dumps(rows).replace("<", "\\u003c")
    page = """<!doctype html><html><meta charset="utf-8"><title>Sparse MLA 32K benchmark</title>
<style>body{font:15px system-ui;margin:24px;color:#17212b}table{border-collapse:collapse;width:100%}th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#edf2f6}td:nth-child(2),td:nth-child(3){text-align:left}.slow{color:#ac2424}.fast{color:#146832}select,label{margin-right:16px}header{margin-bottom:18px}</style>
<h1>Sparse MLA: fixed 32K context</h1><p>Native D512 · 128 SWA + K selected candidates · matched inputs · cold L2 + CUDA Graphs. Positive gap means TS is slower.</p>
<header><span id="controls"></span><label><input id="slow" type="checkbox">Only TS slower</label><label><input id="sort" type="checkbox">Largest gap first</label><strong id="count"></strong></header><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table><script>
const rows=PAYLOAD;
const columns=['id','phase','dtype','batch','queries','heads','topk','ts_us','trt_us','gap_percent','ts_status','trt_status','family','tile_q','splits','v_partitions'];
const labels=['ID','Phase','Type','B','Q','H','K','TS µs','TRT µs','Gap %','TS status','TRT status','Family','M','S','V'];
labels.forEach(x=>{const e=document.createElement('th');e.textContent=x;document.getElementById('head').append(e)});
const filters={};for(const field of ['phase','dtype','batch','queries','heads','topk']){const e=document.createElement('select');for(const value of ['All '+field,...new Set(rows.map(x=>x[field]))]){const o=document.createElement('option');o.textContent=value;o.value=value;e.append(o)}e.onchange=render;document.getElementById('controls').append(e);filters[field]=e}
document.getElementById('slow').onchange=render;document.getElementById('sort').onchange=render;
function render(){let selected=rows.filter(r=>Object.entries(filters).every(([k,e])=>e.selectedIndex===0||String(r[k])===e.value));if(document.getElementById('slow').checked)selected=selected.filter(r=>r.gap_percent>0.0001);if(document.getElementById('sort').checked)selected.sort((a,b)=>(b.gap_percent??-Infinity)-(a.gap_percent??-Infinity));document.getElementById('count').textContent=selected.length+' rows';const body=document.getElementById('body');body.replaceChildren();for(const r of selected){const tr=document.createElement('tr');for(const k of columns){const td=document.createElement('td');const v=r[k];td.textContent=v==null?'—':(['ts_us','trt_us','gap_percent'].includes(k)?v.toFixed(k==='gap_percent'?2:3):v);if(k==='gap_percent')td.className=v>0?'slow':'fast';if(k==='ts_status')td.title=r.ts_error;if(k==='trt_status')td.title=r.trt_error;tr.append(td)}body.append(tr)}}render();
</script></html>"""
    (output / "results.html").write_text(page.replace("PAYLOAD", payload))
    print(json.dumps({k: v for k, v in summary.items() if k != "source"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.directory, arguments.output)

#!/usr/bin/env node
"use strict";

/*
Error-structure diagnostics on saved static outputs (paper Section 6.2,
Figures 3-4, Appendix G.1 and G.11):
  - finds the saved per-question JSONL outputs
  - recomputes the composite disagreement score
  - applies adaptive p80 hard-case threshold
  - writes summary tables
  - computes per-agent accuracy, ECE, pairwise error Jaccard and unique-correct
    counts for the largest heterogeneous file
  - writes simple SVG plots

Run from repo root:
  node analysis/error_structure.js

Optional:
  node analysis/error_structure.js --input experiments/static_disagreement/outputs --out results/error_structure
*/

const fs = require("fs");
const path = require("path");

function argValue(name, fallback) {
  const idx = process.argv.indexOf(name);
  if (idx >= 0 && process.argv[idx + 1]) return process.argv[idx + 1];
  return fallback;
}

const ROOT = path.resolve(__dirname, "..");
const INPUT_DIR = path.resolve(ROOT, argValue("--input", "experiments/static_disagreement/outputs"));
const OUT_DIR = path.resolve(ROOT, argValue("--out", "results/error_structure"));
const RESCORED_DIR = path.join(OUT_DIR, "rescored_jsonl");

fs.mkdirSync(RESCORED_DIR, { recursive: true });

function normAnswer(x) {
  const s = String(x ?? "").trim().toUpperCase();
  return s && /^[A-Z]/.test(s[0]) ? s[0] : s;
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function percentile(xs, p) {
  if (!xs.length) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function rank(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const out = Array(values.length).fill(0);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    const r = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) out[order[k][1]] = r;
    i = j + 1;
  }
  return out;
}

function spearman(x, y) {
  if (x.length !== y.length || x.length < 2) return null;
  const rx = rank(x);
  const ry = rank(y);
  const mx = mean(rx);
  const my = mean(ry);
  const num = rx.reduce((acc, v, i) => acc + (v - mx) * (ry[i] - my), 0);
  const dx = Math.sqrt(rx.reduce((acc, v) => acc + (v - mx) ** 2, 0));
  const dy = Math.sqrt(ry.reduce((acc, v) => acc + (v - my) ** 2, 0));
  return dx && dy ? num / (dx * dy) : null;
}

function entropyAnswer(answers, numOptions = 4) {
  const counts = new Map();
  for (const answer of answers.map(normAnswer)) counts.set(answer, (counts.get(answer) || 0) + 1);
  const n = Math.max(answers.length, 1);
  let h = 0;
  for (const c of counts.values()) {
    const p = c / n;
    h -= p * Math.log(p + 1e-12);
  }
  const maxH = Math.log(Math.min(n, numOptions));
  return maxH > 0 ? Math.min(Math.max(h / maxH, 0), 1) : 0;
}

function approxDist(answer, confidence, labels) {
  const conf = Math.min(Math.max(Number(confidence || 0), 0), 1);
  const chosen = normAnswer(answer);
  const dist = Array(labels.length).fill(0);
  const idx = labels.indexOf(chosen);
  if (idx < 0) return dist.map(() => 1 / labels.length);
  dist[idx] = conf;
  const spill = (1 - conf) / Math.max(labels.length - 1, 1);
  for (let i = 0; i < labels.length; i++) if (i !== idx) dist[i] = spill;
  const total = dist.reduce((a, b) => a + b, 0) || 1;
  return dist.map((v) => v / total);
}

function explicitDist(probs, labels) {
  if (!probs || typeof probs !== "object") return null;
  const dist = labels.map((label) => Math.max(Number(probs[label] ?? probs[label.toLowerCase()] ?? 0), 0));
  const total = dist.reduce((a, b) => a + b, 0);
  return total > 0 ? dist.map((v) => v / total) : null;
}

function jsDiv(p, q) {
  const m = p.map((v, i) => 0.5 * (v + q[i]));
  const kl = (a, b) => a.reduce((acc, v, i) => acc + v * Math.log((v + 1e-12) / (b[i] + 1e-12)), 0);
  return Math.min(Math.max(0.5 * (kl(p, m) + kl(q, m)) / Math.log(2), 0), 1);
}

function pairwiseJs(outputs) {
  const labels = [...new Set([
    "A", "B", "C", "D",
    ...outputs.map((o) => normAnswer(o.answer)).filter(Boolean),
    ...outputs.flatMap((o) => o.probs ? Object.keys(o.probs).map(normAnswer) : []),
  ])].sort();
  const dists = outputs.map((o) => explicitDist(o.probs, labels) || approxDist(o.answer, o.confidence, labels));
  const vals = [];
  for (let i = 0; i < dists.length; i++) {
    for (let j = i + 1; j < dists.length; j++) vals.push(jsDiv(dists[i], dists[j]));
  }
  return mean(vals) ?? 0;
}

function confVariance(outputs) {
  const cs = outputs.map((o) => Math.min(Math.max(Number(o.confidence || 0), 0), 1));
  const m = mean(cs) ?? 0;
  const variance = mean(cs.map((c) => (c - m) ** 2)) ?? 0;
  return Math.min(Math.max(variance / (1 / 3), 0), 1);
}

function correctnessDiv(outputs, correct) {
  if (correct == null || correct === "") return 0;
  const gold = normAnswer(correct);
  const vals = outputs.map((o) => normAnswer(o.answer) === gold);
  return new Set(vals).size === 1 ? 0 : 1;
}

function revisedD(outputs, correct) {
  const dAnswer = entropyAnswer(outputs.map((o) => o.answer));
  const dJs = pairwiseJs(outputs);
  const dConf = confVariance(outputs);
  const dCorrect = correctnessDiv(outputs, correct);
  return {
    D_revised: round4(0.45 * dAnswer + 0.35 * dJs + 0.15 * dConf + 0.05 * dCorrect),
    D_answer_entropy: round4(dAnswer),
    D_pairwise_js: round4(dJs),
    D_confidence: round4(dConf),
    D_correctness: round4(dCorrect),
  };
}

function round4(x) {
  return Math.round(Number(x) * 10000) / 10000;
}

function first(row, keys, fallback = null) {
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== null && row[key] !== "") return row[key];
  }
  return fallback;
}

function extractOutputs(row) {
  if (row.per_agent && typeof row.per_agent === "object") {
    const outs = [];
    for (const name of ["R", "D", "C"]) {
      const item = row.per_agent[name];
      if (!item || typeof item !== "object" || item.answer == null) return null;
      outs.push({
        name,
        answer: item.answer,
        confidence: Number(item.confidence ?? item.conf ?? 0),
        probs: item.all_probs || item.probs || item.probabilities || null,
      });
    }
    return outs;
  }

  const raw = row.agent_outputs || row.agents || row.outputs;
  if (Array.isArray(raw)) {
    const outs = raw
      .filter((o) => o && typeof o === "object")
      .map((o, i) => ({
        name: String(o.agent_name || o.name || ["R", "D", "C"][i] || `A${i}`),
        answer: o.answer || o.prediction || "",
        confidence: Number(o.confidence ?? o.conf ?? 0),
        probs: o.all_probs || o.probs || o.probabilities || null,
      }));
    return outs.length === 3 ? outs : null;
  }

  const specs = [
    ["R", ["agent_r_answer", "r_answer", "R_answer", "answer_r"], ["agent_r_confidence", "r_confidence", "R_confidence", "confidence_r"]],
    ["D", ["agent_d_answer", "d_answer", "D_answer", "answer_d"], ["agent_d_confidence", "d_confidence", "D_confidence", "confidence_d"]],
    ["C", ["agent_c_answer", "c_answer", "C_answer", "answer_c"], ["agent_c_confidence", "c_confidence", "C_confidence", "confidence_c"]],
  ];
  const outs = [];
  for (const [name, answerKeys, confKeys] of specs) {
    const answer = first(row, answerKeys);
    if (answer == null) return null;
    outs.push({ name, answer, confidence: Number(first(row, confKeys, 0)) });
  }
  return outs;
}

function readJsonl(file) {
  const text = fs.readFileSync(file, "utf8");
  const rows = [];
  const bad = [];
  text.split(/\r?\n/).forEach((line, idx) => {
    if (!line.trim()) return;
    try {
      rows.push(JSON.parse(line));
    } catch (err) {
      bad.push(idx + 1);
    }
  });
  return { rows, bad };
}

function writeJsonl(file, rows) {
  fs.writeFileSync(file, rows.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");
}

function csvEscape(v) {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCsv(file, rows) {
  if (!rows.length) {
    fs.writeFileSync(file, "", "utf8");
    return;
  }
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const lines = [keys.join(",")];
  for (const row of rows) lines.push(keys.map((k) => csvEscape(row[k])).join(","));
  fs.writeFileSync(file, lines.join("\n") + "\n", "utf8");
}

function listJsonlFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter((name) => name.toLowerCase().endsWith(".jsonl"))
    .map((name) => path.join(dir, name))
    .filter((file) => fs.statSync(file).isFile());
}

function summarizeFile(file) {
  const { rows, bad } = readJsonl(file);
  const rescored = [];
  let dropped = bad.length;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const outputs = extractOutputs(row);
    if (!outputs) {
      dropped++;
      continue;
    }
    const correct = first(row, ["answer", "correct_answer", "gold", "label"]);
    const difficulty = first(row, ["difficulty", "difficulty_score", "task1_difficulty", "score"]);
    rescored.push({
      ...row,
      agent_outputs_extracted: outputs,
      correct_answer_norm: normAnswer(correct),
      difficulty_for_corr: difficulty,
      ...revisedD(outputs, correct),
    });
  }
  const scores = rescored.map((r) => Number(r.D_revised));
  const threshold = percentile(scores, 80);
  for (const row of rescored) row.is_hard_case_revised = Number(row.D_revised) >= threshold;
  const paired = rescored
    .map((r) => [Number(r.D_revised), Number(r.difficulty_for_corr)])
    .filter(([, d]) => Number.isFinite(d));

  const summary = {
    file: path.basename(file),
    n_total: rows.length + bad.length,
    n_scored: rescored.length,
    n_dropped: dropped,
    adaptive_threshold_p80: round4(threshold ?? 0),
    hard_case_rate_revised: round4(mean(rescored.map((r) => r.is_hard_case_revised ? 1 : 0)) ?? 0),
    mean_D_revised: round4(mean(scores) ?? 0),
    spearman_rho_revised: paired.length ? round4(spearman(paired.map((p) => p[0]), paired.map((p) => p[1])) ?? 0) : "",
    mean_answer_entropy: round4(mean(rescored.map((r) => r.D_answer_entropy)) ?? 0),
    mean_pairwise_js: round4(mean(rescored.map((r) => r.D_pairwise_js)) ?? 0),
    mean_confidence: round4(mean(rescored.map((r) => r.D_confidence)) ?? 0),
    mean_correctness_div: round4(mean(rescored.map((r) => r.D_correctness)) ?? 0),
  };

  const outFile = path.join(RESCORED_DIR, `${path.basename(file, ".jsonl")}_rescored.jsonl`);
  writeJsonl(outFile, rescored);
  return { summary, rescored, outFile };
}

function agentOutput(row, name) {
  const outs = row.agent_outputs_extracted || extractOutputs(row) || [];
  return outs.find((o) => String(o.name).toUpperCase().includes(name)) || outs.find((o) => o.name === name);
}

function jaccardError(records, a, b) {
  const ea = new Set(records.map((r, i) => !r[`${a}_correct`] ? i : null).filter((x) => x !== null));
  const eb = new Set(records.map((r, i) => !r[`${b}_correct`] ? i : null).filter((x) => x !== null));
  const union = new Set([...ea, ...eb]);
  const inter = [...ea].filter((x) => eb.has(x));
  return union.size ? inter.length / union.size : 0;
}

function ece(records, agent, bins = 10) {
  let val = 0;
  for (let b = 0; b < bins; b++) {
    const lo = b / bins;
    const hi = (b + 1) / bins;
    const bucket = records.filter((r) => r[`${agent}_conf`] >= lo && (b === bins - 1 ? r[`${agent}_conf`] <= hi : r[`${agent}_conf`] < hi));
    if (!bucket.length) continue;
    const acc = mean(bucket.map((r) => r[`${agent}_correct`] ? 1 : 0));
    const conf = mean(bucket.map((r) => r[`${agent}_conf`]));
    val += (bucket.length / records.length) * Math.abs(acc - conf);
  }
  return val;
}

function orthogonality(rows, sourceName) {
  const records = [];
  for (const row of rows) {
    const gold = normAnswer(first(row, ["answer", "correct_answer", "gold", "label", "correct_answer_norm"]));
    if (!gold) continue;
    const r = agentOutput(row, "R");
    const d = agentOutput(row, "D");
    const c = agentOutput(row, "C");
    if (!r || !d || !c) continue;
    records.push({
      source_file: sourceName,
      question_id: first(row, ["question_id", "id"], records.length),
      question: first(row, ["question", "prompt", "text"], ""),
      correct: gold,
      R_answer: normAnswer(r.answer),
      D_answer: normAnswer(d.answer),
      C_answer: normAnswer(c.answer),
      R_correct: normAnswer(r.answer) === gold,
      D_correct: normAnswer(d.answer) === gold,
      C_correct: normAnswer(c.answer) === gold,
      R_conf: Number(r.confidence || 0),
      D_conf: Number(d.confidence || 0),
      C_conf: Number(c.confidence || 0),
      D_revised: Number(row.D_revised || 0),
      D_pairwise_js: Number(row.D_pairwise_js || 0),
    });
  }
  const agents = ["R", "D", "C"];
  const summary = {
    source_file: sourceName,
    n: records.length,
    R_accuracy: round4(mean(records.map((r) => r.R_correct ? 1 : 0)) ?? 0),
    D_accuracy: round4(mean(records.map((r) => r.D_correct ? 1 : 0)) ?? 0),
    C_accuracy: round4(mean(records.map((r) => r.C_correct ? 1 : 0)) ?? 0),
    RD_error_jaccard: round4(jaccardError(records, "R", "D")),
    RC_error_jaccard: round4(jaccardError(records, "R", "C")),
    DC_error_jaccard: round4(jaccardError(records, "D", "C")),
    R_only_correct: records.filter((r) => r.R_correct && !r.D_correct && !r.C_correct).length,
    D_only_correct: records.filter((r) => !r.R_correct && r.D_correct && !r.C_correct).length,
    C_only_correct: records.filter((r) => !r.R_correct && !r.D_correct && r.C_correct).length,
    R_ECE: round4(ece(records, "R")),
    D_ECE: round4(ece(records, "D")),
    C_ECE: round4(ece(records, "C")),
    mean_pairwise_JS: round4(mean(records.map((r) => r.D_pairwise_js)) ?? 0),
  };
  return { records, summary };
}

function svgBar(rows, key, title, file) {
  const width = 980, height = 420, margin = { left: 64, right: 20, top: 48, bottom: 130 };
  const vals = rows.map((r) => Number(r[key]) || 0);
  const maxVal = Math.max(...vals.map(Math.abs), 0.05);
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const barW = chartW / Math.max(rows.length, 1) * 0.7;
  const zeroY = margin.top + chartH * (maxVal / (2 * maxVal));
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">`,
    `<rect width="100%" height="100%" fill="white"/>`,
    `<text x="${width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">${title}</text>`,
    `<line x1="${margin.left}" y1="${zeroY}" x2="${width - margin.right}" y2="${zeroY}" stroke="#444"/>`];
  rows.forEach((r, i) => {
    const x = margin.left + (i + 0.15) * (chartW / rows.length);
    const v = Number(r[key]) || 0;
    const h = Math.abs(v) / (2 * maxVal) * chartH;
    const y = v >= 0 ? zeroY - h : zeroY;
    parts.push(`<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="#2f6f9f"/>`);
    parts.push(`<text x="${x + barW / 2}" y="${v >= 0 ? y - 5 : y + h + 15}" text-anchor="middle" font-family="Arial" font-size="11">${round4(v)}</text>`);
    parts.push(`<text x="${x + barW / 2}" y="${height - 10}" text-anchor="end" transform="rotate(-35 ${x + barW / 2},${height - 10})" font-family="Arial" font-size="11">${r.file}</text>`);
  });
  parts.push(`</svg>`);
  fs.writeFileSync(file, parts.join("\n"), "utf8");
}

function svgScatter(rows, xKey, yKey, title, file) {
  const width = 760, height = 500, margin = { left: 70, right: 30, top: 50, bottom: 70 };
  const xs = rows.map((r) => Number(r[xKey]) || 0);
  const ys = rows.map((r) => Number(r[yKey]) || 0);
  const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1e-3);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1e-3);
  const sx = (x) => margin.left + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (width - margin.left - margin.right);
  const sy = (y) => height - margin.bottom - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (height - margin.top - margin.bottom);
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">`,
    `<rect width="100%" height="100%" fill="white"/>`,
    `<text x="${width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18">${title}</text>`,
    `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#444"/>`,
    `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#444"/>`,
    `<text x="${width / 2}" y="${height - 20}" text-anchor="middle" font-family="Arial" font-size="13">${xKey}</text>`,
    `<text x="20" y="${height / 2}" transform="rotate(-90 20,${height / 2})" text-anchor="middle" font-family="Arial" font-size="13">${yKey}</text>`];
  rows.forEach((r) => {
    const x = sx(Number(r[xKey]) || 0), y = sy(Number(r[yKey]) || 0);
    parts.push(`<circle cx="${x}" cy="${y}" r="5" fill="#b1443c"/>`);
    parts.push(`<text x="${x + 8}" y="${y - 6}" font-family="Arial" font-size="10">${r.file.slice(0, 26)}</text>`);
  });
  parts.push(`</svg>`);
  fs.writeFileSync(file, parts.join("\n"), "utf8");
}

function main() {
  const files = listJsonlFiles(INPUT_DIR);
  if (!files.length) {
    console.error(`No JSONL files found in ${INPUT_DIR}`);
    process.exit(1);
  }

  // Only files with the heterogeneous `per_agent` schema can be scored here;
  // analysis/saved_output_analysis.py handles the homogeneous `per_role` schema.
  const all = files.map(summarizeFile).filter((x) => {
    if (x.rescored.length) return true;
    fs.rmSync(x.outFile, { force: true });
    console.log(`Skipping ${x.summary.file}: no rows in the per_agent schema`);
    return false;
  });
  const summaries = all.map((x) => x.summary);
  writeCsv(path.join(OUT_DIR, "rescored_summary.csv"), summaries);
  fs.writeFileSync(path.join(OUT_DIR, "rescored_summary.json"), JSON.stringify(summaries, null, 2), "utf8");

  const hetero = all
    .filter((x) => /heterogeneous|mistral|qwen/i.test(x.summary.file) && x.rescored.length)
    .sort((a, b) => b.rescored.length - a.rescored.length)[0] || all.sort((a, b) => b.rescored.length - a.rescored.length)[0];

  const orth = orthogonality(hetero.rescored, hetero.summary.file);
  writeCsv(path.join(OUT_DIR, "orthogonality_records.csv"), orth.records);
  writeCsv(path.join(OUT_DIR, "orthogonality_summary.csv"), [orth.summary]);
  fs.writeFileSync(path.join(OUT_DIR, "orthogonality_summary.json"), JSON.stringify(orth.summary, null, 2), "utf8");

  const qualitative = [...orth.records].sort((a, b) => b.D_revised - a.D_revised).slice(0, 20);
  writeCsv(path.join(OUT_DIR, "qualitative_examples.csv"), qualitative);

  svgBar(summaries, "spearman_rho_revised", "Revised disagreement vs difficulty", path.join(OUT_DIR, "plot_revised_rho.svg"));
  svgBar(summaries, "hard_case_rate_revised", "Adaptive p80 hard-case rate", path.join(OUT_DIR, "plot_hard_case_rate.svg"));
  svgScatter(summaries, "mean_pairwise_js", "spearman_rho_revised", "Diversity vs verification signal", path.join(OUT_DIR, "plot_diversity_vs_rho.svg"));

  console.log("\nDONE error-structure analysis");
  console.log(`Input:  ${INPUT_DIR}`);
  console.log(`Output: ${OUT_DIR}`);
  console.log("\nMain files:");
  console.log(`- ${path.join(OUT_DIR, "rescored_summary.csv")}`);
  console.log(`- ${path.join(OUT_DIR, "orthogonality_summary.csv")}`);
  console.log(`- ${path.join(OUT_DIR, "qualitative_examples.csv")}`);
  console.log(`- ${path.join(OUT_DIR, "plot_revised_rho.svg")}`);
  console.log(`- ${path.join(OUT_DIR, "plot_hard_case_rate.svg")}`);
  console.log(`- ${path.join(OUT_DIR, "plot_diversity_vs_rho.svg")}`);
  console.log("\nSummary:");
  console.table(summaries);
  console.log("\nOrthogonality:");
  console.table([orth.summary]);
}

main();

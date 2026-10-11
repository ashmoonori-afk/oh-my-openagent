#!/usr/bin/env bun
// Independent-author retrieval comparison on the actual omo memory.search selector.
// Corpus blobs come from a pinned Git revision, not today's edited documentation.
// Run practice first and freeze its chosen cap before running evaluation:
//   bun packages/omo-senpi/scripts/qa/recall-chunks-bench.mjs --split practice --json
//   bun packages/omo-senpi/scripts/qa/recall-chunks-bench.mjs --split evaluation --json
// Expansions are frozen author-written terms, not timed live model generation.

import { createHash } from "node:crypto"
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { selectRecallCandidates, chooseRecallStrategy } from "@oh-my-opencode/memory-core"
import { prepareRecallBm25Index } from "../../../memory-core/src/recall/bm25"
import { chunkRecallDocuments } from "../../../memory-core/src/recall/chunks"

const fixtureRoot = new URL("./fixtures/", import.meta.url)
const repoRoot = fileURLToPath(new URL("../../../../", import.meta.url))
const splitArg = process.argv.indexOf("--split")
const split = splitArg < 0 ? "practice" : process.argv[splitArg + 1]
if (split !== "practice" && split !== "evaluation") throw new Error("--split must be practice or evaluation")
const sha256 = (text) => createHash("sha256").update(text).digest("hex")
const manifestBytes = readFileSync(new URL("recall-chunks-manifest.json", fixtureRoot), "utf8")
const manifest = JSON.parse(manifestBytes)
const blob = (path) => execFileSync("git", ["show", `${manifest.revision}:${path}`], { cwd: repoRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 })
const shortBytes = blob(manifest.shortCorpus)
const documents = [...JSON.parse(shortBytes).documents]
const inputs = [{ path: manifest.shortCorpus, sha256: sha256(shortBytes) }]
for (const source of manifest.sources) {
  const body = blob(source)
  const description = /^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$/mu.exec(body)?.[1] ?? source.split("/").at(-1)
  documents.push({ path: `public/${source}`, description, body })
  inputs.push({ path: source, sha256: sha256(body) })
}
const byPath = new Map(documents.map((document) => [document.path, document]))
const authors = ["sol", "claude"].map((name) => {
  const bytes = readFileSync(new URL(`recall-chunks-${name}.json`, fixtureRoot), "utf8")
  const dataset = JSON.parse(bytes)
  if (!Array.isArray(dataset.questions)) throw new Error(`${name}: questions must be an array`)
  const ids = new Set()
  for (const question of dataset.questions) {
    if (ids.has(question.id)) throw new Error(`${name}: duplicate id ${question.id}`)
    ids.add(question.id)
    if (!["practice", "evaluation"].includes(question.split) || !manifest.questionProtocol.slices.includes(question.slice)) {
      throw new Error(`${name}: invalid split or slice ${question.id}`)
    }
    if (typeof question.query !== "string" || question.query.trim() === "" || question.gold.length === 0) {
      throw new Error(`${name}: empty query/gold ${question.id}`)
    }
    for (const path of question.gold) {
      if (!byPath.has(path)) throw new Error(`${name}: nonexistent gold ${path}`)
    }
    if (!question.gold.some((path) => byPath.get(path).body.replace(/\s+/g, " ").includes(question.evidence.replace(/\s+/g, " ")))) {
      throw new Error(`${name}: evidence not found ${question.id}`)
    }
  }
  for (const part of ["practice", "evaluation"]) {
    const count = part === "practice" ? 4 : 12
    for (const slice of manifest.questionProtocol.slices) {
      if (dataset.questions.filter((question) => question.split === part && question.slice === slice).length !== count) {
        throw new Error(`${name}: wrong ${part}/${slice} count`)
      }
    }
  }
  return { name, author: dataset.author, model: dataset.model, sha256: sha256(bytes), questions: dataset.questions.filter((question) => question.split === split) }
})

const percentile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)] ?? 0
}
const sizes = [0, 1000, 2000, 4000]
const builds = []
// An independent fresh array each sample ensures both WeakMap caches are cold.
// This reports projection and pure index construction separately; no query is included.
for (const size of sizes) {
  const projectionTimes = []
  const indexTimes = []
  let chunks = documents.length
  for (let sample = 0; sample < 7; sample += 1) {
    const fresh = documents.map((document) => ({ ...document }))
    const beforeProjection = performance.now()
    const projected = size === 0 ? fresh : chunkRecallDocuments(fresh, size)
    projectionTimes.push(performance.now() - beforeProjection)
    chunks = projected.length
    const beforeIndex = performance.now()
    prepareRecallBm25Index(projected)
    indexTimes.push(performance.now() - beforeIndex)
  }
  builds.push({ size, notes: documents.length, chunks, projectionMedianMs: percentile(projectionTimes, 0.5), indexMedianMs: percentile(indexTimes, 0.5), samples: 7 })
}

const details = []
const rows = []
for (const author of authors) {
  for (const size of sizes) {
    for (const expansion of [false, true]) {
      const options = {
        maxItems: documents.length,
        surfaced: new Set(),
        ...(size === 0 ? {} : { chunking: true, chunkChars: size }),
      }
      const results = []
      for (const question of author.questions) {
        const params = expansion ? { ...options, expansions: question.expansions } : options
        selectRecallCandidates(documents, [question.query], params)
        const timings = []
        let selected
        for (let repetition = 0; repetition < 5; repetition += 1) {
          const start = performance.now()
          selected = selectRecallCandidates(documents, [question.query], params)
          timings.push(performance.now() - start)
        }
        const paths = selected.map((candidate) => candidate.path)
        if (new Set(paths).size !== paths.length) throw new Error(`duplicate note hits ${author.name}/${question.id}/${size}`)
        const offset = paths.findIndex((path) => question.gold.includes(path))
        const result = {
          author: author.name, id: question.id, slice: question.slice, size, expansion,
          rank: offset < 0 ? null : offset + 1,
          rr: offset < 0 ? 0 : 1 / (offset + 1),
          recall5: offset >= 0 && offset < 5 ? 1 : 0,
          medianMs: percentile(timings, 0.5),
          strategy: chooseRecallStrategy(documents, [question.query]),
          top5: paths.slice(0, 5),
        }
        results.push(result)
        details.push(result)
      }
      for (const slice of ["all", ...manifest.questionProtocol.slices]) {
        const selected = slice === "all" ? results : results.filter((result) => result.slice === slice)
        rows.push({
          author: author.name, size, expansion, slice, queries: selected.length,
          MRR: selected.reduce((sum, result) => sum + result.rr, 0) / selected.length,
          "R@5": selected.reduce((sum, result) => sum + result.recall5, 0) / selected.length,
          recallMedianMs: percentile(selected.map((result) => result.medianMs), 0.5),
          recallP95Ms: percentile(selected.map((result) => result.medianMs), 0.95),
        })
      }
    }
  }
}
// Fixed practice-only objective, never selected using frozen evaluation rows.
const choices = split === "practice" ? sizes.filter((size) => size !== 0).map((size) => {
  const all = rows.filter((row) => row.size === size && row.slice === "all")
  return {
    size,
    MRR: all.reduce((sum, row) => sum + row.MRR, 0) / all.length,
    "R@5": all.reduce((sum, row) => sum + row["R@5"], 0) / all.length,
  }
}).sort((a, b) => b.MRR - a.MRR || b["R@5"] - a["R@5"] || b.size - a.size) : undefined
const regressions = rows.filter((row) => row.size !== 0 && row.slice !== "all").flatMap((row) => {
  const baseline = rows.find((base) => base.author === row.author && base.size === 0 && base.expansion === row.expansion && base.slice === row.slice)
  const mrrDelta = row.MRR - baseline.MRR
  const recallDelta = row["R@5"] - baseline["R@5"]
  return mrrDelta < -1e-12 || recallDelta < -1e-12 ? [{ ...row, mrrDelta, recall5Delta: recallDelta }] : []
})
const output = {
  split, runtime: Bun.version, revision: manifest.revision, manifestSha256: sha256(manifestBytes),
  corpusSha256: sha256(JSON.stringify(inputs)), inputs,
  authors: authors.map(({ name, author, model, sha256, questions }) => ({ name, author, model, sha256, queries: questions.length })),
  timing: "7 cold projection/index builds; 5 warm full-selection queries per question; excludes Git loading, planner, model expansion generation and judge",
  builds, rows, details, regressions,
  ...(choices === undefined ? {} : { practiceChoices: choices, selectedSize: choices[0].size }),
}
if (process.argv.includes("--json")) console.log(JSON.stringify(output, null, 2))
else {
  console.log(`split=${split}, corpus=${output.corpusSha256}, notes=${documents.length}`)
  console.log("| author | size | expansion | slice | n | MRR | R@5 | warm median ms | p95 ms |")
  console.log("|---|---:|---|---|---:|---:|---:|---:|---:|")
  for (const row of rows) console.log(`| ${row.author} | ${row.size || "whole"} | ${row.expansion} | ${row.slice} | ${row.queries} | ${row.MRR.toFixed(4)} | ${row["R@5"].toFixed(4)} | ${row.recallMedianMs.toFixed(3)} | ${row.recallP95Ms.toFixed(3)} |`)
  console.log(JSON.stringify({ builds, practiceChoices: choices, regressions }, null, 2))
}

import { describe, expect, it } from "bun:test"
import type { RecallDocument } from "./provider"
import { selectRecallCandidates } from "./select"

const OPTIONS = { maxItems: 5, surfaced: new Set<string>(), chunking: true, chunkChars: 200 }
const longNote: RecallDocument = {
  path: "notes/nebula.md",
  description: "Operations archive",
  body: `# Operations\n## Old incident\nnebula ${"ordinary log entry ".repeat(100)}\n## Listener\nnebula port is 4317; restart the collector here.\n`,
}

describe("selectRecallCandidates with heading chunks", () => {
  it.each(["bm25", "hybrid"] as const)("#given a weak early hit and a strong section #when %s ranks chunks #then the winning section supplies the excerpt", (strategy) => {
    const selected = selectRecallCandidates([longNote], ["nebula"], { ...OPTIONS, strategy })
    expect(selected).toHaveLength(1)
    expect(selected[0]?.path).toBe(longNote.path)
    expect(selected[0]?.description).toBe(longNote.description)
    expect(selected[0]?.excerpt).toContain("port is 4317")
    expect(selected[0]?.excerpt).not.toContain("ordinary log entry")
  })

  it("#given many matching sections of one note #when capped #then other notes still occupy their own slots", () => {
    const documents = [
      { path: "notes/a.md", description: "Rules", body: Array.from({ length: 20 }, (_, index) => `## Section ${index}\nnebula listener\n`).join("") },
      { path: "notes/b.md", description: "Rules", body: "nebula listener" },
      { path: "notes/c.md", description: "Rules", body: "nebula listener" },
    ]
    const selected = selectRecallCandidates(documents, ["nebula"], { ...OPTIONS, maxItems: 2, strategy: "hybrid" })
    expect(selected).toHaveLength(2)
    expect(new Set(selected.map((candidate) => candidate.path)).size).toBe(2)
  })

  it("#given surfaced and excluded notes #when chunks match #then exclusions apply before the note cap", () => {
    const other = { ...longNote, path: "notes/other.md" }
    const selected = selectRecallCandidates([longNote, other], ["nebula"], {
      ...OPTIONS, maxItems: 1, strategy: "bm25",
      surfaced: new Set([longNote.path]),
    })
    expect(selected.map((candidate) => candidate.path)).toEqual([other.path])
    expect(selectRecallCandidates([longNote, other], ["nebula"], {
      ...OPTIONS, strategy: "bm25", excludePaths: new Set([longNote.path, other.path]),
    })).toEqual([])
  })

  it("#given a small English corpus with terms in different headings #when chunking is enabled #then note-level substring matching and strategy stay intact", () => {
    const documents = [{ path: "notes/a.md", description: "Guide", body: `## First\nnebula ${"padding ".repeat(60)}\n## Second\nlistener` }]
    const off = { maxItems: 5, surfaced: new Set<string>() }
    const plain = selectRecallCandidates(documents, ["nebula listener"], off)
    expect(plain.map((candidate) => candidate.path)).toEqual(["notes/a.md"])
    expect(selectRecallCandidates(documents, ["nebula listener"], OPTIONS)).toEqual(plain)
  })

  it("#given query words spanning Korean sections #when expansion finds other notes #then the original full-match note keeps priority", () => {
    const documents = [
      { path: "notes/a.md", description: "운영", body: "## 연결\n원격 접속\n## 장비\n서버 설정\n" },
      { path: "notes/b.md", description: "번들", body: "번들 번들 번들" },
    ]
    const plain = selectRecallCandidates(documents, ["원격 서버"], { ...OPTIONS, strategy: "bm25" })
    const expanded = selectRecallCandidates(documents, ["원격 서버"], {
      ...OPTIONS, strategy: "bm25", expansions: { synonyms: ["번들"] },
    })
    expect(plain[0]?.path).toBe("notes/a.md")
    expect(expanded[0]?.path).toBe("notes/a.md")
    expect(expanded.map((candidate) => candidate.path)).toEqual(["notes/a.md", "notes/b.md"])
  })

  it("#given chunking disabled or omitted #when expansions run #then all legacy candidate fields are identical", () => {
    const off = { maxItems: 5, surfaced: new Set<string>(), expansions: { synonyms: ["collector"] } }
    expect(selectRecallCandidates([longNote], ["nebula"], { ...off, chunking: false, chunkChars: 200 }))
      .toEqual(selectRecallCandidates([longNote], ["nebula"], off))
  })

  it("#given an emoji at the excerpt boundary #when chunked BM25 returns it #then no dangling surrogate reaches the judge", () => {
    const documents = [{
      path: "notes/emoji.md", description: "Symbols",
      body: `nebula ${"x".repeat(192)}🙂 tail`,
    }]
    const selected = selectRecallCandidates(documents, ["nebula"], { ...OPTIONS, chunkChars: 1000, strategy: "bm25" })
    expect(selected[0]?.excerpt).toContain("nebula")
    expect(selected[0]?.excerpt).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u)
    expect(selected[0]?.excerpt.length).toBeLessThanOrEqual(200)
  })
})

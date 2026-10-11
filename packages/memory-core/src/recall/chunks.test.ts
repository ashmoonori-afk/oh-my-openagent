import { describe, expect, it } from "bun:test"
import { chunkRecallDocuments } from "./chunks"

function documents(body: string, description = "Archive") {
  return [{ path: "notes/a.md", description, body }]
}

describe("chunkRecallDocuments", () => {
  it("#given headings and a preamble #when projected #then section bodies stay separate and every chunk indexes the note title", () => {
    const source = documents("Preamble\n# Notebook\nIntroduction\n## First\none\n## Second\n two\n")
    const chunks = chunkRecallDocuments(source, 1000)
    expect(chunks.map((chunk) => chunk.body)).toEqual([
      "Preamble\n", "# Notebook\nIntroduction\n", "## First\none\n", "## Second\n two\n",
    ])
    for (const chunk of chunks) {
      expect(chunk.path).toBe("notes/a.md")
      expect(chunk.description).toContain("Notebook")
    }
    expect(chunks.map((chunk) => chunk.body).join("")).toBe(source[0]?.body)
  })

  it("#given fenced examples #when heading boundaries are found #then fake ATX and Setext headings never split the code", () => {
    const source = documents("## Real\n```ts\n# fake\n```\n~~~\nFake\n====\n~~~\n## Last\nend")
    const chunks = chunkRecallDocuments(source, 1000)
    expect(chunks).toHaveLength(2)
    expect(chunks[0]?.body).toContain("# fake\n```\n~~~\nFake\n====")
    expect(chunks[1]?.body).toBe("## Last\nend")
  })

  it("#given multiline Setext and a blank-separated thematic break #when projected #then the heading begins at the whole title paragraph", () => {
    const source = documents("Preamble.\n\nHeading\non two lines\n====\nbody\n\n---\n")
    const chunks = chunkRecallDocuments(source, 1000)
    expect(chunks.map((chunk) => chunk.body)).toEqual([
      "Preamble.\n\n", "Heading\non two lines\n====\nbody\n\n---\n",
    ])
    expect(chunks[0]?.description).toContain("Heading on two lines")
  })

  it("#given an unclosed fence and indented code #when scanned #then apparent headings stay within the original section", () => {
    const source = documents("# Note\n    ## indented code\n````\n## fake\n```\n~~~\n## still fake\n")
    expect(chunkRecallDocuments(source, 1000)).toHaveLength(1)
  })

  it("#given an oversized Unicode section #when capped #then every body fits without losing characters or splitting surrogate pairs", () => {
    const source = documents(`# Note\n${"𠮷🙂㍿".repeat(100)}\n`)
    const chunks = chunkRecallDocuments(source, 31)
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      expect(Array.from(chunk.body).length).toBeLessThanOrEqual(31)
      expect(chunk.body).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/)
    }
    expect(chunks.map((chunk) => chunk.body).join("")).toBe(source[0]?.body)
  })

  it("#given a fenced block larger than the cap #when split #then continuation text is not reparsed as a note heading", () => {
    const source = documents(`# Real title\n\`\`\`\n${"padding ".repeat(60)}\n# fake title\n\`\`\`\n`)
    const chunks = chunkRecallDocuments(source, 50)
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) expect(chunk.description).toBe("Archive\nReal title")
    expect(chunks.map((chunk) => chunk.body).join("")).toBe(source[0]?.body)
  })

  it("#given an empty or description-only note #when projected #then metadata remains searchable", () => {
    expect(chunkRecallDocuments(documents("", "Useful title"), 1000)).toEqual([
      { path: "notes/a.md", description: "Useful title", body: "" },
    ])
    expect(chunkRecallDocuments([], 1000)).toEqual([])
  })

  it("#given unchanged immutable notes #when reused or replaced #then projection identity follows array and size without stale bodies", () => {
    const source = documents("# Note\nold body")
    const first = chunkRecallDocuments(source, 1000)
    expect(chunkRecallDocuments(source, 1000)).toBe(first)
    expect(chunkRecallDocuments(source, 2000)).not.toBe(first)
    const updated = documents("# Note\nnew body")
    expect(chunkRecallDocuments(updated, 1000)[0]?.body).toContain("new body")
    expect(first[0]?.body).toContain("old body")
  })
})

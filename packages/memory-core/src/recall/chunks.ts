// A scoring-only projection of committed notes. Markdown storage and read output
// never change. The cap counts body code points before NFKC; title/description
// metadata is repeated outside that cap and contributes to index cost.
import type { RecallDocument } from "./provider"

/** Fixed from the practice split of recall-chunks-bench.mjs; not a public setting. */
export const RECALL_CHUNK_CHARS = 4000

const PROJECTIONS = new WeakMap<readonly RecallDocument[], Map<number, readonly RecallDocument[]>>()

/** Recognize headings before any oversized section is split, so fenced continuations stay code. */
function sections(body: string): { readonly bodies: readonly string[]; readonly title: string | undefined } {
  const starts = [0]
  let title: string | undefined
  let offset = 0
  let fenceCharacter = ""
  let fenceLength = 0
  let paragraphStart: number | undefined
  let paragraphTitle: string[] = []
  for (const raw of body.split(/(?<=\n)/u)) {
    const start = offset
    offset += raw.length
    const line = raw.replace(/\r?\n$/u, "")
    if (fenceCharacter !== "") {
      const close = /^ {0,3}(`{3,}|~{3,})[ \t]*$/u.exec(line)?.[1]
      if (close !== undefined && close[0] === fenceCharacter && close.length >= fenceLength) {
        fenceCharacter = ""
        fenceLength = 0
      }
      continue
    }
    const open = /^ {0,3}(`{3,}|~{3,})(.*)$/u.exec(line)
    const delimiter = open?.[1]
    if (delimiter !== undefined && !(delimiter[0] === "`" && open?.[2]?.includes("`"))) {
      fenceCharacter = delimiter[0] ?? ""
      fenceLength = delimiter.length
      paragraphStart = undefined
      paragraphTitle = []
      continue
    }
    const atx = /^ {0,3}#{1,6}(?:[ \t]+(.*)|[ \t]*)$/u.exec(line)
    if (atx !== null) {
      if (starts.at(-1) !== start) starts.push(start)
      const heading = (atx[1] ?? "").replace(/[ \t]+#+[ \t]*$/u, "").trim()
      if (heading !== "" && title === undefined) title = heading
      paragraphStart = undefined
      paragraphTitle = []
      continue
    }
    if (/^ {0,3}(?:=+|-+)[ \t]*$/u.test(line) && paragraphStart !== undefined) {
      if (starts.at(-1) !== paragraphStart) starts.push(paragraphStart)
      title ??= paragraphTitle.join(" ")
      paragraphStart = undefined
      paragraphTitle = []
      continue
    }
    // Top-level headings only: indented code, lists and blockquotes are not title paragraphs.
    if (line.trim() === "" || /^(?: {4}|\t| {0,3}(?:>|[-+*][ \t]|\d+[.)][ \t]))/u.test(line)) {
      paragraphStart = undefined
      paragraphTitle = []
      continue
    }
    paragraphStart ??= start
    paragraphTitle.push(line.trim())
  }
  return { bodies: starts.map((start, index) => body.slice(start, starts[index + 1] ?? body.length)), title }
}

/**
 * Immutable per-(original array, cap) view, matching the BM25 index's cache
 * lifetime. Callers supply a positive integer cap; production uses the fixed
 * constant and the benchmark uses 1000/2000/4000. Never mutate cached arrays.
 */
export function chunkRecallDocuments(documents: readonly RecallDocument[], maxChars: number): readonly RecallDocument[] {
  const cached = PROJECTIONS.get(documents)?.get(maxChars)
  if (cached !== undefined) return cached
  const chunks: RecallDocument[] = []
  for (const document of documents) {
    const parsed = sections(document.body)
    const description = parsed.title === undefined || parsed.title === document.description
      ? document.description
      : `${document.description}\n${parsed.title}`
    for (const section of parsed.bodies) {
      const characters = Array.from(section)
      if (characters.length === 0) {
        chunks.push({ path: document.path, description, body: "" })
        continue
      }
      for (let start = 0; start < characters.length;) {
        let end = Math.min(start + maxChars, characters.length)
        if (end < characters.length) {
          // Prefer a line, then whitespace, without turning a long line into tiny chunks.
          const floor = start + Math.floor(maxChars / 2)
          let whitespace: number | undefined
          for (let position = end - 1; position >= floor; position -= 1) {
            const character = characters[position] ?? ""
            if (character === "\n") {
              end = position + 1
              whitespace = undefined
              break
            }
            if (whitespace === undefined && /\s/u.test(character)) whitespace = position + 1
          }
          if (whitespace !== undefined) end = whitespace
        }
        chunks.push({ path: document.path, description, body: characters.slice(start, end).join("") })
        start = end
      }
    }
  }
  const bySize = PROJECTIONS.get(documents) ?? new Map<number, readonly RecallDocument[]>()
  bySize.set(maxChars, chunks)
  PROJECTIONS.set(documents, bySize)
  return chunks
}

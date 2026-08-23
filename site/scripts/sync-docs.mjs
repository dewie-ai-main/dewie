// Copy the repo's docs/*.md into the Starlight content collection at build
// time, so docs/ stays the single source of truth (no duplicated content in
// git). Adds the frontmatter Starlight needs and normalizes cross-doc links.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const docsDir = resolve(here, "../../docs");
const outDir = resolve(here, "../src/content/docs");
mkdirSync(outDir, { recursive: true });

// slug -> source file (only the public docs; internal docs stay gitignored)
const PAGES = {
  quickstart: "quickstart.md",
  "mcp-tools": "mcp-tools.md",
  configuration: "configuration.md",
  deployment: "deployment.md",
};

for (const [slug, file] of Object.entries(PAGES)) {
  const raw = readFileSync(resolve(docsDir, file), "utf8");
  const lines = raw.split("\n");

  // First "# H1" becomes the Starlight title; drop it from the body.
  let title = slug;
  const h1 = lines.findIndex((l) => /^#\s+/.test(l));
  if (h1 !== -1) {
    title = lines[h1].replace(/^#\s+/, "").trim();
    lines.splice(h1, 1);
  }

  let body = lines.join("\n");
  // Normalize cross-doc links so Astro resolves them (e.g. "docs/x.md",
  // "x.md" -> "./x.md" within the collection).
  body = body.replace(
    /\]\((?:\.\/)?(?:docs\/)?(quickstart|mcp-tools|configuration|deployment)\.md(#[^)]*)?\)/g,
    "](./$1.md$2)",
  );

  const fm = `---\ntitle: ${JSON.stringify(title)}\n---\n\n`;
  writeFileSync(resolve(outDir, `${slug}.md`), fm + body);
  console.log(`synced ${file} -> src/content/docs/${slug}.md`);
}

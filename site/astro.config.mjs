// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// Docs content is synced from ../docs at build time (see scripts/sync-docs.mjs),
// so docs/ stays the single source of truth in the repo root.
export default defineConfig({
  site: "https://dewie-ai-main.github.io",
  base: "/dewie",
  integrations: [
    starlight({
      title: "Dewie",
      description:
        "Agent-native retrieval: navigation tools over a self-enriching corpus.",
      pagefind: false,
      customCss: ["./src/styles/theme.css"],
      components: {
        SocialIcons: "./src/components/HeaderSocials.astro",
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/dewie-ai-main/dewie",
        },
      ],
      sidebar: [
        { label: "Quickstart", slug: "quickstart" },
        { label: "MCP Tools", slug: "mcp-tools" },
        { label: "Configuration", slug: "configuration" },
        { label: "Deployment", slug: "deployment" },
      ],
    }),
  ],
});

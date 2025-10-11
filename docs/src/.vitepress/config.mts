import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/pysr/',
}

const nav = [
  { text: 'Home', link: '/' },
  { text: 'Examples', link: '/examples' },
  { text: 'API', link: '/api' },
  { text: 'Papers', link: '/papers' },
  {
    text: 'GitHub',
    link: 'https://github.com/MilesCranmer/PySR'
  },
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/pysr/',
  title: 'PySR',
  description: 'High-Performance Symbolic Regression in Python and Julia',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../dist',
  srcExclude: ['**/_*.md'],
  head: [
    ['link', { rel: 'icon', href: `${baseTemp.base}favicon.png` }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  ignoreDeadLinks: true,
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify(getBaseRepository(baseTemp.base)),
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: 'https://ai.damtp.cam.ac.uk/symbolicregression/dev/logo.png',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/' },
          { text: 'Examples', link: '/examples' },
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'API Reference', link: '/api' },
          { text: 'Operators', link: '/operators' },
          { text: 'Options', link: '/options' },
          { text: 'Advanced API', link: '/api-advanced' },
        ]
      },
      {
        text: 'Advanced',
        items: [
          { text: 'Tuning', link: '/tuning' },
          { text: 'Backend', link: '/backend' },
        ]
      },
      {
        text: 'Community',
        items: [
          { text: 'Papers', link: '/papers' },
        ]
      }
    ],
    editLink: {
      pattern: 'https://github.com/MilesCranmer/PySR/edit/master/docs/:path',
      text: 'Edit this page on GitHub'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/MilesCranmer/PySR' }
    ],
    footer: {
      message: 'Made with <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
  }
})

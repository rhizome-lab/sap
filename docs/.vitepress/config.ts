import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    title: 'Sap',
    description: 'Minimal expression language with multiple backends',

    base: '/sap/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Backends', link: '/backends' },
        { text: 'Rhizome', link: 'https://rhizome-lab.github.io/' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Introduction', link: '/introduction' },
              { text: 'Getting Started', link: '/getting-started' },
            ]
          },
          {
            text: 'Backends',
            items: [
              { text: 'Overview', link: '/backends' },
              { text: 'WGSL', link: '/backends/wgsl' },
              { text: 'Cranelift', link: '/backends/cranelift' },
              { text: 'Lua', link: '/backends/lua' },
            ]
          },
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhizome-lab/sap' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhizome-lab/sap/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)

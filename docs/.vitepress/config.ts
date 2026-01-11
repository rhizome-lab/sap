import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import dewGrammar from '../../editors/textmate/dew.tmLanguage.json'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    markdown: {
      languages: [dewGrammar as any],
    },
    title: 'Dew',
    description: 'Minimal expression language with multiple backends',

    base: '/dew/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Backends', link: '/backends/wgsl' },
        { text: 'Playground', link: '/dew/playground/' },
        { text: 'Rhizome', link: 'https://rhizome-lab.github.io/' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Introduction', link: '/introduction' },
            ]
          },
          {
            text: 'Crates',
            items: [
              { text: 'dew-core', link: '/core' },
              { text: 'dew-scalar', link: '/scalar' },
              { text: 'dew-linalg', link: '/linalg' },
              { text: 'dew-complex', link: '/complex' },
              { text: 'dew-quaternion', link: '/quaternion' },
            ]
          },
          {
            text: 'Backends',
            items: [
              { text: 'WGSL', link: '/backends/wgsl' },
              { text: 'Lua', link: '/backends/lua' },
              { text: 'Cranelift', link: '/backends/cranelift' },
            ]
          },
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhizome-lab/dew' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhizome-lab/dew/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)

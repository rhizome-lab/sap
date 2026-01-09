import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    title: 'Dew',
    description: 'Minimal expression language with multiple backends',

    base: '/dew/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Backends', link: '/backends/wgsl' },
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
              { text: 'dew-scalar', link: '/scalar' },
              { text: 'dew-linalg', link: '/linalg' },
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

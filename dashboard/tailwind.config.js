/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Space Grotesk', 'ui-sans-serif', 'system-ui'],
        body: ['Manrope', 'ui-sans-serif', 'system-ui'],
      },
      colors: {
        ink: '#07110f',
        panel: '#0f1b18',
        panel2: '#14231f',
        mint: '#6ee7b7',
        amber: '#f8c471',
        danger: '#fb7185',
        line: '#20352f',
      },
      boxShadow: {
        glow: '0 0 40px rgba(110, 231, 183, 0.12)',
      },
    },
  },
  plugins: [],
}

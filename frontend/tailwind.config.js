module.exports = {
  content: ["./index.html",
  // If your HTML files are in the root directory, include them
  "./src/**/*.{js,jsx,ts,tsx}" // Include all JSX, TSX, JS, and TS files in the src directory
  ],
  theme: {
    extend: {}
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ['light', 'dark'],
  }
};
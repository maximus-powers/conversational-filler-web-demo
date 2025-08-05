/**
 * Legacy ESLint configuration for Next.js compatibility
 * This is needed because Next.js still uses the old ESLint config format
 */
module.exports = {
  extends: [
    "eslint:recommended",
    "@typescript-eslint/recommended",
    "prettier"
  ],
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  rules: {
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }]
  },
  env: {
    node: true,
    es6: true
  },
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module"
  }
}
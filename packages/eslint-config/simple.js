/**
 * Simple ESLint configuration for library packages
 * Uses only built-in ESLint rules to avoid dependency issues
 */
export default [
  {
    files: ["**/*.{js,mjs,cjs,ts,tsx}"],
    rules: {
      "no-unused-vars": "warn",
      "no-console": "warn",
      "prefer-const": "error"
    },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module"
    }
  },
  {
    ignores: ["dist/**", "build/**", "node_modules/**", "*.config.*"]
  }
]
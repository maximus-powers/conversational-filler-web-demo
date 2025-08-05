/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["@workspace/ui"],
  serverExternalPackages: ['sharp', 'onnxruntime-node'],
  webpack: (config, { isServer }) => {
    // transformers.js browser/node compatibility
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-node$": "onnxruntime-web",
      };
      
      // ignore node-specific modules in browser
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    
    return config;
  },
}

export default nextConfig

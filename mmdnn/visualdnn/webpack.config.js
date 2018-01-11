var path = require("path");

module.exports = {
  entry: "./src/scripts/index.js",
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "dist")
  },
  // devServer: {
  //   hot: true, // Tell the dev-server we're using HMR
  //   contentBase: path.resolve(__dirname, 'dist'),
  //   publicPath: '/'
  // },
  resolve: {
    extensions: [".js", ".jsx", ".json", ".css"]
  },
  module: {
    rules: [
      {
        test: /\.js|jsx$/,
        exclude: /node_modules/,
        use: ["babel-loader"]
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"]
      },
       {
        test: /\.json$/,
        use: 'json-loader'
      }
    ]
  },
  devtool: "cheap-eval-source-map",
  target: 'electron-renderer'
};

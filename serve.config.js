module.exports = {
  apps: [
    {
      name: "generation",
      script: "serve.py",
      interpreter: "/venv/wala/bin/python",
      args: "--port 8095",
    },
  ],
};

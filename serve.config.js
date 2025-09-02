module.exports = {
  apps: [
    {
      name: "generation",
      script: "serve.py",
      interpreter: "/venv/wala/bin/python",
      args: '--model_name ADSKAILab/WaLa-MVDream-RGB4 --text_to_mv "generate me a cup" --output_dir examples',
    },
  ],
};

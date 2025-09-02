module.exports = {
  apps: [
    {
      name: "generation",
      script: "serve.py",
      interpreter: "${CONDA_INTERPRETER_PATH}",
      args: '--model_name ADSKAILab/WaLa-MVDream-RGB4 --text_to_mv "generate me a cup" --output_dir examples',
    },
  ],
};

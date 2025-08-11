# Convert SafeTensors to ONNX

This package is used for converting safetensors models to ONNX format using the optimum library (required for us to run it in the browser client with Transformers.js).

## Usage

```
pip install -r requirements.txt
python convert_smollm_official.py
```

Note: you need to have huggingface authenticated where you're running this (for the upload to work)

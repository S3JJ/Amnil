import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

MODEL_PATH = "models/final_model_onnx/model.onnx"   # onnx file path
TOKENIZER_PATH = "models/final_model_onnx"               # tokenizer folder


class NepaliSentimentModelONNX:
    def __init__(self):
        print("Loading ONNX model...")

        # Load tokenizer normally
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        # Load ONNX model
        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        # Read input and output node names
        self.input_name = self.session.get_inputs()[0].name
        self.attn_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

        # Label mapping
        self.id2label = {
            0: "negative",
            1: "positive",
            2: "neutral"
        }

    def predict(self, text: str):

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Build ONNX input dictionary
        onnx_inputs = {
            self.input_name: inputs["input_ids"],
            self.attn_name: inputs["attention_mask"],
        }

        # If the ONNX model expects token_type_ids
        if "token_type_ids" in inputs:
            token_type_name = self.session.get_inputs()[2].name
            onnx_inputs[token_type_name] = inputs["token_type_ids"]

        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]

        # Softmax
        probs = softmax(logits)
        pred_id = np.argmax(probs, axis=1)[0]
        confidence = float(np.max(probs))

        return {
            "label_id": int(pred_id),
            "label": self.id2label[int(pred_id)],
            "confidence": confidence,
            "logits": logits.tolist()
        }



def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)


# Load model globally once
sentiment_model = NepaliSentimentModelONNX()

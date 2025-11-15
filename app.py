import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline("text-classification",
                      model="unitary/toxic-bert",
                      top_k=None)


def moderate(text):
    if not text or len(text.strip()) == 0:
        return "Error: Text cannot be empty"

    if len(text) > 5000:
        return "Error: Text too long (max 5000 characters)"

    result = classifier(text)[0]
    top_prediction = max(result, key=lambda x: x['score'])

    output = f"**Prediction:** {top_prediction['label']}\n"
    output += f"**Confidence:** {top_prediction['score']:.2%}\n\n"
    output += "**All Scores:**\n"
    for pred in sorted(result, key=lambda x: x['score'], reverse=True):
        output += f"- {pred['label']}: {pred['score']:.2%}\n"

    return output


# Create interface
demo = gr.Interface(
    fn=moderate,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter text to moderate...",
        label="Input Text"
    ),
    outputs=gr.Markdown(label="Moderation Results"),
    title="🛡️ Content Moderation API",
    description="Detects toxic and harmful content using a fine-tuned BERT model.",
    examples=[
        ["This is a wonderful day!"],
        ["I hate everything about this"],
        ["You are amazing and I appreciate your help"]
    ]
)

if __name__ == "__main__":
    demo.launch()
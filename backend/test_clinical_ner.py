from openmed import OnnxModel

MODEL_NAME = (
    "OpenMed/"
    "OpenMed-ClinicalNER-SuperClinical-Large-434M-v1-onnx-android"
)

def main():
    print("Loading OpenMed ClinicalNER...")
    model = OnnxModel.from_pretrained(MODEL_NAME)

    text = (
        "Patient has hypertension and shortness of breath. "
        "The patient takes metformin. Blood pressure is 140/90 mmHg."
    )

    print("Running clinical entity extraction...")
    entities = model(text)

    for entity in entities:
        print(entity.to_dict())

    print("Model inference completed.")


if __name__ == "__main__":
    main()
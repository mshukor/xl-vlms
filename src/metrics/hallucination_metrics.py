import json
from typing import Any, Callable, Dict
from sklearn.metrics import precision_recall_fscore_support

import torch


__all__ = ["compute_hallucination_metrics"]


def unwrap_data(a):
    return [b[0] for b in a]

def find_assistant_caption(a, model):
    if "molmo" in model:
        return [b.split("Assistant:")[1] for b in a]
        
    elif "llava" in model:
        c = []
        for b in a:
            if len(b.split("ASSISTANT:")) > 1:
                c.append(b.split("ASSISTANT:")[1])
            else:
                c.append(b)

        return c

    else:
        c = []
        for b in a:
            if len(b.split("assistant\n")) > 1:
                c.append(b.split("assistant\n")[1])
            else:
                c.append(b)

        return c

def response_in_caption_start(responses, captions):
    return [
        response.lower() in caption[:15].lower()
        for response, caption in zip(responses, captions)
    ]


def compute_hallucination_metrics(
    data: Dict[str, Any],
    save_filename: str = None,
    save_predictions: bool = False,
    predictions_path: str = None,
    logger: Callable = None,
    model_name: str = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    
    if predictions_path is not None:
        data = torch.load(predictions_path, map_location="cpu")

    results = {}

    responses = unwrap_data(data["response"])
    captions = find_assistant_caption(unwrap_data(data["model_predictions"]), model_name)
    print(responses[:30])
    print(captions[:30])
    print(len(data["response"]))
    print(len(data["model_predictions"]))
    matches = response_in_caption_start(responses, captions)
    accuracy = sum(matches) / len(matches)
    y_true = [True] * len(responses)

    _, _, f1, _ = precision_recall_fscore_support(
        y_true, matches, average='binary', zero_division=0
    )

    
    results["accuracy"] = accuracy
    results["f1"] = f1


    if logger is not None:
        examples = data["model_predictions"][:10]
        logger.info(f"Captioning prediction examples: {examples}")
        logger.info(
            f"accuracy: {results['accuracy']}"
        )

        logger.info(
            f"f1: {results['f1']}"
        )

    if save_filename:
        with open(save_filename, "w") as json_file:
            json.dump(results, json_file, indent=4)
        if logger is not None:
            logger.info(f"Saving data to: {save_filename}")
        if save_predictions:
            to_save = {"predictions": data["model_predictions"]}

            save_filename = save_filename.split(".json")[0] + "_model_prediction.json"
            with open(save_filename, "w") as json_file:
                json.dump(to_save, json_file, indent=4)
            if logger is not None:
                logger.info(
                    f"Saving {len(data["model_predictions"])} predictions to: {save_filename}"
                )

    return results



















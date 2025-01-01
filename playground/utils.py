# Functions used in the notebooks for simple evaluation and visualization
from typing import Tuple, Dict, Any, List
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

COCO_COLORS_WORDS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "cyan",
    "magenta",
    "turquoise",
    "indigo",
    "maroon",
    "beige",
    "gold",
    "silver",
    "olive",
    "lavender",
    "navy",
    "teal",
    "peach",
    "violet",
    "ivory",
    "charcoal",
    "amber",
    "emerald",
    "coral",
]


COCO_PLACES_WORDS = [
    "beach",
    "mountain",
    "forest",
    "desert",
    "city",
    "village",
    "river",
    "ocean",
    "park",
    "island",
    "countryside",
    "jungle",
    "cave",
    "waterfall",
    "lake",
    "garden",
    "market",
    "museum",
    "restaurant",
    "airport",
    "street",
    "cinema",
    "theater",
    "school",
    "library",
    "stadium",
    "bridge",
    "station",
    "bus stop",
    "hotel",
    "zoo",
    "church",
    "temple",
    "mall",
    "hospital",
    "playground",
    "harbor",
    "factory",
    "tower",
    "university",
    "kitchen",
    "field",
    "room",
    "hill",
    "park",
    "store",
    "cabinet",
    "restaurant",
    "school",
    "airport",
    "zoo",
    "shop",
    "place",
]

COCO_SENTIMENTS_WORDS = set(
    [
        "emotion",
        "smile",
        "happy",
        "sad",
        "facial expression",
        "laugh",
        "tears",
        "frown",
        "grin",
        "joy",
        "crying",
        "anger",
        "surprised",
        "content",
        "excited",
        "worried",
        "disappointed",
        "cheerful",
        "upset",
        "blushing",
        "grief",
        "serious",
        "delighted",
        "anxious",
        "fear",
        "confusion",
        "disgust",
        "smirk",
        "love",
        "hate",
        "relaxed",
        "nervous",
        "bored",
        "concentration",
        "amused",
        "grumpy",
        "pensive",
        "thoughtful",
        "frustrated",
        "ecstatic",
        "mournful",
        "proud",
        "relief",
        "resentful",
        "contentment",
        "admiration",
        "determined",
        "gloomy",
        "bashful",
        "intense",
        "awkward",
        "reluctant",
        "insecure",
        "curious",
        "playful",
        "skeptical",
        "sympathetic",
        "bewildered",
        "elated",
        "optimistic",
        "disheartened",
        "triumphant",
        "indifferent",
        "jealous",
        "melancholy",
        "pleased",
        "ashamed",
        "grateful",
        "loneliness",
        "lonely",
        "ambition",
        "ambitious",
        "despair",
        "despairing",
        "hope",
        "hopeful",
        "solitude",
        "solitary",
        "desire",
        "desirous",
        "freedom",
        "free",
        "conflict",
        "conflicted",
        "tranquility",
        "tranquil",
        "chaotic",
        "powerful",
        "weakness",
        "trust",
        "trustworthy",
        "betrayal",
        "betrayed",
        "mystery",
        "mysterious",
        "identity",
        "identifiable",
        "memorable",
        "imagination",
        "imaginative",
        "dreamlike",
        "nightmare",
        "nightmarish",
        "inspiration",
        "inspirational",
        "faith",
        "faithful",
        "doubt",
        "doubtful",
        "regret",
        "regretful",
        "perseverance",
        "persevering",
        "illusion",
        "illusory",
        "destiny",
        "destined",
        "karma",
        "karmic",
        "purposeful",
        "fated",
        "realistic",
        "transformation",
        "transformative",
        "fearlessness",
        "fearless",
        "emptiness",
        "serenity",
        "serene",
        "wisdom",
        "wise",
        "vulnerability",
        "vulnerable",
        "alienation",
        "alienated",
        "acceptance",
        "accepting",
        "yearning",
        "yearning",
        "ambivalence",
        "ambivalent",
        "hopefulness",
        "hopeful",
        "liberation",
        "liberated",
        "introspection",
        "introspective",
    ]
)


COCO_POSITIVE_SENTIMENTS_WORDS = set(
    [
        "smile",
        "happy",
        "laugh",
        "grin",
        "joy",
        "excited",
        "cheerful",
        "delighted",
        "content",
        "love",
        "relaxed",
        "amused",
        "ecstatic",
        "proud",
        "relief",
        "contentment",
        "admiration",
        "playful",
        "elated",
        "optimistic",
        "triumphant",
        "pleased",
        "grateful",
        "hope",
        "hopeful",
        "ambition",
        "ambitious",
        "freedom",
        "free",
        "trust",
        "trustworthy",
        "inspiration",
        "inspirational",
        "faith",
        "faithful",
        "perseverance",
        "persevering",
        "serenity",
        "serene",
        "wisdom",
        "wise",
        "acceptance",
        "accepting",
        "hopefulness",
        "liberation",
        "liberated",
        "tranquility",
        "tranquil",
        "purposeful",
        "fearlessness",
        "fearless",
        "imagination",
        "imaginative",
    ]
)

COCO_NEGATIVE_SENTIMENTS_WORDS = set(
    [
        "sad",
        "tears",
        "frown",
        "crying",
        "anger",
        "worried",
        "disappointed",
        "upset",
        "grief",
        "serious",
        "anxious",
        "fear",
        "confusion",
        "disgust",
        "grumpy",
        "frustrated",
        "mournful",
        "resentful",
        "gloomy",
        "insecure",
        "awkward",
        "reluctant",
        "disheartened",
        "jealous",
        "melancholy",
        "ashamed",
        "loneliness",
        "lonely",
        "despair",
        "despairing",
        "hate",
        "bored",
        "nightmare",
        "nightmarish",
        "doubt",
        "doubtful",
        "regret",
        "regretful",
        "alienation",
        "alienated",
        "emptiness",
        "vulnerability",
        "vulnerable",
        "betrayal",
        "betrayed",
        "weakness",
        "chaotic",
        "conflict",
        "conflicted",
        "yearning",
        "ambivalence",
        "ambivalent",
    ]
)


WORDS = {
    "colors": COCO_COLORS_WORDS,
    "places": COCO_PLACES_WORDS,
    "sentiments": COCO_SENTIMENTS_WORDS,
    "positive_sentiments": COCO_POSITIVE_SENTIMENTS_WORDS,
    "negative_sentiments": COCO_NEGATIVE_SENTIMENTS_WORDS,
}


import os
import json
from typing import Any
from torch.utils.data import Dataset
from PIL import Image


class MinimalCOCODataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        data_dir: str,
        split: str = "train",
        dataset_size: int = -1,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.split = split
        self.dataset_size = dataset_size
        self.rng = np.random.default_rng(seed)  # Initialize random generator with seed
        self.data = self.load_data()

    def load_data(self):
        """Load the dataset from the annotation file."""
        annotation_path = os.path.join(self.data_dir, self.annotation_file)
        with open(annotation_path, "r") as f:
            karpathy_data = json.load(f)

        data = []
        for datum in karpathy_data["images"]:
            if datum["split"] != self.split:
                continue

            img_id = datum["filename"].split(".")[0]

            if "train" in img_id:
                source = "train2014"
            elif "val" in img_id:
                source = "val2014"
            else:
                raise NotImplementedError(f"Unknown image directory for {img_id}")

            image_path = os.path.join(self.data_dir, source, datum["filename"])
            data.append({"img_id": img_id, "image_path": image_path})

        # Optionally sample a subset of the dataset
        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False).tolist()

        return data

    def __getitem__(self, idx: int) -> Any:
        """Retrieve an image by index."""
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")  # Convert to RGB
        return {"img_id": item["img_id"], "image": image}

    def __len__(self) -> int:
        return len(self.data)




def get_dict_of_top_k_items(
    input_dict: Dict[str, Any], topk: int, reference_dict: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Extract the top k key/value pairs from a dictionary based on the values.

    Args:
    - input_dict: The input dictionary where keys are the labels and values are the magnitudes.
    - k: The number of top items to extract.
    - reference_dict: if passed, compute the relative values to this dictionary.

    Returns:
    - A list of tuples containing the top k key/value pairs sorted by value.
    """
    if reference_dict:
        diff_dict = {k: input_dict[k] - reference_dict.get(k, 0) for k in input_dict}
    else:
        diff_dict = input_dict
    # Sort the dictionary by value in descending order and get the top k items
    top_k_items = sorted(diff_dict.items(), key=lambda item: item[1], reverse=True)[
        :topk
    ]
    topk_dict = {}
    for item in top_k_items:
        topk_dict[item[0]] = item[1]
    return topk_dict


def get_shift_vector_scores(
    results: Dict[str, Any],
    topk: int = 5,
    score_key: str = "",
    keep_first_word: bool = False,
    reference_dict: Dict[str, Any] = {},
) -> Tuple[List, int]:

    if score_key:
        answer_counts = results["answer_counts"][score_key]
        ref_dict = reference_dict.get("answer_counts", {}).get(score_key, {})
    else:
        answer_counts = results["answer_counts"]
        ref_dict = reference_dict.get("answer_counts", {})
    
    ref_dict_ = {}
    answer_counts = {k: v for k, v in answer_counts.items() if k}
    if keep_first_word:
        answer_counts_ = {}
        ref_dict_ = {}
        for k, v in answer_counts.items():
            k_ = k.split(" ")[0]
            if k_ in answer_counts_:
                answer_counts_[k_] += v
            else:
                answer_counts_[k_] = v

            if k in ref_dict:
                if k_ in ref_dict_:
                    ref_dict_[k_] += ref_dict[k]
                else:
                    ref_dict_[k_] = ref_dict[k]

        answer_counts = answer_counts_
    top_k_items = get_dict_of_top_k_items(answer_counts, topk, reference_dict=ref_dict_)
    if len(top_k_items) > 1:
        counts = np.array(list(top_k_items.values()))[:, None]
        model = KMeans(n_clusters=2, max_iter=50).fit(counts)

        predicted_clusters = [
            (k, model.predict(np.array([v])[:, None])) for k, v in top_k_items.items()
        ]
        main_cluster_idx = model.predict(
            np.array([max([v for v in top_k_items.values()])])[:, None]
        )

        main_answers = [k for k in predicted_clusters if k[1] == main_cluster_idx]
        main_answers_keys = [k[0] for k in main_answers]
        main_answers_values = [top_k_items[k[0]] for k in main_answers]

        counts_difference_to_main_answers = [
            min(main_answers_values) - top_k_items[k[0]]
            for k in predicted_clusters
            if k[1] != main_cluster_idx
        ]
        counts_difference_to_main_answers = (
            min(counts_difference_to_main_answers)
            if counts_difference_to_main_answers
            else 0
        )
    else:
        main_answers_keys = list(top_k_items.keys())[0]
        counts_difference_to_main_answers = top_k_items[main_answers_keys]

    return main_answers_keys, counts_difference_to_main_answers


def get_topk_shift_vectors(
    results_paths: List[str],
    answer_types: List[str],
    number_target_concepts: int = 1,
    score_keys: Dict[str, str] = {},
    topk: int = 5,
    reference_dict_path: str = "",
    num_shift_vectors: int = 5,
    unique_vectors: bool = True,
    keep_first_word: bool = False,
) -> Dict[str, Any]:

    reference_dict = json.load(open(reference_dict_path)) if reference_dict_path else {}
    all_scores = {}
    for i, results_path in enumerate(results_paths):
        if os.path.exists(results_path):
            results = json.load(open(results_path))
            if answer_types:
                score_key = score_keys.get(answer_types[i], "other")
                ans_type = answer_types[i]
            else:
                score_key = None
                ans_type = ""
            main_answers, counts_difference_to_main_answers = get_shift_vector_scores(
                results,
                topk=topk,
                score_key=score_key,
                reference_dict=reference_dict,
                keep_first_word=keep_first_word,
            )
            if len(main_answers) <= number_target_concepts and len(main_answers) > 0:
                all_scores[results_path] = {
                    "main_answers": main_answers,
                    "counts_difference_to_main_answers": counts_difference_to_main_answers,
                    "answer_type": ans_type,
                }
    if unique_vectors:
        sorted_dict = dict(
            sorted(
                all_scores.items(),
                key=lambda item: item[1]["counts_difference_to_main_answers"],
                reverse=True,
            )
        )
        shift_targets = []
        all_scores_filtered = {}
        for k, v in sorted_dict.items():
            main_answers = v["main_answers"]
            if main_answers not in shift_targets:
                all_scores_filtered[k] = v
                shift_targets.append(main_answers)
            if len(all_scores_filtered) >= num_shift_vectors:
                break
        sorted_dict = all_scores_filtered
    else:
        sorted_dict = dict(
            sorted(
                all_scores.items(),
                key=lambda item: item[1]["counts_difference_to_main_answers"],
                reverse=True,
            )[:num_shift_vectors]
        )

    return sorted_dict


def csv_print(data, keys=[]):
    dict_ = {}
    for key in keys:
        dict_[key] = [data[key]]
    df = pd.DataFrame(dict_)
    df.index = [''] * len(df)
    display(df)

def visualize_bars(data_list: list, keys_list: list, titles=[], y_min=None, y_max=None, save_path='', figsize=[5, 6]):
    """
    Visualize magnitudes from a list of dictionaries as horizontal bars in multiple subplots.

    Args:
    - data_list: List of dictionaries where each dictionary contains keys and their corresponding values.
    - keys_list: List of lists where each sublist contains keys corresponding to the respective dictionary.
    """

    # Number of subplots
    num_subplots = len(data_list)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_subplots, figsize=(figsize[0] * num_subplots, figsize[1]))


    # Ensure axes is iterable even if there's only one subplot
    if num_subplots == 1:
        axes = [axes]

    for i, (ax, data, keys) in enumerate(zip(axes, data_list, keys_list)):
        # Extract magnitudes based on the keys provided
        keys_ = [key for key in keys if key in data]
        magnitudes = [data[key] for key in keys_]

        # Plot the horizontal bar chart
        bars = ax.bar(keys_, magnitudes, color=['green' if val > 0 else 'red' for val in magnitudes], alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center')
                        
        ax.set_xticklabels(keys, rotation=30, ha='right')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        if i < len(titles):
            ax.set_title(titles[i])

    # Adjust layout to ensure everything fits
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+'.pdf', format="pdf", bbox_inches="tight")
        print(f'Saved to: {save_path}')
    plt.show()

def plot_shift_results(results_paths=[], results_paths_dict={}, reference_dict_path='', score_keys=[], answer_types=[], print_keys=[], 
                       keep_first_word=False, keys_answer_type=[], save_dir='', save_name=''):
    if reference_dict_path:
        reference_dict = json.load(open(reference_dict_path)) 
    else:
        reference_dict = {}
    
    save_path = ''
    if save_dir:
        save_name_ = reference_dict_path.split('/')[-1].split('.')[0]
        if save_name:
            save_name_ = f"{save_name_}_{save_name}"
        save_path = os.path.join(save_dir, f'steering_vector_{save_name_}')
    print(f'\nBaseline:')
    datas, keys, titles = get_answer_types_and_count_dicts(reference_dict, topk=5, print_keys=print_keys, keys_answer_type=keys_answer_type)                        
    visualize_bars(datas, keys, titles, save_path=save_path)



    if not answer_types:
        if results_paths_dict:
            results_paths = []
            answer_types = []
            for k, v in results_paths_dict.items():
                results_paths.append(k)
                answer_types.append(v['answer_type'])
        else:
            answer_types = ['other']*len(results_paths)
    for k, results_path in enumerate(results_paths):
        answer_type = answer_types[k]
        print_keys_ = print_keys
        print_keys_ = print_keys + [score_keys[answer_type]]

        if os.path.exists(results_path):
            results = json.load(open(results_path))
            print(f'\n{answer_type} {results_path}:')

            main_answers, counts_difference_to_main = get_shift_vector_scores(results, topk=5, score_key=score_keys[answer_type], reference_dict=reference_dict, keep_first_word=keep_first_word)
            print(f'main_answers: {main_answers} counts_difference_to_main: {counts_difference_to_main}:')
            main_answers, counts_difference_to_main = get_shift_vector_scores(results, topk=5, score_key='overall', reference_dict=reference_dict, keep_first_word=keep_first_word)
            print(f'Overall: main_answers: {main_answers} counts_difference_to_main: {counts_difference_to_main}:')

            datas, keys, titles = get_answer_types_and_count_dicts(results, topk=5, print_keys=print_keys_, reference_dict=reference_dict, 
                                                                   keep_first_word=keep_first_word, keys_answer_type=keys_answer_type) 
            if save_dir:
                save_name_ = results_path.split('/')[-1].split('.')[0]
                if save_name:
                    save_name_ = f"{save_name_}_{save_name}"
                save_path = os.path.join(save_dir, f'steering_vector_{save_name_}')        
            visualize_bars(datas, keys, titles, save_path=save_path)
            
def get_answer_types_and_count_dicts(results, topk=5, print_keys=[], reference_dict={}, keep_first_word=False, keys_answer_type=[]):
    
    if reference_dict:
        answer_types_count = {k: results['answer_types_count'][k] - reference_dict.get('answer_types_count', {}).get(k, 0) for k in results['answer_types_count']}
    else:
        answer_types_count = results['answer_types_count']
        
    answer_types_count = {k: v for k, v in answer_types_count.items() if 'unrecog' not in k}
        
    datas = []
    keys = []
    titles = []
    if 'accuracy' in print_keys:
        if reference_dict:
            acc = {k: results['global_accuracy'][k] - reference_dict.get('global_accuracy', {}).get(k, 0) for k in results['global_accuracy']}
        else:
            acc = results['global_accuracy']
        acc_keys = ['yes/no', 'number', 'other'] #[k for k in acc if 'category' not in k and 'overall' not in k]
        acc = {k: acc[k] for k in acc_keys}
        datas.append(acc)
        keys.append(acc_keys)
        titles.append("Accuracy")
        
    datas.append(answer_types_count)
    keys.append(keys_answer_type)
    titles.append("Answers type")
    for key in print_keys:
        if key in results['answer_counts']:
            counts = results['answer_counts'][key]
            ref_dict = reference_dict.get('answer_counts', {}).get(key, {})
                
            if keep_first_word:
                counts, ref_dict = keep_first_word_keys(counts, ref_dict)
        
            top_k_items = get_dict_of_top_k_items(counts, topk, reference_dict=ref_dict)
            datas.append(top_k_items)
            keys.append(list(top_k_items.keys()))
            titles.append(f"Answers ({key})")
    return datas, keys, titles



def get_plot_coco_results(results, topk=3, plot_results={}, target_answers=[], 
                          label='', dataset_size=3000, words_dict=None):

    cider_scores = [results['scores']['CIDEr']*100.]
    cider_titles = ['',]
    
    words_type_count_titles = []
    words_type_count_scores = []
    if words_dict is not None:
        words_type_count = get_words_type_count(results, words_dict=words_dict)
        words_type_count_titles = list(words_type_count.keys())
        words_type_count_scores = [words_type_count[k] for k in words_type_count_titles]
    
    answer_count_scores = []
    answer_count_titles = []
    answer_count_scores.extend([get_values_for_matched_keys(results['answer_counts'], t) for t in target_answers])
    answer_count_titles.extend([f'{t}' for t in target_answers])
        
    
    dataset_size = results.get('dataset_size', dataset_size)
    valid_steered_captions_scores = [100.*(results['num_preds_with_toi'] / dataset_size), 
                               100.*(results['num_preds_and_targets_with_toi'] / dataset_size), 
                               100.*(results['num_preds_and_baseline_preds_with_toi'] / dataset_size)]
    valid_steered_captions_titles = [f'--/generated', 'gt/generated', 'original/generated']
    
    steered_captions_scores = [100.*(results['num_preds_changed'] / dataset_size)]
    steered_captions_titles = ['',]
    
        
    if not plot_results:
        plot_results['CIDEr'] = {
            'scores': [cider_scores],
            'titles': [cider_titles],
            'x_labels': [[label]],
        }
        plot_results['Captions affected (%)'] = {
            'scores': [steered_captions_scores],
            'titles': [steered_captions_titles],
            'x_labels': [[label]],
        }
        plot_results[f'Captions with {target_answers[0]}/{target_answers[1]} in (%)'] = {
            'scores': [valid_steered_captions_scores],
            'titles': [valid_steered_captions_titles],
            'x_labels': [[label]],
        }
        plot_results['Words count'] = {
            'scores': [answer_count_scores],
            'titles': [answer_count_titles],
            'x_labels': [[label]],
        }
        if words_dict is not None:
            plot_results['Words type count'] = {
                'scores': [words_type_count_scores],
                'titles': [words_type_count_titles],
                'x_labels': [[label]],
            }
    else:
        plot_results['CIDEr']['scores'].append(cider_scores) 
        plot_results['CIDEr']['titles'].append(cider_titles) 
        plot_results['CIDEr']['x_labels'].append([label]) 
        
        plot_results['Captions affected (%)']['scores'].append(steered_captions_scores) 
        plot_results['Captions affected (%)']['titles'].append(steered_captions_titles) 
        plot_results['Captions affected (%)']['x_labels'].append([label]) 

        plot_results[f'Captions with {target_answers[0]}/{target_answers[1]} in (%)']['scores'].append(valid_steered_captions_scores) 
        plot_results[f'Captions with {target_answers[0]}/{target_answers[1]} in (%)']['titles'].append(valid_steered_captions_titles) 
        plot_results[f'Captions with {target_answers[0]}/{target_answers[1]} in (%)']['x_labels'].append([label]) 
        
        plot_results['Words count']['scores'].append(answer_count_scores) 
        plot_results['Words count']['titles'].append(answer_count_titles) 
        plot_results['Words count']['x_labels'].append([label]) 
        
        if words_dict is not None:
            plot_results['Words type count']['scores'].append(words_type_count_scores) 
            plot_results['Words type count']['titles'].append(words_type_count_titles) 
            plot_results['Words type count']['x_labels'].append([label]) 
        
    return plot_results



def get_words_type_count(data, words_dict=None, add_other=False):
    words_type_count = {k: 0 for k in words_dict.keys()}
    other = 0
    all_words_list = []
    for v in words_dict.values():
        all_words_list.extend(v)
    for ans_key, ans_count in tqdm(data['answer_counts'].items()):
        valid = False
        for words_type, words_list in words_dict.items():
            if ans_key.lower().strip() in words_list:
                words_type_count[words_type] += ans_count
                valid = True
                break
        if not valid and ans_key.lower().strip() not in all_words_list:
            other+=ans_count
    if add_other:
        words_type_count['other'] = other
    return words_type_count


def plot_shift_results_coco(results_paths=[], results_paths_dict={}, reference_dict_path='', score_keys=[], answer_types=[], print_keys=[], 
                       keep_first_word=False, save_dir='', save_name='', words_dict={}, words_type_keys=[''], topk=5, 
                            figsize=[5, 6], save_names=[]):
                            
    if reference_dict_path:
        reference_dict = json.load(open(reference_dict_path)) 
    else:
        reference_dict = {}
    
    save_path = ''
    if save_dir:
        save_name_ = reference_dict_path.split('/')[-1].split('.')[0]
        if save_name:
            save_name_ = f"{save_name_}_{save_name}"
        save_path = os.path.join(save_dir, f'{save_name_}')
    print(f'\nBaseline:')
    
    datas, keys, titles = get_count_dicts_coco(reference_dict, topk=topk, words_dict=words_dict, print_keys=print_keys, 
                                               words_type_keys=words_type_keys, keep_first_word=keep_first_word)                        
    visualize_bars(datas, keys, titles, save_path=save_path, figsize=figsize)
            
        
    if not results_paths:
        if results_paths_dict:
            results_paths = []
            for k, v in results_paths_dict.items():
                results_paths.append(k)
            
    for k, results_path in enumerate(results_paths):
        print(os.path.exists(results_path))
        if os.path.exists(results_path):
            results = json.load(open(results_path))
            print(f'\n{results_path}:')

            main_answers, counts_difference_to_main = get_shift_vector_scores(results, topk=topk, reference_dict=reference_dict, 
                                                                              keep_first_word=keep_first_word)
            print(f'main_answers: {main_answers} counts_difference_to_main: {counts_difference_to_main}:')

            datas, keys, titles = get_count_dicts_coco(results, topk=topk, reference_dict=reference_dict, 
                                                                   keep_first_word=keep_first_word, words_dict=words_dict, 
                                                       print_keys=print_keys, words_type_keys=words_type_keys) 
            if save_dir:
                save_name_ = results_path.split('/')[-1].split('.')[0]
                if save_names:
                    save_name_ = save_names[k]
                    save_path = os.path.join(save_dir, f'{save_name_}')   
                elif save_name:
                    save_name_ = f"{save_name_}_{save_name}"
                    save_path = os.path.join(save_dir, f'{save_name_}')   
                else:
                    save_path = os.path.join(save_dir, f'{save_name_}')        
            visualize_bars(datas, keys, titles, save_path=save_path, figsize=figsize)
            
            
def keep_first_word_keys(dict1, dict2):
    dict1_ = {}
    dict2_ = {}
    for k, v in dict1.items():
        k_ = k.split(" ")[0]
        if k_ in dict1_:
            dict1_[k_] += v
        else:
            dict1_[k_] = v

        if k in dict2:
            if k_ in dict2_:
                dict2_[k_] += dict2[k]
            else:
                dict2_[k_] = dict2[k]
                
    return dict1_, dict2_
                    
def get_count_dicts_coco(results, topk=5, reference_dict={}, keep_first_word=False, words_dict={}, print_keys=[''], 
                         words_type_keys=[''], add_other=True):
    
    datas = []
    keys = []
    titles = []

    
    if "type" in print_keys:
        if words_dict:
            words_type_count = get_words_type_count(results, words_dict=words_dict, add_other=add_other)
            if reference_dict:
                ref_words_type_count = get_words_type_count(reference_dict, words_dict=words_dict, add_other=add_other)
                words_type_count = {k: words_type_count[k] - ref_words_type_count[k] for k in ref_words_type_count}
                
            words_type_count = {k: v for k, v in words_type_count.items() if k in words_type_keys or ('other' in k and add_other)}
            words_type_count_titles = list(words_type_count.keys())

            datas.append(words_type_count)
            keys.append(words_type_count_titles)
            titles.append(f"Words type count")
    
    counts = results['answer_counts']
    ref_dict = reference_dict.get('answer_counts', {})

    if keep_first_word:
        counts, ref_dict = keep_first_word_keys(counts, ref_dict)

    top_k_items = get_dict_of_top_k_items(counts, topk, reference_dict=ref_dict)
    datas.append(top_k_items)
    keys.append(list(top_k_items.keys()))
    titles.append(f"Words count")
    
    print(top_k_items)
    return datas, keys, titles


def get_values_for_matched_keys(data, word):
    values = []
    
    word = word.lower().strip()
    for k, v in data.items():
        k_ = k.lower().strip()
        if word in k_:
            values.append(v)
            
    return sum(values)
    
def format_large_number(num):
    # Check if number is above 1000, and if so, format in scientific notation
    if num >= 1000:
        return f"{num:.0e}"  # Format as scientific notation without decimals
    return num  # Otherwise, return the number as a regular string


def get_plot_results(results, topk=3, plot_results={}, target_answers=[], target_print_keys='yes/no', 
                     print_keys='overall', accuracy_keys=['other', 'overall', 'yes/no', 'number'], x_label='', label='', dataset_size=5000):

    label_ = label+': ' if label else label
    
    accuracy_scores = []
    accuracy_titles = []
    for key in accuracy_keys:
        accuracy_scores.append(results['global_accuracy'][key])
        accuracy_titles.append(f'{label_}{key}')
        
    
    answer_types_count_scores = []
    answer_types_count_titles = []
    for k in accuracy_keys:
    # for k, v in results['answer_types_count'].items():
        v =  results['answer_types_count'].get(k, 0)
        answer_types_count_scores.append(v)
        answer_types_count_titles.append(f'{label_}{k}')

        
    answer_count_scores = []
    answer_count_titles = []
    answer_count_scores.extend([get_values_for_matched_keys(results['answer_counts'][print_keys], t) for t in target_answers])
    answer_count_titles.extend([f'{label_}{t}' for t in target_answers])
        
    answer_count_scores_target = []
    answer_count_titles_target = []
    answer_count_scores_target.extend([get_values_for_matched_keys(results['answer_counts'][target_print_keys], t) for t in target_answers])
    answer_count_titles_target.extend([f'{label_}{t}' for t in target_answers])
        
    
    dataset_size = results.get('dataset_size', dataset_size)
    valid_steered_answers_scores = [100.*(results['num_preds_with_toi'] / dataset_size), 
                               100.*(results['num_preds_and_targets_with_toi'] / dataset_size)]
    valid_steered_answers_titles = [f'{label_}--/generated', f'{label_}gt/generated']
    
    steered_answers_scores = [100.*(results['num_preds_changed'] / dataset_size)]
    steered_answers_titles = [f'{label}',]
    
    if not plot_results:
        plot_results['Accuracy'] = {
            'scores': [accuracy_scores],
            'titles': [accuracy_titles],
            'x_labels': [[x_label]],
        }
        plot_results['Answer types count'] = {
            'scores': [answer_types_count_scores],
            'titles': [answer_types_count_titles],
            'x_labels': [[x_label]],
        }
        # plot_results['Answers count'] = {
        #     'scores': [answer_count_scores],
        #     'titles': [answer_count_titles],
        #     'x_labels': [[x_label]],
        # }
        plot_results['Answers count (target)'] = {
            'scores': [answer_count_scores_target],
            'titles': [answer_count_titles_target],
            'x_labels': [[x_label]],
        }
        
        plot_results['Answers affected (%)'] = {
            'scores': [steered_answers_scores],
            'titles': [steered_answers_titles],
            'x_labels': [[x_label]],
        }
        plot_results[f'Answers with {target_answers[0]}/{target_answers[1]} in (%)'] = {
            'scores': [valid_steered_answers_scores],
            'titles': [valid_steered_answers_titles],
            'x_labels': [[x_label]],
        }
        
    else:
        plot_results['Accuracy']['scores'].append(accuracy_scores) 
        plot_results['Accuracy']['titles'].append(accuracy_titles) 
        plot_results['Accuracy']['x_labels'].append([x_label]) 
        
        plot_results['Answer types count']['scores'].append(answer_types_count_scores) 
        plot_results['Answer types count']['titles'].append(answer_types_count_titles) 
        plot_results['Answer types count']['x_labels'].append([x_label]) 
        
        # plot_results['Answers count']['scores'].append(answer_count_scores) 
        # plot_results['Answers count']['titles'].append(answer_count_titles) 
        # plot_results['Answers count']['x_labels'].append([x_label]) 

        plot_results['Answers count (target)']['scores'].append(answer_count_scores_target) 
        plot_results['Answers count (target)']['titles'].append(answer_count_titles_target) 
        plot_results['Answers count (target)']['x_labels'].append([x_label]) 
        
        plot_results['Answers affected (%)']['scores'].append(steered_answers_scores) 
        plot_results['Answers affected (%)']['titles'].append(steered_answers_titles) 
        plot_results['Answers affected (%)']['x_labels'].append([x_label]) 

        plot_results[f'Answers with {target_answers[0]}/{target_answers[1]} in (%)']['scores'].append(valid_steered_answers_scores) 
        plot_results[f'Answers with {target_answers[0]}/{target_answers[1]} in (%)']['titles'].append(valid_steered_answers_titles) 
        plot_results[f'Answers with {target_answers[0]}/{target_answers[1]} in (%)']['x_labels'].append([x_label]) 
        
    return plot_results

    
def plot_curve_results(plot_data, x_axis, save_path='', x_axis_label='Layer', figsize=(7, 6)):
    """
    Plot results in a structured manner with each main key from `plot_data` as a separate subplot row.

    Args:
    - plot_data: Dictionary containing the plot data organized as returned by `get_plot_results`.
                 Each score entry is expected to correspond to a data point on the curve.
    """
    num_categories = len(plot_data.keys())  # Number of main categories (e.g., Accuracy, Answer types count)
    fig, axes = plt.subplots(1, num_categories, figsize=(figsize[0] * num_categories, figsize[1]), sharey=False)

    # Flatten axes if only one row or column
    if num_categories == 1:
        axes = [axes]
    
    # Iterate over each category (Accuracy, Answer types count, etc.)
    for idx, (category, details) in enumerate(plot_data.items()):
        ax = axes[idx]
        scores = details['scores']  # List of lists; each inner list represents a curve
        titles = details['titles']  # List of titles for each curve
        x_labels = details['x_labels']
        # Plot each score curve
        curves = {}
        curves_to_x_labels = {}
        for score, title, x_label in zip(scores, titles, x_labels):
            for s, t in zip(score, title):
                if t in curves:
                    curves[t].append(s)
                    curves_to_x_labels[t].append(x_label[0])
                else:
                    curves[t] = [s]
                    curves_to_x_labels[t] = [x_label[0]]
        for k, v in curves.items():
            ax.plot(curves_to_x_labels[k], v, marker='o', label=k, linestyle='--')  # Plot with markers on each point

        # Set plot details
        ax.set_title(category, )
        ax.set_xlabel(x_axis_label,)
        # ax.set_ylabel("Value", )
        ax.legend(loc='upper left', framealpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout to ensure everything fits
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    
    if save_path:
        plt.savefig(save_path+'.pdf', format="pdf", bbox_inches="tight")
        print(f'Saved to: {save_path}')
        
    plt.show()
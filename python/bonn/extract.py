import os
import logging
import json
from sortedcontainers import SortedDict
from collections import Counter
from elasticsearch2 import Elasticsearch
from elasticsearch_dsl import Search, Q
from nltk import download
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer

from ._bonn import FfModel
from .category_manager import CategoryManager
from .taxonomy import get_taxonomy, taxonomy_to_categories, categories_to_classifier_bow

DEFAULT_TAXONOMY_LOCATION = "/app/test_data/taxonomy.json"


def get_datasets(cm, classifier_bow, settings):
    ltzr = WordNetLemmatizer()
    host = settings.get(
        "ELASTICSEARCH_HOST",
        os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch-master:9200"),
    )
    elasticsearch_index = settings.get(
        "ELASTICSEARCH_INDEX",
        os.getenv("ELASTICSEARCH_INDEX", "ons1639492069322")
    )

    classifier_bow_vec = {
        k: [cm._model[w[1]] for w in words] for k, words in classifier_bow.items()
    }
    datasets = {}
    # results_df = pd.DataFrame((d.to_dict() for d in s.scan()))
    # /businesseconomy../business/activitiespeopel/123745
    client = Elasticsearch([host])

    s = Search(using=client, index=elasticsearch_index).filter(
        "bool", must=[Q("exists", field="description.title")]
    )
    expecting = s.count()
    size = 50
    s = s.params(size=size)
    all_words = Counter()
    with tqdm(total=expecting) as pbar:
        for hit in s.scan():
            try:
                title = hit.description.title
                datasets[title] = {
                    "category": tuple(hit.uri.split("/")[1:4]),
                    "text": f"{title} {hit.description.metaDescription}",
                }
                cat = datasets[title]["category"]
                if cat not in classifier_bow_vec and cat[:-1] in classifier_bow_vec:
                    cat = cat[:-1]
                    datasets[title]["category"] = cat

                datasets[title]["bow"] = cm.closest(
                    datasets[title]["text"], cat, classifier_bow_vec
                )
                document = (
                    title
                    + " "
                    + datasets[title]["text"]
                )
                all_words.update(
                    {
                        ltzr.lemmatize(v)
                        for v in set(sum(cm.strip_document(document), []))
                    }
                )
                datasets[title]["bow"] = cm.closest(
                    document, cat, classifier_bow_vec
                )
            except AttributeError:
                pass
            pbar.update(1)

    all_words = {w: v for w, v in all_words.items() if v > 100}
    return datasets, all_words


def discover_terms(datasets, classifier_bow, settings):
    discovered_terms = {}
    # could do with lemmatizing
    for ds in datasets.values():
        if ds["category"][0:2] not in discovered_terms:
            discovered_terms[ds["category"][0:2]] = Counter()
        discovered_terms[ds["category"][0:2]].update(set(ds["bow"]))
        if ds["category"] not in discovered_terms:
            discovered_terms[ds["category"]] = Counter()
        discovered_terms[ds["category"]].update(set(ds["bow"]))

    appearance_threshold = settings.get("APPEARANCE_THRESHOLD", 5)
    upper_appearance_threshold = settings.get("UPPER_APPEARANCE_THRESHOLD", 10)
    discovered_terms = {
        k: [
            w
            for w, c in count.items()
            if c > (appearance_threshold if len(k) > 2 else upper_appearance_threshold)
        ]
        for k, count in discovered_terms.items()
    }
    for key, terms in classifier_bow.items():
        if key in discovered_terms:
            terms += [("WSSC", w) for w in discovered_terms[key]]
        if key[0:2] in discovered_terms:
            terms += [("WC", w) for w in discovered_terms[key[0:2]]]


def save_to_cache(cache_target, all_words, classifier_bow, settings):
    cached = {
        "config": dict(settings),
        "all-words": all_words,
        "classifier-bow": [
            ["|".join(key), [[c, v] for c, v in terms]]
            for key, terms in classifier_bow.items()
        ],
    }
    try:
        with open(cache_target, "w") as cache_f:
            json.dump(cached, cache_f)
    except OSError:
        logging.warning("Could not write cache to target %s", cache_target)


def append_discovered_terms_from_elasticsearch(cm, classifier_bow, settings):
    datasets, all_words = get_datasets(cm, classifier_bow, settings)
    discover_terms(datasets, classifier_bow, settings)
    cm.set_all_words(all_words)
    return all_words


def load(model_file, settings):
    model = FfModel(model_file)
    # Import and download stopwords from NLTK.
    download("stopwords")  # Download stopwords list.
    download("omw-1.4")  # Download lemma list.
    download("wordnet")  # Download lemma list.

    category_manager = CategoryManager(model, settings)
    taxonomy_location = settings.get(
        "TAXONOMY_LOCATION", DEFAULT_TAXONOMY_LOCATION
    )

    taxonomy = get_taxonomy(taxonomy_location)
    categories = taxonomy_to_categories(taxonomy)

    cache_target = settings.get("CACHE_TARGET", None)
    rebuild_cache = settings.get("REBUILD_CACHE", False)

    if cache_target and os.path.isfile(cache_target) and not rebuild_cache:
        with open(cache_target, "r") as cache_f:
            cached = json.load(cache_f)
        classifier_bow = SortedDict(
            {
                tuple(key.split("|")): tuple((c, v) for c, v in terms)
                for key, terms in cached["classifier-bow"]
            }
        )
        category_manager.set_all_words(cached["all-words"])
    else:
        classifier_bow = categories_to_classifier_bow(
            category_manager.strip_document, categories
        )
        all_words = append_discovered_terms_from_elasticsearch(
            category_manager, classifier_bow, settings
        )
        if cache_target and (not os.path.exists(cache_target) or rebuild_cache):
            save_to_cache(cache_target, all_words, classifier_bow, settings)

    category_manager.add_categories_from_bow("onyxcats", classifier_bow)

    return category_manager

if __name__ == "__main__":
    from .settings import settings
    load(settings["MODEL_FILE"], settings)

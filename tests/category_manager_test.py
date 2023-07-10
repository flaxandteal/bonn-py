import pytest
from nltk import download
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from bonn.extract import CategoryManager, FfModel
from dynaconf import Dynaconf

def get_test_data(datafile):
    return Path(__file__).parent / "data" / datafile

@pytest.fixture(scope='module')
def model():
    model = FfModel(str(get_test_data("wiki.en.fifu")))
    # Import and download stopwords from NLTK.
    download("stopwords")  # Download stopwords list.
    download("omw-1.4")  # Download lemma list.
    download("wordnet")  # Download lemma list.
    return model



def make_category_manager(model, settings):
    category_manager = CategoryManager(model, settings)
    return category_manager

@pytest.fixture
def category_manager(model, settings):
    return make_category_manager(model, settings)

@pytest.fixture
def settings():
    settings = Dynaconf()
    settings.STOPWORDS_LANGUAGE = "english"
    return settings

def test_can_create_category_manager(category_manager):
    pass

def test_can_add_categories(category_manager):
    category_manager.add_categories_from_bow(
        "cities",
        {
            "manchester": (("C", "manchester"),),
            "cardiff": (("C", "cardiff"),),
            "ealing": (("C", "ealing"),),
            "underground": (("C", "underground"),),
        }
    )
    print(category_manager.test("london", "cities"))

import pytest
from nltk import download
from pathlib import Path
from fastapi.testclient import TestClient
from category_api.server import make_app
from unittest.mock import MagicMock
from bonn.extract import CategoryManager, FfModel
from category_api.healthcheck import Healthcheck
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

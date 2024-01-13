import re
from typing import Optional, Callable
from pathlib import Path
from urllib.request import urlopen
from fiona.io import ZipMemoryFile
import numpy as np
import pandas as pd
import geopandas as gpd
import geoviews as gv
from thefuzz import fuzz
import hvplot
import hvplot.pandas
import holoviews as hv
from holoviews import streams
import colorcet as cc
import cartopy.crs as ccrs
import panel as pn
import panel.widgets as pnw
import seaborn as sns
from wordcloud import WordCloud
from PIL import ImageDraw, Image

from translate_app import translate_list_to_dict


def convert_to_snake_case(item):
    """Function to convert a string to snake case"""
    # Add _ before uppercase in camelCase
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", item)
    # Add _ before uppercase following lowercase or digit
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Add _ between letter and digit
    s3 = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s2)
    s4 = re.sub(r"[-\s]", "_", s3).lower()  # Replace hyphen or space with _
    return s4


def sanitize_df_column_names(df):
    """Function to danitize column names by translating and conveting to snake case"""
    column_list = df.columns.tolist()
    # translate the column names
    translated_dict = translate_list_to_dict(column_list)
    # map the translated column names to the column names
    df.rename(columns=translated_dict, inplace=True)
    # convert the column names to snake case
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    return df


def get_gdf_from_zip_url(zip_url: str) -> Optional[dict[str, gpd.GeoDataFrame]]:
    """Function to get the geojson data from the zip url.
    In the zip url, the geojson files are in the data folder."""
    gpd_dict = {}

    with urlopen(zip_url) as u:
        zip_data = u.read()
    with ZipMemoryFile(zip_data) as z:
        geofiles = z.listdir("data")
        for file in geofiles:
            with z.open("data/" + file) as g:
                gpd_dict[Path(file).stem] = gpd.GeoDataFrame.from_features(g, crs=g.crs)
    return gpd_dict if gpd_dict else None


def rename_keys(d, prefix="zurich_gdf_"):
    """Rename the keys of a dictionary with a prefix."""
    return {f"{prefix}{i}": v for i, (k, v) in enumerate(d.items())}


def find_breed_match(
    input_breed: str,
    breeds_df: pd.DataFrame,
    scoring_functions: list[Callable[[str, str], int]],
    scoring_threshold: int = 85,
) -> Optional[str]:
    """
    Find the match for the breed in the FCI breeds dataframe.
    breeds_df dataframe must have both a breed_en and alt_names column.
    """
    # Initialize the maximum score and best match
    max_score = scoring_threshold
    best_match = None

    # Iterate over each row in the breeds dataframe
    for index, breed_row in breeds_df.iterrows():
        # Get the alternative names for the current breed
        alternative_names = breed_row["alt_names"]

        # Calculate the score for the input breed and each alternative name
        # using each scoring function, and take the maximum of these scores
        current_score = max(
            max(
                scoring_function(input_breed, alt_name)
                for scoring_function in scoring_functions
            )
            for alt_name in alternative_names
        )
        # If the current score is greater than the maximum score, update the
        # maximum score and best match
        if current_score > max_score:
            max_score = current_score
            best_match = breed_row["breed_en"]

        # If the maximum score is 100, we have a perfect match and can break
        # out of the loop early
        if max_score == 100:
            break

    # Return the best match
    return best_match


def apply_fuzzy_matching_to_breed_column(
    dataframe: pd.DataFrame,
    breed_column: str,
    fci_df: pd.DataFrame,
    scoring_functions: list[Callable[[str, str], int]],
    scoring_threshold: int = 85,
) -> pd.Series:
    """Apply fuzzy matching to the breed column in the dataframe."""

    return dataframe[breed_column].apply(
        lambda breed: find_breed_match(
            breed, fci_df, scoring_functions, scoring_threshold=scoring_threshold
        )
    )

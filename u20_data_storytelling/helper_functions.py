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
from joblib import Memory

from translate_app import translate_list_to_dict


# Set the cache directory
cache_dir = "./zurich_cache_directory"
memory = Memory(cache_dir, verbose=0)


# Create a player widget
yearly_player = pnw.Player(
    name="Yearly Player",
    start=2015,
    end=2022,
    value=2020,
    step=1,
    loop_policy="loop",
    interval=3000,
)
# Create a slider for the roster
roster_slider = pnw.IntSlider(value=2020, start=2015, end=2022)


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


def get_gdf_from_zip_url(zip_url: str) -> dict[str, gpd.GeoDataFrame]:
    """Function to get the geojson data from the zip url.
    In the zip url, the geojson files are in the data folder."""
    gpd_dict = {}

    try:
        with urlopen(zip_url) as u:
            zip_data = u.read()
        with ZipMemoryFile(zip_data) as z:
            geofiles = z.listdir("data")
            for file in geofiles:
                with z.open("data/" + file) as g:
                    gpd_dict[Path(file).stem] = gpd.GeoDataFrame.from_features(
                        g, crs=g.crs
                    )
    except Exception as e:
        raise Exception(f"Error reading geojson data: {e}")

    return gpd_dict


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


def get_line_plots(data, x, group_by, highlight_list=None, **kwargs):
    """
    Generates an overlaid plot from data, highlighting specified groups with distinct colors.
    """
    if highlight_list is None:
        highlight_list = []
    # Default highlight colors
    default_highlight_colors = [
        "#DC143C",  # Crimson Red
        "#4169E1",  # Royal Blue
        "#50C878",  # Emerald Green
        "#DAA520",  # Goldenrod
    ]

    plots = []
    colors = kwargs.get("colors", ["gray" if not highlight_list else "lightgray"])
    highlight_colors = kwargs.get("highlight_colors", default_highlight_colors)

    # Extend the highlight_colors list if there are more highlighted groups than colors
    if len(highlight_list) > len(highlight_colors):
        highlight_colors = highlight_colors * (
            len(highlight_list) // len(highlight_colors) + 1
        )

    for i, group_value in enumerate(data[group_by].unique()):
        # Filter the DataFrame for the specified value
        filtered_data = data.query(f"{group_by} == @group_value")

        # Determine the color for the plot
        plot_color = (
            highlight_colors[highlight_list.index(group_value)]
            if group_value in highlight_list
            else colors[i % len(colors)]
        )

        # Create a line plot for the specified value
        line_plot = filtered_data.hvplot(color=plot_color, x=x, by=group_by, alpha=0.9)

        # Create a scatter plot for the specified value
        scatter_plot = filtered_data.hvplot.scatter(color=plot_color, x=x, by=group_by)

        # Combine the line plot and scatter plot
        plot = line_plot * scatter_plot
        plots.append(plot)

    # Overlay the plots
    combined_plot = hv.Overlay(plots)

    return combined_plot


def get_line_plots(data, x, group_by, highlight_list=None, **kwargs):
    """
    Generates an overlaid plot from data, highlighting specified groups with distinct colors.
    """
    if highlight_list is None:
        highlight_list = []
    # Default highlight colors
    default_highlight_colors = [
        "#DC143C",  # Crimson Red
        "#4169E1",  # Royal Blue
        "#50C878",  # Emerald Green
        "#DAA520",  # Goldenrod
    ]

    plots = []
    colors = kwargs.get("colors", ["gray" if not highlight_list else "lightgray"])
    highlight_colors = kwargs.get("highlight_colors", default_highlight_colors)

    # Extend the highlight_colors list if there are more highlighted groups than colors
    if len(highlight_list) > len(highlight_colors):
        highlight_colors = highlight_colors * (
            len(highlight_list) // len(highlight_colors) + 1
        )

    for i, group_value in enumerate(data[group_by].unique()):
        # Filter the DataFrame for the specified value
        filtered_data = data.query(f"{group_by} == @group_value")

        # Determine the color for the plot
        plot_color = (
            highlight_colors[highlight_list.index(group_value)]
            if group_value in highlight_list
            else colors[i % len(colors)]
        )

        # Create a line plot for the specified value
        line_plot = filtered_data.hvplot(color=plot_color, x=x, by=group_by, alpha=0.9)

        # Create a scatter plot for the specified value
        scatter_plot = filtered_data.hvplot.scatter(color=plot_color, x=x, by=group_by)

        # Combine the line plot and scatter plot
        plot = line_plot * scatter_plot
        plots.append(plot)

    # Overlay the plots
    combined_plot = hv.Overlay(plots)

    return combined_plot


@pn.depends(yearly_player.param.value)
def get_dog_age_butterfly_plot(roster):
    """
    Decorated with @pn.depends, this function generates a butterfly plot of male and female dog age distributions for a given roster year.

    Parameters:
    roster (int): The roster year to filter the dog data by.

    Returns:
    hvplot: A butterfly plot of male and female dog age distributions for the given roster year.
    """
    # Define bar plot options
    bar_opts = dict(
        invert=True,
        height=500,
        width=400,
        rot=90,
        xlim=(0, 24),
        xlabel="",
        yaxis="bare",
        ylabel="Count",
    )
    # Filter the DataFrame for the roster
    filtered_dog_data = pd.read_csv("../data/processed_dog_data.csv")
    roster_dog_data = filtered_dog_data.query(f"roster=={roster}")
    # Filter for the is_male_dog
    male_roster_dog_data = roster_dog_data.loc[roster_dog_data["is_male_dog"]]
    male_roster_dog_data = (
        male_roster_dog_data.groupby(["dog_age"])
        .size()
        .reset_index(name="age_frequency")
    )
    male_roster_dog_data = male_roster_dog_data.set_index("dog_age")
    total_male = male_roster_dog_data["age_frequency"].sum()
    male_plot = male_roster_dog_data.hvplot.bar(
        **bar_opts,
        ylim=(0, 620),
        title=f"Male Dog Age Distribution || {roster} || {total_male} Canines",
        color="skyblue",
    ).opts(active_tools=["box_zoom"])

    female_roster_dog_data = roster_dog_data[~roster_dog_data["is_male_dog"]]
    female_roster_dog_data = (
        female_roster_dog_data.groupby(["dog_age"])
        .size()
        .reset_index(name="age_frequency")
    )
    female_roster_dog_data = female_roster_dog_data.set_index("dog_age")
    total_female = female_roster_dog_data["age_frequency"].sum()
    female_roster_dog_data["age_frequency"] = (
        -1 * female_roster_dog_data["age_frequency"]
    )
    female_plot = female_roster_dog_data.hvplot.bar(
        **bar_opts,
        ylim=(-620, 0),
        title=f"Female Dog Age Distribution || {roster} || {total_female} Canines",
        color="pink",
    ).opts(active_tools=["box_zoom"])
    return (female_plot + male_plot).opts(
        shared_axes=False,
    )

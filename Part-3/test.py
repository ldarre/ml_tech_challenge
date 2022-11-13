# This notebook contains some tools and examples to cluster products based on textual descriptions,
# identify available product types, and use the latter to assign each product in a given dataset
# to the corresponding product type.

# %% ----- Imports -----
from ast import keyword, operator
from operator import mod
import pandas as pd
import numpy as np
from requests import get
import umap
import hdbscan
import umap.plot
from top2vec import Top2Vec
from sentence_transformers import SentenceTransformer
from multi_rake import Rake

# %% ----- Utils -----
def clean_dataset(
    data: pd.DataFrame,
    check_only: bool = True,
    dup_cols: list = [],
    nan_cols: list = [],
) -> pd.DataFrame:

    # Check raw dataset info
    print(f"Raw dataset info:\n")
    print(f"{data.info()}\n")
    # Check duplicates
    if len(dup_cols) > 0:
        for col in dup_cols:
            print(
                f"Number of duplicates in column {col}: {data.duplicated(subset=[col]).sum()}\n"
            )

    if not check_only:
        # Drop NaNs
        data.dropna(subset=nan_cols, inplace=True)
        # Drop duplicates
        data.drop_duplicates(subset=dup_cols, inplace=True)

        # Check dataset info after cleaning
        print(f"Cleaned dataset info:\n")
        print(f"{data.info()}\n")

    return data


def transform_data_lowecase(data: pd.DataFrame, lower_cols: list = []) -> pd.DataFrame:

    if len(lower_cols) > 0:
        for col in lower_cols:
            data[f"{col}"] = data[col].apply(lambda x: x.lower())
    else:
        print("No column to lower case was specifed. The dataframe won't be modified")
    return data


def keep_alpha(text: str) -> str:

    return " ".join(x for x in text.split(" ") if (x.isalpha() or ":" in x))


def clean_text(text: str, rm_words: list, min_word_len: int = 0) -> str:

    if len(rm_words) > 0:
        for w in rm_words:
            text = text.replace(w, " ")
    # else:
    #     print("No words to remove were specified.")
    text = " ".join(x for x in text.split(" ") if len(x) > min_word_len)

    return text


def combine_cols(
    data: pd.DataFrame,
    col_1: str = "",
    col_2: str = "",
    connector: str = " ",
    new_col_name: str = "new_col",
) -> pd.DataFrame:

    if col_1 != "" and col_2 != "":
        data[new_col_name] = data[col_1] + connector + data[col_2]
    else:
        print("Missing column(s) to combine. Dataset won't be modified")

    return data


def dim_reduction(
    embeddings_matrix,
    umap_config: dict,
    umap_use: str = "visualization",
    random_init: int = 124,
):
    """Use UMAP to reduce embeddings dimensionality

    Args:
        embeddings_matrix (numpy array): matrix with embeddings as rows
        umap_config (dict): umap config options
        config (str): umap usage options: visualization or clustering, Defaults to visualization.
        random_init (int, optional): random initialization. Defaults to 124.

    Returns:
        UMAP object
    """
    red_embeddings = None
    if umap_use == "visualization":
        red_embeddings = umap.UMAP(
            n_neighbors=umap_config[umap_use]["n_neighbors"],
            n_components=umap_config[umap_use]["n_components"],
            metric=umap_config[umap_use]["metric"],
            min_dist=umap_config[umap_use]["min_dist"],
            random_state=random_init,
        ).fit(embeddings_matrix)
    elif umap_use == "clustering":
        red_embeddings = umap.UMAP(
            n_neighbors=umap_config[umap_use]["n_neighbors"],
            n_components=umap_config[umap_use]["n_components"],
            metric=umap_config[umap_use]["metric"],
            min_dist=umap_config[umap_use]["min_dist"],
            random_state=random_init,
        ).fit_transform(embeddings_matrix)

    return red_embeddings


def plot_embeddings(
    umap_embeddings_vis, cluster_labels, data, do_connections: bool = False
):
    """Function to plot clustering results on UMAP embedding (for visualization:2D) space

    Args:
        umap_embeddings_vis: reduced dimension embeddings for visualization (2D)
        cluster_labels (numpy array): cluster labels for colouring points.
        data (pandas dataframe): dataframe with products information for interactive plot.
        do_connections (bool, optional): Whether to plot connections plot. Defaults to False.
    """

    f = umap.plot.interactive(
        umap_embeddings_vis,
        # labels=list(data.index),
        theme="fire",
        hover_data=data,
        point_size=5,
        # labels=data["Water Footprint cc water/g o cc of food TYPOLOGY"]
        labels=cluster_labels,
    )
    umap.plot.show(f)

    if do_connections:
        umap.plot.connectivity(umap_embeddings_vis, edge_bundling="hammer")


def get_keyword(text: str, keyword_buffer_list: list) -> dict:

    words = text.split(" ")
    words_freq = {}
    for w in words:
        if w not in words_freq.keys():
            words_freq[w] = 1
        else:
            words_freq[w] += 1
    word_freq_sorted = sorted(
        words_freq.items(), key=lambda item: item[1], reverse=True
    )
    keywords = {}
    for f in keyword_buffer_list:
        top_words = [x for x in word_freq_sorted if x[1] >= word_freq_sorted[0][1] * f]
        top_keyword = " ".join(x[0] for x in top_words)
        keywords[f] = top_keyword

    return words_freq, top_words, keywords


# %% ----- Settings -----
## Choose pre-trained model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

## Configuration
umap_config = {
    "visualization": {
        "n_neighbors": 20,
        "n_components": 2,
        "min_dist": 0.05,
        "metric": "cosine",
    },
    "clustering": {
        "n_neighbors": 20,
        "n_components": 100,
        "min_dist": 0.05,
        "metric": "cosine",
    },
}

TEXT_EMBED_COL = "text_to_embed"

# %% ----- Load Data -----
DATA_PATH = "./data"
DATA_FILE_NAME = "final_data.xlsx"
DATA_FILE_PATH = f"{DATA_PATH}/{DATA_FILE_NAME}"
data = pd.read_excel(DATA_FILE_PATH)
# %% ----- Clean Data -----
data = clean_dataset(
    data, check_only=False, dup_cols=["products"], nan_cols=["products"]
)
# %% ----- Pre-process products description text -----
data[TEXT_EMBED_COL] = data["products"]
# %%
data = combine_cols(
    data,
    col_1="category",
    col_2="products",
    new_col_name=TEXT_EMBED_COL,
    connector=": ",
)
# %%
data = transform_data_lowecase(data, lower_cols=[TEXT_EMBED_COL])
data[TEXT_EMBED_COL] = data[TEXT_EMBED_COL].transform(
    lambda x: clean_text(x, rm_words=[".", " grs"], min_word_len=2)
)
data[TEXT_EMBED_COL] = data[TEXT_EMBED_COL].transform(lambda x: keep_alpha(x))
# data.drop(labels=["category_products"], axis=1, inplace=True)

# %% ----- Generate product embeddings ------

prod_embedings = model.encode(data[TEXT_EMBED_COL].to_list())
# %%
# prod_embed_dict = {
#     prod: emb for prod, emb in zip(data["category_products_lower"], prod_embedings)
# }
# prod_matrix = np.array([x for x in prod_embed_dict.values()])

# %%
prod_matrix = np.array([x for x in prod_embedings])
# %% ----- UMAP embedding dimensionality reduction -----

umap_embeddings_vis = dim_reduction(prod_matrix, umap_config, umap_use="visualization")

umap_embeddings_clu = dim_reduction(prod_matrix, umap_config, umap_use="clustering")
# %% ----- Cluster UMAP embeddings -----

cluster_labels = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=13).fit_predict(
    umap_embeddings_clu
)
data["cluster_id"] = cluster_labels

# %% ----- Visualize embeddings -----
# plot_embeddings(umap_embeddings_vis, cluster_labels, data, do_connections=True)

# %%
data.shape

# %%
cluster_groups = data.groupby(by=["cluster_id"])
# %%
cluster_groups_size_df = cluster_groups.size().reset_index(name="count")
# %%
group_id = 10
cluster_groups.get_group(name=group_id)
# %%
print("Percentage of clustered products:")
100 / (
    cluster_groups_size_df["count"].sum()
    / (
        cluster_groups_size_df["count"].sum()
        - cluster_groups_size_df.loc[cluster_groups_size_df["cluster_id"] == -1][
            "count"
        ]
    )
)
# %%
cluster_groups_size_df
# %%
# keyword_buffer_list = [1.0, 0.9, 0.8, 0.6, 0.4]
keyword_buffer_list = [0.6]
keyword_buffer_keep = 0.6
cluster_id_keyword_map = {}
for k in cluster_groups.groups.keys():
    corpus = " ".join(
        x for x in cluster_groups.get_group(name=k)["text_to_embed"].to_list()
    )
    words_freq, top_words, keywords = get_keyword(
        corpus, keyword_buffer_list=keyword_buffer_list
    )
    cluster_id_keyword_map[k] = keywords[keyword_buffer_keep]

# %%
cluster_id_keyword_map[group_id]
# %%
data["keywords"] = data["cluster_id"].transform(lambda x: cluster_id_keyword_map[x])
# %%
sample500 = data.loc[data["cluster_id"] != -1].sample(500)
# %%
sample500.to_csv("./sample500.csv")
data[["category", "subcat", "products", "prices", "cluster_id", "keywords"]].to_csv(
    "./clustered_data_v1_no_duplicates.csv"
)
# %%
cluster_id_keyword_map
# %%
type(cluster_id_keyword_map[0])
# %%
sample500.head(20)
# %%
data.loc[data["cluster_id"] == 277]
# %%
data.columns
# %%
data.sort_values(by=["cluster_id"], inplace=True)
# %%

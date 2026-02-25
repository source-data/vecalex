# vecalex

Augment OpenAlex entities with vector representations computed from their associated works' abstracts.

## How it works

To get the vector representation of an OpenAlex entity like an author, `vecalex` performs the following steps:

1. Retrieve the works associated with that entity (e.g., works authored by that author) using the OpenAlex API.
2. Extract the abstracts from these works.
3. Embed the abstracts using a specified embedding model (e.g., a `sentence-transformers` model).
4. Aggregate the resulting vectors (e.g., by averaging) to produce a single vector representation for the entity.

## Usage

Basic examples:

### Which journal should I submit my article to?

```python
from pyalex import Journals
from vecalex import Scope

my_abstract = """
In this study, we explore the applications of machine learning in genomics. We
develop novel algorithms to analyze large-scale genomic data, demonstrating
improved accuracy in predicting gene expression patterns. Our findings highlight
the potential of integrating machine learning techniques in genomic research.
"""

# compute the scope of my abstract
my_scope = Scope(my_abstract)

# fetch highly cited journals
journals = Journals().sort(cited_by_count="desc").get()

# find the most similar journals to my scope
closest_journals, similarities = my_scope.closest(journals, top_n=3)

for rank, (journal, similarity) in enumerate(zip(closest_journals, similarities), start=1):
    print(f"{rank}. {journal['display_name']} (similarity: {similarity:.2f})")
# Sample output:
# 1. Nature Genetics (similarity: 0.89)
# 2. Genome Research (similarity: 0.85)
# 3. PLOS Genetics (similarity: 0.82)
```

### Are the most-cited researchers at EMBL working on similar topics?

```python
import pandas as pd
import plotly.express as px

from pyalex import Authors, Institutions
from vecalex import Scope

# fetch top authors at EMBL
embl = Institutions().search("EMBL").get()[0]
top_authors = Authors().filter(affiliations={"institution": {"id": embl["id"]}}).sort(cited_by_count="desc").get()[:5]

# compute pairwise similarities
similarities = Scope(top_authors).similarities(top_authors)

# display similarity matrix
names = [author["display_name"] for author in top_authors]
fig = px.imshow(pd.DataFrame({
    "x": names,
    "y": names,
    "similarity": similarities
}))
fig.show()

# Sample output: a heatmap showing high similarity among the top EMBL researchers, indicating they work on related topics.
```

## Configuration

### OpenAlex API Key

Required for retrieving entity metadata and abstracts from the OpenAlex API via the `pyalex` package.

```python
import pyalex

pyalex.config.api_key = "<YOUR_API_KEY>"
```

### Work Retrieval

Configure how many works and in which order to retrieve for OpenAlex entities like authors, institutions, etc.
Only works with abstracts will be considered.

```python
from vecalex import config

config.max_works_per_entity = 100     # default: 20
config.work_sorting = "display_name"  # default: "publication_date:desc"
```

If you want to provide a custom work retrieval function (e.g. to fetch works from the OpenAlex snapshot), you can do so as follows:

```python
from vecalex import config

def my_work_retrieval_function(entity_id: str) -> list[dict]:
    # must return a list of works (dicts) associated with the given entity_id,
    # each with at least an "abstract" or "abstract_inverted_index" field
    return ...

config.work_retrieval_function = my_work_retrieval_function
```

### Embedding Model

Either set a `sentence-transformers` model name or path:

```python
from vecalex import config

config.model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"  # default: EMBO/ModernBERT-neg-sampling-PubMed
```

Or provide a custom embedding function:

```python
import numpy as np
from vecalex import config

def my_embedding_function(texts: list[str]) -> np.ndarray:
    # must return a 2D numpy array of shape (len(texts), embedding_dim)
    return ...

config.embedding_function = my_embedding_function
```

### Entity Embeddings

Configure how to aggregate work vectors into an entity vector (e.g., by averaging):

```python
import numpy as np
from vecalex import config

def my_aggregate_embeddings(work_vectors: np.ndarray) -> np.ndarray:
    # must accept a 2D numpy array (num_works, embedding_dim)
    # and return a 1D vector (embedding_dim,)
    return ...

config.aggregate_embeddings = my_aggregate_embeddings
```

If you have precomputed entity vectors, you can provide a custom entity embedding function that retrieves them directly:

```python
import numpy as np
from vecalex import config

def my_entity_embedding_function(entity_id: str) -> np.ndarray:
    # must accept an entity_id and return a 1D vector
    return ...

config.entity_embedding_function = my_entity_embedding_function
```

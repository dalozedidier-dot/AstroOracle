API
===

FastAPI endpoints
-----------------

- ``GET /`` landing page
- ``POST /run`` upload candidates (CSV/Parquet) and run ranking
- ``POST /rank`` rank a list of candidates (JSON)
- ``POST /annotate`` write annotation rows
- ``GET /viz3d`` Plotly JSON figure
- ``GET /api/cutouts`` preview cutouts as PNG (base64)
- ``GET /graph_anomaly`` graph diagnostics + optional figure
- ``GET /explain`` lightweight explanations + shareable prompts
- ``GET /chaos`` chaos-style proxies (requires a time series column)
- ``GET /gaia/cone`` and ``GET /gaia/adql`` (requires ``.[astro]``)

Python modules
--------------

.. automodule:: astrooracle.chaos_metrics
   :members:

.. automodule:: astrooracle.graph_anomaly
   :members:

.. automodule:: astrooracle.explainability
   :members:

.. automodule:: astrooracle.hybrid_fusion
   :members:

Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e ".[dev,api,plotly,graph]"

Run the web UI
--------------

.. code-block:: bash

   astrooracle serve --candidates candidates.parquet --annotations annotations.csv

Then open http://127.0.0.1:8000/.

CLI triage
----------

.. code-block:: bash

   astrooracle run --candidates candidates.parquet --annotations annotations.csv

Graph anomaly context
---------------------

.. code-block:: bash

   astrooracle graph-anomaly --input candidates.parquet --output graph_anomaly.html

Explainability prompts
----------------------

.. code-block:: bash

   astrooracle explain-top --n 10 --output explanations.jsonl

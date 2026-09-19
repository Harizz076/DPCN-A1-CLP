# Opinion Network Analysis

## Method 1: People As Nodes

This is the code for the network that shows how similar people's survey answers are. It lives in `scripts/method-1/`.

## What you need

Open a terminal and go to the method 1 folder:

```
cd scripts/method-1
```

Then install these Python packages:

```
pip install pandas numpy networkx matplotlib
```

## How to run it

Run these two scripts in order, from inside `scripts/method-1/`.

### 1. Process the data

```
python3 process_data.py
```

This reads `data/Survey_Results_UC.csv`, cleans it up, and builds the network. It writes two files into `scripts/method-1/`:

- `report_data.json`: the main numbers, like nodes, edges, centrality, and communities
- `interactive_data.json`: a bigger dataset used by the 3D viewer

It also updates `respondent_network_3d.html` so the 3D viewer shows the newest data.

### 2. Make the plots

```
python3 generate_plots.py
```

This reads `report_data.json` and saves four chart images into a `figures/` folder:

- `percolation_analysis.png`
- `leaning_pie.png`
- `category_means_bar.png`
- `correlation_distribution.png`

### 3. Look at the 3D network

Just open `respondent_network_3d.html` in a browser. Double-click the file, or run:

```
open respondent_network_3d.html
```

or on Linux:

```
xdg-open respondent_network_3d.html
```

No server is needed. It works right from the file.

## Notes

- Always run `process_data.py` before `generate_plots.py`, since the plots are made from its output.
- Run both scripts again any time `Survey_Results_UC.csv` changes.

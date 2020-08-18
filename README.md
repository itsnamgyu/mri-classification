# mri-classification
Cardiac MRI classification using Recursive Neural Networks

## Getting Started

1. Install requirements
2. Copy data folder to root directory

```
mri-classification
├── data  <-- this one
├── data.py
├── labeler.py
├── LICENSE
├── notebooks
├── project.py
├── README.md
├── requirements.txt
├── setup.py
```

3. Install pip package (this is required for absolute imports)

```
pip install -e .
```

## Labeler Controls

- `h`: move backward by 10 slides
- `l`: move forward by 10 slides
- `left-arrow`: move backward by 1 slide
- `right-arrow`: move forward by 1 slide
- `[1-5]`: select current label (oap to obs)
- `left-click`: label the slice under the cursor
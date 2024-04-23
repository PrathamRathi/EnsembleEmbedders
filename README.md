# EnsembleEmbedders

## CSCI 2470 Final Project

### Dataset
We used LLMD-FULL from [Lakh MIDI Dataset v0.1](https://colinraffel.com/projects/lmd/). The compressed file name is [lmd_full.tar.gz](). extract the file under **data/**.
```
>wget **link**
>tar -zxvf **filename.tar.gz**
```
We have filtered midi files to contain lyrics, and split the latter into four sections. You can download our dataset called [LyricalLakh.zip](#)
### Virtual Environment
Make sure to make a python virtual environment
```
python -m venv .venv
```

Activate venv. this should make your terminal start with (.venv)
```
 source .venv/bin/activate
```
Windows?
```
> cd .venv/Scripts/
> activate.bat
> cd ../../
```

Now install dependencies in that venv
```
> pip install -r requirements.txt
```
Shortcut:
```
> python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
### Execution/Demo

Make sure before running scripts, that you are in the path
```
EnsembleEmbedders/
```

not in 
```
EnsembleEmbedders/src/
```
### Dependencies
```
prettymidi ==
tensorflow == 2.16.1
joblib == 1.4.0
```

### Acknowledgement

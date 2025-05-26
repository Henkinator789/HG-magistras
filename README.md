# Giliojo mokymosi objektų aptikimo modelių tyrimai pažeistiems medžiams klasifikuoti iš RGB vaizdų
Baigiamasis magistro studijų projektas

Kauno technologijos universitetas

Informatikos fakultetas

autorius: Henrikas Gricius

vadovas: Prof. Rytis Maskeliūnas

recenzentas: Prof. Tomas Blažauskas

Kaunas, 2025

## Įvadas
Projekto tyrimams naudojamas [MMDetection](https://github.com/open-mmlab/mmdetection) objektų aptikimo modelių įrankis [Python 3.9.13](https://www.python.org/downloads/release/python-3913/) aplinkoje.
Ligotų maumedžių duomenų rinkinys parsisiųstas iš [Forest Damages – Larch Casebearer 1.0](https://lila.science/datasets/forest-damages-larch-casebearer/), o šie duomenys surinkti [National Forest Data Lab](https://skogsdatalabbet.se/) Švedijoje.

## MMDetection instaliacija

Pirmiausia, paruošiama Python virtuali aplinka

```
virtualenv venv --python=your/path/to/Python/Python39/python.exe
venv/Scripts/activate
```

Instaliuojamas PyTorch

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Instaliuojamas openmim, o su juo mmengine ir mmcv

```
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
```

Galiausiai, klonuojamas mmdetection

```
git clone https://github.com/open-mmlab/mmdetection.git
```

## Duomenų rinkinio paruošimas
Duomenys parsisiunčiami ir padalijami į mokymo, validacijos ir testavimo rinkinius paleidus data_preparation.ipynb failą.

## Modelių konfigūraciniai failai
Modelių konfigūraciniai failai saugojami configs/ aplanke. Skirtingi metodai pažymėti failų pavadinimuose:

- fs - be perkėlimo mokymosi
- pt - iš anksto apmokyti svoriai (mokyti su COCO duomenimis naudojant numatytas konfigūracijas)
- bal - pritaikomas klasių balansavimas
- df - numatytieji duomenų papildymo metodai
- pmd - fotometriniai iškraipymai
- aff - geometriniai pakeitimai

## Modelių apmokymas ir testavimas
Apmokymas paleidžiamas terminale, pvz.:

```
python mmdetection/tools/train.py configs/fasterrcnn/fasterrcnn_fs_flip_hor.py
```

Pasibaigus mokymo procesui, geriausio modelio svoriai išsaugojami, pvz., weights/fasterrcnn_fs_flip_hor.pth, paleidžiamas testavimas, išsaugojamas .log ir .pkl failas

```
python mmdetection/tools/test.py configs/fasterrcnn/fasterrcnn_fs_flip_hor.py weights/fasterrcnn_fs_flip_hor.pth --out rezultatai/fasterrcnn/fasterrcnn_fs_flip_hor.pkl
```

## Modelių rezultatų generavimas

Painiavos matricą nupiešti galima paleidus modifikuotą (mmdet_scripts/) confusion_matrix.py programą

```
python mmdetection/tools/analysis_tools/confusion_matrix.py configs/fasterrcnn/fasterrcnn_fs_flip_hor.py rezultatai/fasterrcnn/fasterrcnn_fs_flip_hor.pkl your/path/to/save/img --show
```

Mokymo nuostolių kreives nupiešti galima paleidžiant plot_loss.py, kur reikia ties file1=, file2=... įrašyti atitinkamų mokymų scalars.json failų pavadinimus

ANOVA ir Tukey HSD analizės atliekamos paleidžiant anova_tykey_hsd.py programą (prieš tai rezultatų duomenys iš .log failų surenkami į atskirus .xlsx failus tyrimu_rezultatai/ aplanke).

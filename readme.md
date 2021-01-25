# PGPR
PGPR is a unified Bayesian framework that integrates a machine learning model that takes into account review features with peer grading for predicting review conformity.
## Installation
### 1. Create a virtual environment

We are using python3.6 in our implementation, you can create a virtual environment for PGPR using the following command:
``` bash
sudo apt-get install python3-venv
python3.6 -m venv env-pgpr
source env-pgpr/bin/activate
```
### 2. Install requirements
After cloning this repository, navigate inside it:
``` bash
git clone https://webconf_pgpr@bitbucket.org/webconf_pgpr/pgpr.git
cd pgpr
```
Install all requirements using the following command:
``` bash
pip install --upgrade pip
sudo apt-get install gcc python3-dev
pip install -r requirements.txt
```
If you would like to use PGPR with a GPU machine, you can install CUDA with the following comman line.
``` bash
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Running PGPR
To precit conformity of reviews from ICLR 2018, you can use the script: iclr_18.sh in the scripts folder
``` bash
cd code
sh ../scripts/iclr_18.sh
```


To precit conformity of reviews from ICLR 2019, you can use the script: iclr_19.sh in the scripts folder
``` bash
cd code
sh ../scripts/iclr_19.sh
```

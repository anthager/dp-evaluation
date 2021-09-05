## Installation locally
* After cloning the repository and setting up virtual environment, install requirements.
```bash
git clone git@github.com:anthager/smartnoise-tests.git && cd smartnoise-tests
python3 -m venv venv
. venv/bin/activate
cd analysis
pip install -r requirements.txt
```

## Run test on pi
* Sync the `/data` and the `/tests` directories 
``` bash 
scp -r ./analysis pi@noise.anton.pizza:~/smartnoise-test
```
* Ssh to the pi
``` bash
ssh noise.anton.pizza
cd smartnoise-test
```
* Run the test you want to run
``` bash
python analysis/tests/<test>.py
```
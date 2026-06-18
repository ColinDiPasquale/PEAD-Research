# Post earnings announcement drift prediction using machine learning
# By Colin DiPasquale

# This is just a controller file that will run the different steps in succession
# The code for each individual step can also be found in this repo
# The full research paper can also be found in this repo
# config.py is just to control output directories (for now, though I may add more controllable variables there down the road)
# Runtime is about 30 minutes on my i7 laptop

from step1_compustat import runStep1
from step2_crsp import runStep2
from step3_features  import runStep3
from step4_modeling  import runStep4

runStep1()
runStep2()
runStep3()
runStep4()
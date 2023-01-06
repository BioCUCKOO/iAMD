# iAMD
**Note**: For integration of multiomics  datasets, we developed an artificial intelligence (AI) framework, inference of autophagy regulators from multi-omic data (iAMD).

## Requirements

The main requirements are listed below:

* Python 3.8
* Numpy
* Scikit-Learn
* Joblib
* Keras
* Pandas
* Matplotlib

## The description of iAMD source codes

* PPI_proteomics_predictor.py

    The code is used for the Atg1-interacting partners proteomics model loading and data prediction.
* PPI_transcriptomics_predictor.py

    The code is used for the Atg1-interacting partners transcriptomics model loading and data prediction.
* PPI_phosphoproteomics_predictor.py

    The code is used for the Atg1-interacting partners phosphoproteomics model loading and data prediction.
* PPI_sequence_predictor.py

    The code is used for the Atg1-interacting partners sequence model loading and data prediction.
* PPI_predictor.py

    The code is used for the Atg1-interacting partners integrated LR model loading and data prediction.

* Substrate_proteomics_predictor.py

    The code is used for the Atg1 substrates proteomics model loading and data prediction.
* Substrate_transcriptomics_predictor.py

    The code is used for the Atg1 substrates transcriptomics model loading and data prediction.
* Substrate_phosphoproteomics_predictor.py

    The code is used for the Atg1 substrates phosphoproteomics model loading and data prediction.
* Substrate_sequence_predictor.py

    The code is used for the Atg1 substrates sequence model loading and data prediction.
* Substrate_predictor.py

    The code is used for the Atg1 substrates integrated LR model loading and data prediction.

## The models in iAMD

* PPI_proteomics_predictor_model.h5

    The model is used for the Atg1-interacting partners proteomics data prediction.
* PPI_transcriptomics_predictor_model.h5

    The model is used for the Atg1-interacting partners transcriptomics data prediction.
* PPI_phosphoproteomics_predictor_model.h5

    The model is used for the Atg1-interacting partners phosphoproteomics data prediction.
* PPI_sequence_predictor_model.h5

    The model is used for the Atg1-interacting partners sequence data prediction.
* PPI_predictor_model.pkl

    The model is used for the Atg1-interacting partners integrated LR prediction.

* Substrate_proteomics_predictor_model.h5

    The model is used for the Atg1 substrates proteomics data prediction.
* Substrate_transcriptomics_predictor_model.h5

    The model is used for the Atg1 substrates transcriptomics data prediction.
* Substrate_phosphoproteomics_predictor_model.h5

    The model is used for the Atg1 substrates phosphoproteomics data prediction.
* Substrate_sequence_predictor_model.h5

    The model is used for the Atg1 substrates sequence data prediction.
* Substrate_predictor_model.pkl

    The model is used for the Atg1 substrates integrated LR prediction.

## The description of Additional_Models

    The model, data and code are used to predict the potential functional interacting partners and substrates of TORC1, Gcn2, Rim15, Yak1, Slt2 and Npr1 in autophagy using iAMD.

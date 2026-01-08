# DEFT: Demystifying VLN Failures via a Unified Dual-View Explainability Framework for LLM-based Agents

Large Language Models (LLMs) have emerged as central planners in Vision-and-Language Navigation (VLN), yet their complexity increasingly obscures their internal decision-making. Existing interpretability methods typically isolate temporal criticality from feature salience, creating an alignment gap and failing to account for the behavioral instability of black-box agents. To address this, we propose DEFT, a unified dual-view framework that demystifies agent behavior by jointly analyzing when a decision is pivotal and what visual evidence grounds it. Featuring a dual-head architecture with a shared latent representation, DEFT employs a Mask Head for counterfactual-based criticality detection and an Action Head that leverages an ensemble of surrogates to recover robust visual cues. Extensive experiments on MatterPort3D across three LLM-based agents demonstrate that DEFT outperforms baselines in both temporal and feature fidelity. User studies further validate its utility, showing 78% alignment with human intuition.
## Environment Setup

### Prerequisites

- Python 3.10
- Matterport3D Simulator (out-of-docker setup)

### Installation

1. **Install Matterport3D Simulator** (out-of-docker):
   ```bash
   # Follow instructions at https://github.com/peteanderson80/Matterport3DSimulator
   # We use the latest version instead of v0.1
   ```

2. **Create conda environment**:
   ```bash
   conda create -n DEFT python=3.10
   conda activate DEFT
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_feature_eval.txt  # For feature-level evaluation
   ```

4. **Set up MapGPT** (if using MapGPT as target agent):
   ```bash
   # Follow MapGPT/README.md for additional setup
   # Set your OpenAI API key in MapGPT/GPT/api.py
   ```

### Data Preparation

- Place R2R dataset annotations in `data/` directory
- Download preprocessed data files (e.g., `R2R_MapGPT_72_scenes_processed.json`)
- Prepare image features or use pre-extracted features in `img_features/`

## Usage

### Timestep-level Explanations

Timestep-level explanations identify which navigation timesteps are most important for the model's decision-making process.

#### 1. Train Surrogate Model

First, train a surrogate model that approximates the target agent's behavior:

```bash
bash run/train_surrogate.bash
```

This script:
- Trains a VLN-BERT surrogate model on surrogate data
- Uses the surrogate model to approximate the target agent (e.g., MapGPT)
- Saves the trained model to `snap/VLNBERT-train-Surrogate/`

#### 2. Generate Timestep-level Explanations

Generate timestep-level explanations using the trained surrogate:

```bash
bash run/test_mask_mapgpt.bash
```


### Feature-level Explanations

Feature-level explanations generate pixel-level or region-level saliency maps showing which visual features influence the model's actions.

#### 1. Train Bagging Ensemble

Train multiple agents using bootstrap sampling for robust feature-level explanations:

```bash
bash run/train_bagging.bash
```


#### 2. Generate Feature-level Explanations

Generate feature-level explanations using the ensemble:

```bash
bash run/test_feature_mapgpt_ensemble.bash
```


## Project Structure

```
DEFT/
├── r2r_src/              
│   ├── agent_mask.py    
│   ├── agent_feature.py 
│   ├── agent_feature_ensemble.py  
│   ├── train_mask.py    
│   └── vlnbert/         
├── MapGPT/              
├── NavGPT/              
├── NavGPT_2/           
├── configs/           
│   ├── test_mask_mapgpt.yaml      
│   ├── test_feature_mapgpt_ensemble.yaml  
│   └── bagging.yaml     
├── run/                
│   ├── train_surrogate.bash
│   ├── test_mask_mapgpt.bash
│   ├── train_bagging.bash
│   └── test_feature_mapgpt_ensemble.bash
├── data/              
├── docs/             
└── scripts/         
```


## Related Projects

- [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT)
- [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)

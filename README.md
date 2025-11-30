# Factory ACT


## AI training
### Setup
Create conda environment
~~~
conda create --name act python=3.9
conda activate act
~~~

Install torch (for reference we add the versions we are using)
~~~
conda install pytorch==1.13.1 torchvision==0.14.1
~~~

You can now install the requirements:
~~~
pip install -r requirements.txt
~~~

Go to `TASK_CONFIG` in `config/config.py` and change the paths of the ports that connect leader and follower robots to your computer. 

You will also need to connect a camera to your computer and point it towards the robot while collecting the data via teleoperation. You can change the camera port in the config (set to 0 by default). It's important the camera doesn't move otherwise evaluation of the policy is likely to fail. 

### Data collection
In order to collect data simply run:
~~~
python record_episodes.py --task sort
~~~
You can define the name of the task you are doing and the episodes will be stored at `data/<task>`. You can also select how many episodes to collect when running the script by passing the argument `--num_episodes 1` (set to 1 by default). After getting a hold of it you can easily do 20 tasks in a row.

Turn on the volume of your pc-- data for each episode will be recorded after you hear "Go" and it will stop when you hear "Stop".

### Train policy
We slightly re-adapt [Action Chunking Tranfosrmer](https://github.com/tonyzhaozh/act/tree/main) to account for our setup. To start training simply run:
~~~
python train.py --task sort
~~~
The policy will be saved in `checkpoints/<task>`.

### Evaluate policy
Make sure to keep the same setup while you were collecting the data. To evaluate the policy simply run:
~~~
python evaluate.py --task sort
~~~

## Training Configuration
The model was trained using the following hyperparameters (as defined in `config/config.py`):
- **Epochs**: 2000
- **Batch Size**: 4
- **Learning Rate**: 1e-5
- **Backbone**: ResNet18
- **Transformer Architecture**: 
    - Encoder Layers: 4
    - Decoder Layers: 7
    - Attention Heads: 8
    - Hidden Dimension: 512
    - Feedforward Dimension: 3200

## Project Findings & Analysis

### Simulation Results
The following video demonstrates the policy's performance in the simulation environment.

<video src="sim_result.mp4" width="640" height="480" controls></video>

[Watch Simulation Result](sim_result.mp4)

### Prediction Analysis
We compared the model's predicted trajectory against the expert demonstration on an unseen test episode.

![Prediction Analysis](prediction_analysis.png)

**Interpretation**:
The plot above visualizes the joint positions (Base, Shoulder, Elbow) over time:
- **Black Dashed Line**: Ground truth (Expert trajectory).
- **Colored Lines**: ACT Model prediction.


## Future Work (v2)
We are currently working on **v2** of this project. While the current results are hacky, we are actively exploring several improvement strategies:
- **Data Augmentation**: Implementing more aggressive augmentation to handle lighting and pose variations.
- **Model Scaling**: Testing larger transformer backbones to capture finer motor skills.
- **Sim-to-Real**: Focusing on domain randomization to facilitate smoother deployment to the physical robot.

Stay tuned for updates!

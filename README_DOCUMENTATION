The `Episode` Class in online.py allow us to specify the target and model, and do on `step` at a time. It also support taking in a arbitrary images (300 by 300) and return an action by the agent.

```
ClASS Episode

    Purpose: A wrapper class that include an agent and an Ai2thor controller for stepwise observation-action cycle.
    
    Parameters:
        controller: Ai2thor controller
        target: (str) target object name
        model_name: (str) the name of navigation model. 'SAVN', 'GCN', 'NON_ADAPTIVE_A3C'.
        model_path: (str) path to the pretrained model. Must correspond to 'model_name'
        glove_file_path: (str) path to Glove Embedding
    
    Methods:
        step(self) - the agent takes an action based on the current observation
        isolated_step(self, image) - the agent takes an action based on the input 'image'.
        stop(self) - stop the Ai2thor controller
        
```

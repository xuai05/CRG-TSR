class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(self, state=None, hidden=None,  graphmemory=None, detection_inputs=None, action_probs=None,state_name=None,scene_name=None):
        self.state = state
        self.hidden = hidden
        self.graphmemory = graphmemory
        # self.global_feature_memory = global_feature_memory
        self.detection_inputs = detection_inputs
        self.action_probs = action_probs
        # modify
        self.state_name = state_name
        self.scene_name = scene_name
        self.trainflag = True


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, graphmemory=None, embedding=None):
        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.graphmemory = graphmemory
        # self.global_feature_memory=global_feature_memory
        self.embedding = embedding

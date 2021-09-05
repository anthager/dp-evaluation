#!/usr/bin/python3.7

from dpevaluation.utils.log import log

# Tensorflow
try:
    import tensorflow as tf
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
except ImportError:
    pass
# PyTorch
try:
    import torch
    import numpy as np
    from opacus import PrivacyEngine
    from opacus.dp_model_inspector import DPModelInspector
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from tqdm import tqdm
except ImportError:
    pass
# Diffprivlib
try:
    from sklearn.linear_model import LinearRegression as LinearRegressionDPL
    from diffprivlib.models import LinearRegression as DPLinearRegressionDPL
except ImportError:
    pass


class ModelFactory:
    def __init__(self, lib: str):
        """
        TF and Torch models.
        Pass exception if there is no tf or torch module.
        """
        try:
            self.models = {"tensorflow_privacy":
                           {
                               "LinearRegression": LinearRegressionTF,
                               "LogisticRegression": None
                           }
                           }[lib]
        except Exception:
            pass
        try:
            self.models = {"opacus":
                           {
                               "LinearRegression": LinearRegressionTorch,
                               "LogisticRegression": None
                           }
                           }[lib]
        except Exception:
            pass

        self.lib = lib

    def build_model(self, model: str, optimizer: str, config: dict):
        """
        Diffprivlib takes epsilon as DP parameter and has different DP setup
        for various classifiers and regressors in contrast to TF and Opacus.
        """
        if self.lib == "diffprivlib":
            mode = "non_private" if config["epsilon"] == -1 else "private"

            if model == "LinearRegression":
                if mode == "private":
                    _model = DPLinearRegressionDPL(
                        epsilon=config["epsilon"],
                        bounds_X=config["bounds_X"],
                        bounds_y=config["bounds_y"])
                else:
                    _model = LinearRegressionDPL()

            elif model == "LogisticRegression":
                pass
        else:
            mode = "non_private" if config["noise_multiplier"] == -1 \
                else "private"
            _model = self.models[model](config["input_shape"])

            if self.lib == "tensorflow_privacy":
                # Build private or plain optimizer
                _optimizer = self.__build_tf_optimizer(mode, optimizer, config)
                _model.compile(
                    optimizer=_optimizer,
                    loss=_model.loss,
                    metrics=_model.metrics_.values())

            elif self.lib == "opacus":
                _optimizer = \
                    self.__build_torch_optimizer(
                        mode, _model, optimizer, config)
                # Attach PrivacyEngine for private learning
                pe = self.__attach_privacy_engine(
                    _model, _optimizer, config) if mode == "private" else None
                # Attach fit and evaluation functions
                _model = \
                    PyTorchModule(_model, _optimizer, pe, config["epochs"])

        return _model

    def __build_tf_optimizer(self, mode, optimizer, config):
        """
        If the updates are noisy (such as when the additive noise is large compared to the 
        clipping threshold), the learning rate must be kept low for the training procedure 
        to converge.
        """
        if optimizer == "SGD":
            # DP-SGD
            if mode == "private":
                _optimizer = DPKerasSGDOptimizer(
                    learning_rate=config["learning_rates"]["private"],
                    noise_multiplier=config["noise_multiplier"],
                    l2_norm_clip=config["l2_norm_clip"],
                    num_microbatches=config["num_microbatches"])
            # SGD
            else:
                _optimizer = tf.keras.optimizers.SGD(
                    learning_rate=config["learning_rates"]["non_private"],
                    momentum=config["momentum"],
                    nesterov=config["nesterov"])

        elif optimizer == "Adam":
            pass

        return _optimizer

    def __build_torch_optimizer(self, mode, model, optimizer, config):
        if optimizer == "SGD":
            """
            Compared to the non-private training, Opacus-trained models converge with a smaller
            learning rate (each gradient update is noisier, thus we want to take smaller steps).
            """
            lr = config["learning_rates"]["private"] if mode == "private" else config["learning_rates"]["non_private"]
            _optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config["momentum"],
                nesterov=config["nesterov"])

        elif optimizer == "Adam":
            pass

        return _optimizer

    def __attach_privacy_engine(self, model, optimizer, config):
        inspector = DPModelInspector()
        inspector.validate(model)
        """
        Secure RNG turned off: Fine for experimentation as it allows for much
        faster training performance, but remember to turn it on and retrain one
        last time before production with 'secure_rng' turned on.
        """
        pe = PrivacyEngine(
            model,
            sample_size=config["sample_size"],
            batch_size=config["batch_size"],
            max_grad_norm=config["max_grad_norm"],
            noise_multiplier=config["noise_multiplier"],
            alphas=config["alphas"],
            secure_rng=config["secure_rng"],
            target_delta=config["target_delta"],
            epochs=config["epochs"])

        pe.attach(optimizer)

        return pe


"""
TF and Torch models.
Pass exception if there is no tf or torch module.
"""
try:
    class LinearRegressionTF(tf.keras.Model):
        def __init__(self, _input_shape):
            super(LinearRegressionTF, self).__init__()
            self.loss = "mean_squared_error"
            # Conflict with existing metric variable
            self.metrics_ = {
                "mape": "mean_absolute_percentage_error"
            }
            self.dense = tf.keras.layers.Dense(units=1,
                                               input_shape=(_input_shape[1],),
                                               activation="linear")

        def call(self, x):
            return self.dense(x)
except NameError:
    pass
try:
    class LinearRegressionTorch(torch.nn.Module):
        def __init__(self, input_shape):
            super(LinearRegressionTorch, self).__init__()
            """
            "Using a non-full backward hook when the forward contains multiple autograd Nodes"
            https://stackoverflow.com/questions/65011884/understanding-backward-hooks
            https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L937
            """
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            # Layers
            self.hid = torch.nn.Linear(input_shape[1], 1)
            # Mean squared error
            self.loss = torch.nn.MSELoss()
            self.loss_functions = {
                "mape": self.__MAPELoss
            }
            self.metrics = {
                "mse": mean_squared_error,
                "mape": mean_absolute_percentage_error
            }

        def forward(self, x):
            return self.hid(x)

        def __MAPELoss(self, y_pred, actual):
            return torch.mean(torch.abs((actual - y_pred) / actual))
except NameError:
    pass


class PyTorchModule():
    def __init__(self, model, optimizer, privacy_engine, epochs):
        self.model = model
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine
        self.epochs = epochs
        # Check this once here
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        # Privacy engine is None for non-private training
        try:
            self.privacy_engine.to(self._device)
        except AttributeError:
            pass

    def fit(self, mode, train_loader, delta):
        metrics = dict()

        for epoch in range(self.epochs):
            self.model.train()
            losses = dict()

            for inputs, target in tqdm(train_loader):
                inputs, target = \
                    inputs.to(self._device), target.to(self._device)
                # Clear the gradients
                self.optimizer.zero_grad()
                # Calc predictions
                y_pred = self.model(inputs)
                # Calc loss
                loss = self.model.loss(y_pred, target)
                # Compute gradients
                loss.backward()
                # Save loss
                losses.setdefault("loss", []).append(loss.item())
                # Calc & save other loss
                for name, lf in self.model.loss_functions.items():
                    loss = lf(y_pred, target).item()
                    losses.setdefault(name, []).append(loss)
                # Update model weights
                self.optimizer.step()
            # Append average loss per epoch
            for name, metric in losses.items():
                metrics.setdefault(name, []).append(np.mean(metric))
            # Calc epsilon
            if mode == 'private':
                epsilon, opt_order = \
                    self.optimizer.privacy_engine.get_privacy_spent(delta)
                dp_metric = f"(ε={epsilon:.4f}, δ={delta:.4E}, α={opt_order:.1f})"

            log("debug", "Epoch: {}, loss: {:.4f}, {}"
                .format(epoch + 1, metrics["loss"][-1], dp_metric if mode == 'private' else ''))

        return metrics

    def evaluate(self, test_loader):
        self.model.eval()
        predictions, y_test = list(), list()
        metrics = dict()
        """
        This saves run time and memory consumption and won't affect accuracy.
        Basically skips the gradient calculation over the weights, which
        means we are not changing any weight in the specified layers.
        """
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs, target = \
                    inputs.to(self._device), target.to(self._device)
                y_pred = self.model(inputs)
                # Retrieve prediction numpy array
                y_pred = y_pred.detach().numpy()
                # Get actual target values
                actual = target.numpy()
                # Save
                predictions.append(y_pred)
                y_test.append(actual)
        # Extract values
        predictions = np.vstack(predictions).flatten()
        y_test = np.vstack(y_test).flatten()
        # Calc metrics
        for name, metric in self.model.metrics.items():
            metrics[name] = metric(y_test, predictions)

        return metrics, y_test, predictions

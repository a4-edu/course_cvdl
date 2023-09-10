import numpy as np


class TestLayer:
    @classmethod
    def error_message(cls, meta, hint):
        return f"[{meta['description']}]:{hint}"

    @classmethod
    def check_tensors_equality(cls, query, reference, meta, prefix_message, check_values=True):
        assert isinstance(query, np.ndarray), cls.error_message(
            meta,
            prefix_message + f"- expected numpy array, got {type(query)}"
        )
        assert (query.shape == reference.shape), cls.error_message(
            meta,
            prefix_message + f"- expected shape {reference.shape}, got {query.shape}"
        )
        if check_values:
            assert np.allclose(query, reference, rtol=1e-5, atol=2e-5), cls.error_message(
                meta,
                prefix_message + f"- values mismatch, max diff is {np.max(np.abs(query - reference))}"
            )

    def test_init(self, test_data, test_layer_cls):
        values = test_data['values']
        meta = test_data['meta']
        layer_kwargs = meta['layer_kwargs']

        # check layer creation
        try:
            layer = test_layer_cls(**layer_kwargs)
        except Exception as exc:
            raise AssertionError(
                self.error_message(meta, f"Failed __init__ with kwargs '{layer_kwargs}':{exc}")
            )
        # check layer parameters
        expected_parameters = [np.array(p) for p in test_data['values']['parameters']]
        assert len(layer.parameters) == len(values['parameters']),\
            self.error_message(meta, f"Expected layer to have {len(values['parameters'])} parameters, got {len(layer.parameters)}")
        for i in range(len(layer.parameters)):
            self.check_tensors_equality(
                layer.parameters[i],
                expected_parameters[i],
                meta,
                f"From __init__, .parameters[{i}]",
                check_values=False
            )
        layer.parameters = expected_parameters
        return layer

    def test_forward(self, test_data, test_layer_cls):
        layer = self.test_init(test_data, test_layer_cls)
        values = test_data['values']
        meta = test_data['meta']

        try:
            output = layer.forward(np.array(test_data['values']['input']))
        except Exception as exc:
            raise AssertionError(
                self.error_message(meta, f"Failed .forward with exception type '{type(exc)}'")
            )

        expected_output = np.array(test_data['values']['output'])
        self.check_tensors_equality(output, expected_output, meta, f"From .forward")
        return layer

    def test_backward(self, test_data, test_layer_cls):
        layer = self.test_forward(test_data, test_layer_cls)
        output_grads = np.array(test_data['values']['output_grads'])
        expected_input_grads = np.array(test_data['values']['input_grads'])
        meta = test_data['meta']
        try:
            input_grads = layer.backward(output_grads)
        except Exception as exc:
            raise AssertionError(
                self.error_message(meta, f"Failed .backward with exception type '{type(exc)}'")
            )
        self.check_tensors_equality(input_grads, expected_input_grads, test_data['meta'], "From .backward, output_grad")
        return layer

    def test_parameters_grads(self, test_data, test_layer_cls):
        layer = self.test_backward(test_data, test_layer_cls)
        expected_parameters_grads = [ np.array(param_grad) for param_grad in test_data['values']['parameters_grads']]
        meta = test_data['meta']

        assert len(layer.parameters_grads) == len(expected_parameters_grads),\
            self.error_message(meta,
                f"Expected layer to have {len(expected_parameters_grads)} parameter_grads, found {len(layer.parameters_grads)}"
        )

        for i in range(len(layer.parameters)):
            self.check_tensors_equality(
                layer.parameters_grads[i],
                expected_parameters_grads[i],
                test_data['meta'],
                f"From .backward, .parameters_grads[{i}]",
            )
